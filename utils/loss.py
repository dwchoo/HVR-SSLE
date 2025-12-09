import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np

import pyiqa

from icecream import ic


class Loss_pyiqa(nn.Module):
    def __init__(self, name, as_loss=True, device='cpu', **kwargs):
        super(Loss_pyiqa, self).__init__()
        self.metric = pyiqa.create_metric(name, as_loss = as_loss, device = device, **kwargs)


    def forward(self, x, target,**kwargs):
        return self.metric(x, target, **kwargs)


class Loss_lpips(nn.Module):
    def __init__(self, device='cpu', **kwargs):
        super(Loss_lpips, self).__init__()
        self.loss_func = Loss_pyiqa(name='lpips', as_loss=True, device=device, **kwargs)

    def forward(self, x, target, normalize=True):
        # If x is [0,1], then 'normalize' have to be 'True'.
        # But if x is [-1,1], 'normalize' have to be 'False'.
        return self.loss_func(x, target,normalize=normalize)
    
    def from_latent(self, x_latent, target_latent, decoder, normalize=False):
        x = decoder(x_latent, no_grad=False)
        target = decoder(target_latent, no_grad=False)
        return self.forward(x, target, normalize=normalize)


class Loss_hybrid(nn.Module):
    def __init__(self, device, accelerator=None, loss_weights=None, exposure_mean=0.4):
        super(Loss_hybrid, self).__init__()
        self.accelerator = accelerator
        if accelerator is not None:
            self.device = accelerator.device
        else:
            self.device = device

        default_weights = {
            'mse': 1.0,
            #'lpips': 1.0,
            #'exposure': 0.05,
            #'color': 1.0,
            #'spa': 0.2,
            #'sat': 0.5
            #satv : 0.01
        }
        self.loss_weights = loss_weights if loss_weights is not None else default_weights

        # --- Initialize ALL potential sub-networks with parameters here ---
        # This ensures they are registered as submodules and their parameters
        # are visible to `accelerate.prepare()`, which is crucial for DDP.
        if 'lpips' in self.loss_weights and self.loss_weights.get('lpips', 0) > 0:
            self.loss_lpips_func = Loss_lpips(device=self.device)

        # The following losses do not have trainable parameters, so their
        # initialization location is less critical, but __init__ is still good practice.
        if any(k in self.loss_weights for k in ['exposure', 'exposure_l2', 'color', 'spa', 'tv']):
            self.init_zero_ref_loss(exposure_mean=exposure_mean)

        # --- End of sub-network initialization ---

        self.track_loss = {}
        self.custom_track_loss = {}



    def weight_changer(self,l1_loss, threshold, loss_weights_org, loss_weights_new):
        _loss_weights_org = copy.deepcopy(loss_weights_org)
        _loss_weights_new = copy.deepcopy(loss_weights_new)
        if l1_loss < threshold:
            return _loss_weights_new, _loss_weights_org
        else:
            return _loss_weights_org, _loss_weights_new
            
    def init_zero_ref_loss(self, exposure_patch_size=16, exposure_mean=0.4, **kwargs):
        self.loss_exposure_func = ExposureControlLoss(
            patch_size=exposure_patch_size, 
            mean_val=exposure_mean
        )
        self.loss_exposure_l2_func = ExposureControlL2Loss(
            patch_size=exposure_patch_size, 
            mean_val=exposure_mean
        )
        self.loss_color_func = ColorConstancyLoss()
        self.loss_spa_func = SpatialConsistencyLoss()
        self.loss_TV = IlluminationSmoothnessLoss()


    def forward(
        self, 
        pred_image=None,
        target_image=None, 
        input_image = None, 
        input_zero_to_one = False,
    ):
        if input_zero_to_one:

            _pred_image   = pred_image   * 2 - 1. if pred_image is not None else None
            _target_image = target_image * 2 - 1. if target_image is not None else None
            _input_image  = input_image  * 2 - 1. if input_image is not None else None

            pred_image_clamp_one_to_one    = _pred_image.clamp(  min=-1, max=1.0) if _pred_image is not None else None
            target_image_clamp_one_to_one  = _target_image.clamp(min=-1, max=1.0) if _target_image is not None else None
            input_image_clamp_one_to_one   = _input_image.clamp( min=-1, max=1.0) if _input_image is not None else None
            pred_image_clamp_zero_to_one   = pred_image.clamp(  min=0., max=1.0) if pred_image is not None else None
            target_image_clamp_zero_to_one = target_image.clamp(min=0., max=1.0) if target_image is not None else None
            input_image_clamp_zero_to_one  = input_image.clamp( min=0., max=1.0) if input_image is not None else None

        else:
            _pred_image   = (pred_image + 1.)   / 2 if pred_image is not None else None
            _target_image = (target_image + 1.) / 2 if target_image is not None else None
            _input_image  = (input_image + 1.)  / 2 if input_image is not None else None

            pred_image_clamp_zero_to_one   = _pred_image.clamp(min=0, max=1.0) if _pred_image is not None else None
            target_image_clamp_zero_to_one = _target_image.clamp(min=0, max=1.0) if _target_image is not None else None
            input_image_clamp_zero_to_one  = _input_image.clamp(min=0, max=1.0) if _input_image is not None else None
            pred_image_clamp_one_to_one    = pred_image.clamp(min=-1, max=1.0) if pred_image is not None else None
            target_image_clamp_one_to_one  = target_image.clamp(min=-1, max=1.0) if target_image is not None else None
            input_image_clamp_one_to_one   = input_image.clamp(min=-1, max=1.0) if input_image is not None else None
            
        
        loss_dict = {}

        if pred_image is not None and target_image is not None:
            if 'mae' in self.loss_weights:
                loss_mae = F.l1_loss( pred_image, target_image)
                loss_dict['mae'] = loss_mae
            
            if 'mse' in self.loss_weights:
                loss_mse = F.mse_loss(pred_image, target_image)
                loss_dict['mse'] = loss_mse

            if 'lpips' in self.loss_weights and hasattr(self, 'loss_lpips_func'):
                loss_lpips = self.loss_lpips_func(pred_image_clamp_one_to_one, target_image_clamp_one_to_one, normalize=False)
                loss_dict['lpips'] = loss_lpips

            if 'ssim' in self.loss_weights:
                if not hasattr(self, 'loss_ssim_func'):
                    self.loss_ssim_func = Loss_pyiqa(
                        name='ssim', as_loss=True, 
                        channels=3, test_y_channel=True,
                        device=self.device,
                    )
                loss_value_ssim = self.loss_ssim_func(pred_image_clamp_zero_to_one, target_image_clamp_zero_to_one)
                loss_ssim = 1 - loss_value_ssim
                loss_dict['ssim'] = loss_ssim

            if 'psnr' in self.loss_weights:
                if not hasattr(self, 'loss_psnr_func'):
                    self.loss_psnr_func = Loss_pyiqa(
                        name='psnr', as_loss=True,
                        test_y_channel=False, 
                        data_range=1.0, eps=1e-8,
                        device=self.device
                    )
                loss_value_psnr = self.loss_psnr_func(pred_image_clamp_zero_to_one, target_image_clamp_zero_to_one)
                loss_psnr = 1 - loss_value_psnr/80
                loss_dict['psnr'] = loss_psnr

            if 'nlpd' in self.loss_weights:
                if not hasattr(self, 'loss_nlpd_func'):
                    self.loss_nlpd_func = Loss_pyiqa(
                        name = 'nlpd', as_loss=True,
                        channels = 3, test_y_channel=False,
                        device=self.device,
                    )
                loss_nlpd = self.loss_nlpd_func(pred_image_clamp_zero_to_one, target_image_clamp_zero_to_one)
                loss_dict['nlpd'] = loss_nlpd
           
            if 'sat' in self.loss_weights:
                if not hasattr(self, 'loss_sat_func'):
                    self.loss_sat_func = ChromaSatLoss(
                        lambda_sat=0.1, lambda_chroma=0.5, eps=1e-6
                    )
                loss_sat_lab, loss_sat, loss_chroma = self.loss_sat_func(
                    pred_image_clamp_zero_to_one, target_image_clamp_zero_to_one
                )
                loss_dict['sat'] = loss_sat_lab
            
        if pred_image_clamp_zero_to_one is not None:
            if 'exposure' in self.loss_weights and hasattr(self, 'loss_exposure_func'):
                loss_exposure = self.loss_exposure_func(pred_image_clamp_zero_to_one)
                loss_dict['exposure'] = loss_exposure
            
            if 'exposure_l2' in self.loss_weights and hasattr(self, 'loss_exposure_l2_func'):
                loss_exposure_l2 = self.loss_exposure_l2_func(pred_image_clamp_zero_to_one)
                loss_dict['exposure_l2'] = loss_exposure_l2

            if 'color' in self.loss_weights and hasattr(self, 'loss_color_func'):
                loss_color = self.loss_color_func(pred_image_clamp_zero_to_one)
                loss_dict['color'] = loss_color
            
            if 'tv' in self.loss_weights and hasattr(self, 'loss_TV'):
                loss_TV = self.loss_TV(pred_image_clamp_zero_to_one)
                loss_dict['tv'] = loss_TV

            if 'satv' in self.loss_weights:
                if not hasattr(self, 'loss_satv_func'):
                    self.loss_satv_func = StructureAwareSmoothnessRGB(
                        lambda_g=10.0, eps=1e-6
                    )
                loss_satv = self.loss_satv_func(pred_image_clamp_zero_to_one)
                loss_dict['satv'] = loss_satv

            if 'nrqm' in self.loss_weights:
                if not hasattr(self, 'loss_nrqm_func'):
                    self.loss_nrqm_func = Loss_pyiqa(
                        name = 'nrqm', as_loss=True,
                        test_y_channel=True,
                        device = self.device,
                    )
                with torch.cuda.amp.autocast(enabled=False):
                    loss_value_nrqm = self.loss_nrqm_func(pred_image_clamp_zero_to_one.float())
                    loss_nrqm = 1 - 0.1 * loss_value_nrqm
                    loss_dict['nrqm'] = loss_nrqm
            

        if pred_image_clamp_zero_to_one is not None and input_image_clamp_zero_to_one is not None:
            if 'spa' in self.loss_weights and hasattr(self, 'loss_spa_func'):
                loss_spa = self.loss_spa_func(
                    enhanced_image = pred_image_clamp_zero_to_one, 
                    input_image = input_image_clamp_zero_to_one,
                )
                loss_dict['spa'] = loss_spa
            

          


        if len(loss_dict) == 0:
            raise ValueError(f"loss was not calculated")
        # Combine losses with respective weights
        #total_loss = 0.0
        __sample_value = next(iter(loss_dict.values()), torch.tensor(0.0))
        total_loss = torch.zeros((), device = self.device, dtype=__sample_value.dtype)
        for key, value in loss_dict.items():
            weight = self.loss_weights.get(key, 1.0)  # Default weight is 1.0 if not specified
            total_loss += weight * value

        loss_dict['total'] = total_loss.clone().detach()

        # Append the loss values to track_loss lists dynamically
        for key, value in loss_dict.items():
            if key not in self.track_loss:
                self.track_loss[key] = []
            self.track_loss[key].append(value.item())

        return total_loss, loss_dict

    def add_value_to_track(self, dict_value, save_custom = True):
        assert isinstance(dict_value, dict), f"input must be a dictionary, but {type(dict_value)}"
        if save_custom:
            _track_loss = self.custom_track_loss
        else:
            _track_loss = self.track_loss
        for key, value in dict_value.items():
            if key not in _track_loss:
                _track_loss[key] = []
            _track_loss[key].append(value.item())

    def compute_mean_losses(self, track_loss = None,return_tensor=False):
        """Compute the mean of each tracked loss component."""
        if track_loss is None:
            track_loss = self.track_loss
        mean_losses = {key: np.mean(values) if values else 0.0 for key, values in track_loss.items()}
        self.sorted_key = sorted(list(mean_losses.keys()))
        sorted_value = np.array([mean_losses[_key] for _key in self.sorted_key])

        if return_tensor:
            mean_losses = {
                key: torch.tensor(value).to(self.device) for key, value in mean_losses.items()
            }
            sorted_value = torch.from_numpy(sorted_value).to(self.device)
            
        
        return mean_losses, sorted_value

    def reset_track_loss(self):
        """Reset the tracked loss lists."""
        self.track_loss.clear()
        self.custom_track_loss.clear()


    def seperate_metrics(self, data, metrics=None):
        if metrics is None:
            metrics = self.sorted_key
            
        separated = {metric: [] for metric in metrics}
        step = len(metrics)
        for i, value in enumerate(data):
            metric = metrics[i % step]
            separated[metric].append(value)
        
        mean_loss_dict = {key: np.mean(values) if values else 0.0 for key, values in separated.items()}
        
        return mean_loss_dict



class WeightTotalLoss(Loss_hybrid):
    def __init__(self, device, accelerator=None):
        super(WeightTotalLoss, self).__init__(device, accelerator)
        self.accelerator = accelerator
        if accelerator is not None:
            self.device = accelerator.device
        else:
            self.device = device

        self.loss_weights = {}

        self.track_loss = {}

    
    def forward(self, input_loss_dict):
        loss_dict = {}
        
        for _key, _value in input_loss_dict.items():
            _loss_name   = _key
            _loss_value  = _value.get('loss', None)
            _loss_weight = _value.get('weight', None)
            
            if all(x is not None for x in (_loss_value, _loss_weight)):
                loss_dict[str(_loss_name)] = _loss_value
                self.loss_weights[str(_loss_name)] = _loss_weight

        # Combine losses with respective weights
        total_loss = 0.0
        for key, value in loss_dict.items():
            weight = self.loss_weights.get(key, 1.0)  # Default weight is 1.0 if not specified
            total_loss += weight * value

    
        loss_dict['dis_total'] = total_loss.clone().detach()

        # Append the loss values to track_loss lists dynamically
        for key, value in loss_dict.items():
            if key not in self.track_loss:
                self.track_loss[key] = []
            self.track_loss[key].append(value.item())

        return total_loss, loss_dict

    @staticmethod
    def binary_accuracy(pred, label, threshold=0.5):
        # ex) pred > 0.5 => 1, else 0
        pred_prob = torch.sigmoid(pred)
        pred_label = (pred_prob >= threshold).float()
        correct = (pred_label == label).float().sum()
        total = label.numel()
        acc = correct / total
        return acc




# ===================================================================
# 1. Exposure Control Loss (L_exp)
# - Controls the exposure level of the enhanced image.
# - It penalizes images that are too bright or too dark.
# ===================================================================
class ExposureControlLoss(nn.Module):
    """
    Calculates the exposure control loss.

    This loss encourages the average intensity of local patches in the enhanced image
    to be close to a well-exposed level (E), typically set to 0.6. It is based on
    the L1 distance to be robust against outliers.

    Args:
        patch_size (int): The size of the local patches to average over.
        mean_val (float): The target well-exposedness level.
    """
    def __init__(self, patch_size=16, mean_val=0.4):
        super(ExposureControlLoss, self).__init__()
        self.patch_size = patch_size
        self.mean_val = mean_val
        # An average pooling layer to efficiently calculate the mean of local patches.
        self.pool = nn.AvgPool2d(self.patch_size)

    def forward(self, enhanced_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enhanced_image (torch.Tensor): The enhanced image tensor (B, C, H, W).
                                           Pixel values are expected to be in the [0, 1] range.
        """
        # Calculate luminance by averaging the RGB channels. Shape: (B, 1, H, W)
        luminance = enhanced_image.mean(dim=1, keepdim=True)
        
        # Calculate the average luminance for each local patch.
        avg_luminance = self.pool(luminance)
        
        # Calculate the L1 distance from the target exposure level E.
        # Using torch.full_like ensures the target tensor has the same shape and device.
        loss = F.l1_loss(avg_luminance, torch.full_like(avg_luminance, self.mean_val))
        return loss

class ExposureControlL2Loss(nn.Module):
    """
    Calculates the exposure control loss.

    This loss encourages the average intensity of local patches in the enhanced image
    to be close to a well-exposed level (E), typically set to 0.6. It is based on
    the L1 distance to be robust against outliers.

    Args:
        patch_size (int): The size of the local patches to average over.
        mean_val (float): The target well-exposedness level.
    """
    def __init__(self, patch_size=16, mean_val=0.4):
        super(ExposureControlL2Loss, self).__init__()
        self.patch_size = patch_size
        self.mean_val = mean_val
        # An average pooling layer to efficiently calculate the mean of local patches.
        self.pool = nn.AvgPool2d(self.patch_size)

    def forward(self, enhanced_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enhanced_image (torch.Tensor): The enhanced image tensor (B, C, H, W).
                                           Pixel values are expected to be in the [0, 1] range.
        """
        # Calculate luminance by averaging the RGB channels. Shape: (B, 1, H, W)
        luminance = enhanced_image.mean(dim=1, keepdim=True)
        
        # Calculate the average luminance for each local patch.
        avg_luminance = self.pool(luminance)
        
        # Calculate the L2 distance from the target exposure level E.
        # Using torch.full_like ensures the target tensor has the same shape and device.
        loss = F.mse_loss(avg_luminance, torch.full_like(avg_luminance, self.mean_val))
        return loss

# ===================================================================
# 2. Color Constancy Loss (L_col)
# - Prevents color shifts and preserves color balance.
# - Based on the Gray-World hypothesis.
# ===================================================================
class ColorConstancyLoss(nn.Module):
    """
    Calculates the color constancy loss.

    This loss is based on the Gray-World hypothesis, which assumes that the average
    color in an image should be gray. It penalizes the deviation of the mean values
    of the three color channels (R, G, B) from each other.
    """
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, enhanced_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enhanced_image (torch.Tensor): The enhanced image tensor (B, 3, H, W).
        """
        # Calculate the mean of each RGB channel across all spatial locations. Shape: (B, 3)
        mean_rgb = torch.mean(enhanced_image, dim=(2, 3), keepdim=False)
        
        # Split the channels.
        mr, mg, mb = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        
        # Calculate the pairwise squared differences between channel means.
        d_rg = (mr - mg)**2
        d_rb = (mr - mb)**2
        d_gb = (mg - mb)**2
        
        # The loss is the mean of the sum of these differences across the batch.
        loss = torch.mean(d_rg + d_rb + d_gb)
        return loss

# ===================================================================
# 3. Spatial Consistency Loss (L_spa)
# - Preserves the spatial structure and contrast of the original image.
# - Enforces that the gradients (local contrast) are similar.
# ===================================================================
class SpatialConsistencyLoss(nn.Module):
    """
    Calculates the spatial consistency loss.

    This loss encourages the spatial structure of the enhanced image to be consistent
    with the original input image. It achieves this by comparing their high-frequency
    components, extracted using a fixed Laplacian kernel.
    """
    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()
        # A fixed 3x3 Laplacian kernel to extract second-order gradients (edges/texture).
        kernel_data = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32)
        
        # Reshape for 2D convolution: (out_channels, in_channels, H, W)
        self.kernel = kernel_data.view(1, 1, 3, 3)

    def forward(self, enhanced_image: torch.Tensor, input_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enhanced_image (torch.Tensor): The enhanced image tensor (B, C, H, W).
            input_image (torch.Tensor): The original dark input image tensor (B, C, H, W).
        """
        # Ensure the kernel is on the same device as the images.
        # This is done here to avoid device mismatch issues.
        kernel = self.kernel.to(enhanced_image.device)
        
        # Convert images to grayscale to focus on structural information (luminance).
        enhanced_gray = enhanced_image.mean(dim=1, keepdim=True)
        input_gray = input_image.mean(dim=1, keepdim=True)
        
        # Apply the Laplacian filter to both images to get their feature maps.
        # 'padding="same"' ensures the output has the same spatial dimensions.
        enhanced_lap = F.conv2d(enhanced_gray, kernel, padding='same')
        input_lap = F.conv2d(input_gray, kernel, padding='same')
        
        # The loss is the L1 distance between the structural feature maps.
        loss = F.l1_loss(enhanced_lap, input_lap)
        return loss


class IlluminationSmoothnessLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(IlluminationSmoothnessLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size





class ChromaSatLoss(nn.Module):
    def __init__(self, lambda_sat=0.1, lambda_chroma=0.5, eps=1e-6):
        """
        Loss module for comparing output image vs ground-truth image:
        - Saturation loss in HSV-like space (S ≈ (max-min)/max)
        - Chroma (color difference) loss in YCbCr-like space (Cb, Cr channels)
        
        Args:
            lambda_sat (float): weight for saturation difference loss
            lambda_chroma (float): weight for chroma (Cb/Cr) difference loss
            eps (float): small constant for numerical stability
        """
        super(ChromaSatLoss, self).__init__()
        self.lambda_sat = lambda_sat
        self.lambda_chroma = lambda_chroma
        self.eps = eps

    def rgb_to_saturation(self, img):
        """
        Estimate saturation S = (max(R,G,B) − min(R,G,B)) / (max(R,G,B) + eps)
        
        Args:
            img: tensor of shape (B,3,H,W), values assumed in [0,1]
        Returns:
            S: tensor of shape (B,1,H,W)
        """
        maxc, _ = img.max(dim=1, keepdim=True)   # (B,1,H,W)
        minc, _ = img.min(dim=1, keepdim=True)
        S = (maxc - minc) / (maxc + self.eps)
        return S

    def rgb_to_ycbcr_chroma(self, img):
        """
        Compute an approximate Y (luma) and chroma-like channels (Cb, Cr) from RGB.
        We only return the chroma components (Cb, Cr) for color-difference loss.
        
        Args:
            img: tensor (B,3,H,W) in [0,1], assumed sRGB
        
        Returns:
            cb: tensor (B,1,H,W)
            cr: tensor (B,1,H,W)
        """
        # Split channels
        r = img[:, 0:1, :, :]
        g = img[:, 1:2, :, :]
        b = img[:, 2:3, :, :]

        # Approximate luma (Rec.601-like coefficients)
        y = 0.299 * r + 0.587 * g + 0.114 * b  # (B,1,H,W)

        # Chroma components as simple color differences (blue−luma, red−luma)
        # Scale is arbitrary for our purposes since we only use differences.
        cb = b - y
        cr = r - y

        return cb, cr

    def forward(self, out_rgb, gt_rgb):
        """
        Compute the combined saturation + chroma loss.
        
        Args:
            out_rgb: tensor (B,3,H,W) — model output image, values in [0,1]
            gt_rgb:  tensor (B,3,H,W) — ground-truth image, values in [0,1]
        
        Returns:
            loss      (scalar tensor): total weighted loss
            loss_sat  (scalar tensor): saturation part
            loss_chr  (scalar tensor): chroma (Cb/Cr) part
        """
        # 1) Saturation loss (HSV-like, only S channel)
        S_out = self.rgb_to_saturation(out_rgb)
        S_gt  = self.rgb_to_saturation(gt_rgb)
        loss_sat = torch.mean(torch.abs(S_out - S_gt))

        # 2) Chroma loss in YCbCr-like space (only Cb, Cr)
        cb_out, cr_out = self.rgb_to_ycbcr_chroma(out_rgb)
        cb_gt,  cr_gt  = self.rgb_to_ycbcr_chroma(gt_rgb)
        loss_chroma = torch.mean(torch.abs(cb_out - cb_gt) + torch.abs(cr_out - cr_gt))

        # 3) Combined weighted loss
        loss = self.lambda_sat * loss_sat + self.lambda_chroma * loss_chroma
        return loss, loss_sat, loss_chroma





class StructureAwareSmoothnessRGB(nn.Module):
    def __init__(self, lambda_g=10.0, eps=1e-6):
        super().__init__()
        self.lambda_g = lambda_g
        self.eps = eps

    def gradient_x(self, img):
        return img[:, :, :, 1:] - img[:, :, :, :-1]

    def gradient_y(self, img):
        return img[:, :, 1:, :] - img[:, :, :-1, :]

    def rgb_to_luminance(self, rgb):
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        Y = 0.299 * r + 0.587 * g + 0.114 * b
        return Y

    def forward(self, rgb):
        # 1) luminance as I
        I = self.rgb_to_luminance(rgb)

        # 2) reflectance surrogate as R
        R = I

        # 3) gradients
        Ix = self.gradient_x(I)
        Iy = self.gradient_y(I)

        Rx = self.gradient_x(R)
        Ry = self.gradient_y(R)

        # 4) structure-aware weights (NO sum between Rx and Ry)
        weight_x = torch.exp(-self.lambda_g * torch.abs(Rx))
        weight_y = torch.exp(-self.lambda_g * torch.abs(Ry))

        # 5) SA-TV loss
        loss = (weight_x * torch.abs(Ix)).mean() + \
               (weight_y * torch.abs(Iy)).mean()

        return loss


