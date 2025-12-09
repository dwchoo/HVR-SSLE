import torch
import numpy as np

import pyiqa
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

#pip install pyiqa

import logging

"""
['ahiq', 'arniqa', 'arniqa-clive', 'arniqa-csiq', 'arniqa-flive', 'arniqa-kadid', 'arniqa-koniq', 'arniqa-live', 'arniqa-spaq', 'arniqa-tid', 'brisque', 'ckdn', 'clipiqa', 'clipiqa+', 'clipiqa+_rn50_512', 'clipiqa+_vitL14_512', 'clipscore', 'cnniqa', 'cw_ssim', 'dbcnn', 'dists', 'entropy', 'fid', 'fsim', 'gmsd', 'hyperiqa', 'ilniqe', 'inception_score', 'laion_aes', 'liqe', 'liqe_mix', 'lpips', 'lpips-vgg', 'mad', 'maniqa', 'maniqa-kadid', 'maniqa-koniq', 'maniqa-pipal', 'ms_ssim', 'musiq', 'musiq-ava', 'musiq-koniq', 'musiq-paq2piq', 'musiq-spaq', 'nima', 'nima-koniq', 'nima-spaq', 'nima-vgg16-ava', 'niqe', 'nlpd', 'nrqm', 'paq2piq', 'pi', 'pieapp', 'psnr', 'psnry', 'qalign', 'ssim', 'ssimc', 'stlpips', 'stlpips-vgg', 'topiq_fr', 'topiq_fr-pipal', 'topiq_iaa', 'topiq_iaa_res50', 'topiq_nr', 'topiq_nr-face', 'topiq_nr-flive', 'topiq_nr-spaq', 'tres', 'tres-flive', 'tres-koniq', 'unique', 'uranker', 'vif', 'vsi', 'wadiqam_fr', 'wadiqam_nr']
"""

# Image Quality Assessment (IQA) 기본 클래스
class IQA_pytorch_class:
    def __init__(self, device='cpu'):
        pyiqa.utils.get_root_logger(log_level=logging.ERROR)
        self.device = set_device(device)
        self.metric = None

    def input_data_check(self, x, y):
        # 텐서의 device가 동일한지 확인
        if not device_checker(x.device, y.device) or not device_checker(y.device, self.device):
            raise ValueError(f"Device of x or y is different from self.device, x:{x.device}, y:{y.device}, self.device:{self.device}")
        
        # 텐서의 shape이 동일한지 확인
        if x.shape != y.shape:
            raise ValueError(f"x and y shape are different, x: {x.shape}, y: {y.shape}")
        
        # BCHW 형식인지 확인 (4차원이어야 함)
        if len(x.shape) != 4:
            raise ValueError(f"Data must be BCHW format, x: {x.shape}, y: {y.shape}")
        
        # 채널 수가 3인지 확인 (RGB 이미지)
        if x.size(1) != 3:
            raise ValueError(f"Data must be 3 channel, but got x: {x.shape}, y: {y.shape}")
        
        # x와 y가 float 타입인지 확인
        if not torch.is_floating_point(x):
            raise TypeError(f"x must be float, but got {x.dtype}")
        if not torch.is_floating_point(y):
            raise TypeError(f"y must be float, but got {y.dtype}")
        
        return True

    def __call__(self, x, y):
        # 입력 데이터 검사
        _ = self.input_data_check(x, y)
        
        # PSNR 등 메트릭 계산
        scores = self.metric(x, y) if self.metric is not None else None
        
        return scores


# PSNR 계산 클래스 (IQA_pytorch_class 상속)
class Metric_PSNR(IQA_pytorch_class):
    def __init__(self, data_max=1.0, eps=1e-08, device='cpu'):
        super().__init__(device=device)
        
        self.metric = pyiqa.create_metric('psnr', as_loss=False, test_y_channel=True, data_range=data_max, eps=eps, device=self.device)


class Metric_SSIM(IQA_pytorch_class):
    def __init__(self, device='cpu'):
        super().__init__(device=device)
        channels=3
        self.metric = pyiqa.create_metric('ssim', as_loss=False, test_y_channel=True, channels=channels, device=self.device)

class Metric_NIQE(IQA_pytorch_class):
    def __init__(self, device='cpu'):
        super().__init__(device=device)
        channels=3
        self.metric = pyiqa.create_metric('niqe', as_loss=False, test_y_channel=True, device=self.device)

    def __call__(self, x):
        _ = self.input_data_check(x, x)
        
        scores = self.metric(x) 
        
        return scores


class Metric_PIQE(IQA_pytorch_class):
    def __init__(self, device='cpu'):
        super().__init__(device=device)
        self.metric = pyiqa.create_metric('piqe', as_loss=False, test_y_channel=True, device=self.device)

    def __call__(self, x):
        _ = self.input_data_check(x, x)
        return self.metric(x)


class Metric_BRISQUE(IQA_pytorch_class):
    def __init__(self, device='cpu'):
        super().__init__(device=device)
        self.metric = pyiqa.create_metric('brisque', as_loss=False, test_y_channel=True, device=self.device)

    def __call__(self, x):
        _ = self.input_data_check(x, x)
        return self.metric(x)

class Metric_LPIPS(IQA_pytorch_class):
    def __init__(self, device='cpu'):
        super().__init__(device=device)
        self.metric = pyiqa.create_metric('lpips', as_loss=False, test_y_channel=True, device=self.device)

    def __call__(self, x, y):
        _ = self.input_data_check(x, y)
        scores = self.metric(x, y, normalize=True)
        return scores



class scipy_metric:
    def metric(self,pred, target):
        return 0

    def __call__(self, pred, target):
        if not (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)):
            raise TypeError("pred and target must be numpy arrays")
            
        if len(pred.shape) == 3 and len(target.shape) == 3:
            return self.metric(pred, target)
        elif len(pred.shape) != 4 or len(target.shape) != 4:
            raise ValueError(f"pred and target must be BHWC arrays, but pred: {pred.shape} and target: {target.shape}")
        
        if pred.shape[-1] != 3 or target.shape[-1] != 3:
            raise ValueError(f"pred and target must have 3 channels, but pred: {pred.shape} and target: {target.shape}")
        
        result = np.array([])
        for _pred, _target in zip(pred, target):
            _result = self.metric(_pred, _target)
            result = np.append(result, _result)
        return result

                



class scipy_SSIM(scipy_metric):
    def __init__(self, data_range = 1.0, channel_axis = -1, win_size = 3):
        self.data_range = data_range
        self.channel_axis = channel_axis
        self.win_size = win_size

    def metric(self, pred, target):
        return ssim(pred, target, data_range=self.data_range, channel_axis=self.channel_axis, win_size=self.win_size)

class scipy_PSNR(scipy_metric):
    def __init__(self, data_range = 1.0):
        self.data_range = data_range

    def metric(self, pred, target):
        return psnr(
            image_true = target, 
            image_test = pred, 
            data_range=self.data_range
        )










def set_device(device='cpu'):
    if isinstance(device, torch.device):
            device = device
    else:
        try:
            device = torch.device(device)
        except:
            device = torch.device('cpu')
    return device



def device_checker(x_device, y_device):
    x_device_type, x_device_index = x_device.type, x_device.index
    y_device_type, y_device_index = y_device.type, y_device.index
    if (x_device_index == None) or (y_device_index == None):
        return x_device_type == y_device_type
    else:
        return (x_device_type == y_device_type) and (x_device_index == y_device_index)


