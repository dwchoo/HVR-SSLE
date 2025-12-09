import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
#from functools import partial
from scipy.stats import truncnorm


import os
import sys
import random
import copy
import math
from pycocotools.coco import COCO
import ujson
from typing import Union, Optional

from PIL import Image

from icecream import ic




class COCODataset(Dataset):
    def __init__(self, image_dir, annotation_file=None, transform=None, width=256, height=256, 
                 image_ids=None, coco_instance=None, 
                 mask_transform=None, preprocessing_func: Optional["image_normalization_func"]=None,
                 blur_transform=False, blur_rate=0.5,
                 fill_noise=None, noise_rate=1e-2,
                 image_preprocessing_config: Optional[dict]=None,
                 shared_image_preprocessing_config: Optional[dict]=None,
                 seed=None):
        self.image_dir = image_dir
        if preprocessing_func is None:
            self.preprocessing_func = image_normalization_func()
        else:
            self.preprocessing_func = preprocessing_func
        self.shared_cfg = shared_image_preprocessing_config

        # image preprocessing config
        alpha_config = {'min_val':0.0, 'max_val':0.4, 'precision':3}
        k_config = {'min_val':0.05, 'max_val':0.3, 'mean':-0.1, 'precision':3}
        l_config = {'min_val':0.4, 'max_val':0.95, 'mean':0.7, 'precision':3}
        if image_preprocessing_config is None:
            self.image_preprocessing_config = dict(
                alpha = alpha_config,
                k = k_config,
                l = l_config,
            )
        else:
            self.image_preprocessing_config = image_preprocessing_config
        self.distributions = {}
        for _dist_name in ['k', 'l']:
            _config_dict = self.image_preprocessing_config[_dist_name]
            min_val = _config_dict['min_val']
            max_val = _config_dict['max_val']
            mean = _config_dict['mean']
            std_dev = (max_val - min_val) / 4
            a = (min_val - mean) / std_dev
            b = (max_val - mean) / std_dev
            dist_obj = truncnorm(a, b, loc=mean, scale=std_dev)
            self.distributions[_dist_name] = dist_obj
        
        # COCO 인스턴스 설정
        if coco_instance is not None:
            self.coco = coco_instance
        elif annotation_file is not None:
            self.coco = COCO(annotation_file)
        else:
            raise ValueError("Either annotation_file or coco_instance must be provided")
            
        # 이미지 ID 설정
        _image_ids = image_ids if image_ids is not None else self.coco.getImgIds()
        self.seed = seed
        self.image_ids = _image_ids.tolist() if isinstance(_image_ids, np.ndarray) else _image_ids
        
        self.transform = transform
        if blur_transform:
            self.blur_transform = Blur_transform(img_height=height, img_width=width, scale=2,p=blur_rate)
        else:
            self.blur_transform = None
        self.width = width
        self.height = height

        # 0~8 사이의 정수 값을 데이터셋 길이만큼 uniform하게 생성하여 리스트로 저장
        #self.random_change_type_list = np.random.randint(0, 9, size=len(self.image_ids))

        self.mask_transform = mask_transform
        self.noise_transform = None
        if fill_noise:
            if isinstance(fill_noise, bool):
                self.noise_transform = FillNoise_transform(noise_rate=noise_rate,p=1.0)
            elif isinstance(fill_noise, float):
                self.noise_transform = FillNoise_transform(noise_rate=noise_rate,p=fill_noise)
            else:
                self.noise_transform = None

        if self.preprocessing_func:
            _dummy_tensor = torch.arange(0, 256, dtype=torch.uint8)
            self.norm_min_value = self.preprocessing_func.norm_target(_dummy_tensor).min()
            print("##################################################################")
            print(f"Normalized MIN value : {self.norm_min_value}")
            ic(f"Normalized MIN value : {self.norm_min_value}")
            print("##################################################################")
        else:
            self.norm_min_value = 0.

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        #change_type = self.random_change_type_list[index]
        #change_type = np.random.randint(3,6) #3~5, N->D, N->N, N->O
        change_type = 3 #N->D
        #change_type_case = [3,3,5]
        #change_type = chnage_type_case[np.random.randint(0,len(change_type_case)) #3~5, N->D, N->N, N->O

        # Load image
        #image = Image.open(image_path).convert("RGB")
        #image = np.array(image)
        # Load image
        with Image.open(image_path) as img:
            image = np.array(img.convert("RGB"))

       
        if self.transform:
            augmented = self.transform(image=image,)
            image = augmented['image']
        annotations = {
            'img_id' : image_id,
        }
        
        normalized_image_zero_one = image / 255.
        # 공유 config가 있으면 그 값을, 없으면 기본 config를 사용
        if self.shared_cfg is not None:
            current_cfg = copy.deepcopy(dict(self.shared_cfg))
        else:
            current_cfg = self.image_preprocessing_config

        if self.shared_cfg is not None:
            alpha_val = sample_float(**current_cfg['alpha'])
            k_val = gaussian_sample_in_range(**current_cfg['k'])
            l_val = gaussian_sample_in_range(**current_cfg['l'])
        else:
            alpha_val = None
            k_val = self.distributions['k'].rvs()
            l_val = self.distributions['l'].rvs()

        preprocessed_data = image_preprocessing(
            org_image = normalized_image_zero_one,
            change_type = change_type,
            image_preprocessing_config = current_cfg,
            k = k_val,
            l = l_val,
            alpha = alpha_val,
        )

        input_image   = preprocessed_data['input_image']
        target_image  = preprocessed_data['target_image']
        info = preprocessed_data['info']

        if self.blur_transform:
            input_image = self.blur_transform(image=input_image)['image']
        if self.mask_transform:
            input_image = self.mask_transform(image=input_image)['image']
        if self.noise_transform:
            input_image = self.noise_transform(image = input_image)['image']
        
        input_image_tensor = torch.from_numpy(input_image).permute(2, 0, 1).float()
        target_image_tensor = torch.from_numpy(target_image).permute(2, 0, 1).float()

        input_image_tensor = QuantizeNormData(
            float_data       = input_image_tensor,
            normalize_func   = self.preprocessing_func.norm_input,
            denormalize_func = None,
        )
        target_image_tensor = QuantizeNormData(
            float_data       = target_image_tensor,
            normalize_func   = self.preprocessing_func.norm_target,
            denormalize_func = None,
        )
        
        normalized_image = self.preprocessing_func.norm_target(image)

        image_info = {
            'org_image' : normalized_image,
            'input_image' : input_image,
            'target_image' : target_image,
            'annotations' : annotations,
            'image_id' : image_id,
            'preprocessing_info' : info,
        }
        #image_info.update(info)

        return {
            'input' : input_image_tensor,
            'target' : target_image_tensor,
            'image_info' : image_info,
        }


    @staticmethod
    def collate_fn(batch):
        inputs = torch.stack([item['input'] for item in batch])
        targets = torch.stack([item['target'] for item in batch])
        image_infos = [item['image_info'] for item in batch]
        
        return {
            'input': inputs,
            'target': targets,
            'image_info': image_infos
        } 


    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        np.random.seed(worker_info.seed % (2**32 - 1))
        random.seed(worker_info.seed % (2**32 - 1))



class LOLDataset(Dataset):
    def __init__(self, ll_image_dir, target_image_dir, 
                 preprocessing_func: Optional["image_normalization_func"]=None,
                 transform=None,
                 fill_noise=False, noise_rate=1e-2):
        self.ll_image_dir = ll_image_dir
        self.target_image_dir = target_image_dir
        if preprocessing_func is None:
            self.preprocessing_func = image_normalization_func()
        else:
            self.preprocessing_func = preprocessing_func
        self.transform = transform
        
        self.ll_image_list = get_image_paths_from_folders(folder=self.ll_image_dir)
        self.target_image_list = get_image_paths_from_folders(folder=self.target_image_dir)

        # N -> D, This dataset is low light image
        self.change_type = 3
        self.attn_type = -1
        self.info_dict = dict(
            change_type = self.change_type,
            alpha = 0,
            k = 0,
            l = 0,
            twin_alpha = 0,
            attn_type = self.attn_type,
        )
        self.fill_noise = fill_noise
        self.noise_rate = noise_rate

    def __len__(self):
        return len(self.ll_image_list)

    def __getitem__(self, index):
        with Image.open(self.ll_image_list[index]) as img:
            ll_image_array = np.array(img.convert('RGB'))
        with Image.open(self.target_image_list[index]) as img:
            target_image_array = np.array(img.convert('RGB'))

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=ll_image_array, mask=target_image_array)
            ll_image_array = augmented['image']
            target_image_array = augmented['mask']

        ll_image_array_zero_one = ll_image_array / 255.0
        target_image_array_zero_one = target_image_array / 255.0

        if self.fill_noise:
            ll_image_array_zero_one = FillNoise(ll_image_array_zero_one, noise_rate=self.noise_rate)
            
        input_image_tensor = torch.from_numpy(ll_image_array_zero_one).permute(2, 0, 1).float()
        target_image_tensor = torch.from_numpy(target_image_array_zero_one).permute(2, 0, 1).float()

        input_image_tensor = QuantizeNormData(
            float_data       = input_image_tensor,
            normalize_func   = self.preprocessing_func.norm_input,
            denormalize_func = None,
        )

        target_image_tensor = QuantizeNormData(
            float_data= target_image_tensor,
            normalize_func=self.preprocessing_func.norm_target,
            denormalize_func=None,
        )
        info = self.info_dict

        image_info = dict(
            org_image = ll_image_array,
            input_image = ll_image_array_zero_one,
            target_image = target_image_array_zero_one,
            **info,
        )


        return {
            'input' : input_image_tensor,
            'target' : target_image_tensor,
            'image_info' : image_info,
        }

    @staticmethod
    def collate_fn(batch):
        inputs = torch.stack([item['input'] for item in batch])
        targets = torch.stack([item['target'] for item in batch])
        image_infos = [item['image_info'] for item in batch]
        
        return {
            'input': inputs,
            'target': targets,
            'image_info': image_infos
        } 




# 데이터 증강 파이프라인 정의
def coco_transform(img_height=640, img_width=640):
    coco_transform = A.Compose([
        A.OneOf([
            A.Rotate(p=1.0, limit=(-45, 45), crop_border=True),
            A.HorizontalFlip(p=1.0,),
            #A.VerticalFlip(always_apply=True),
        ], p=0.4),

        DynamicRandomResizedCrop(
            target_height=img_height,
            target_width=img_width,
            lower_scaler=1.5,
            upper_scaler=1.5,
            ratio=(0.75, 1.33),
            interpolation=4,
            p=1.0,
        ),
    ],)
    return coco_transform

def image_transform(img_height=640, img_width=640):
    transform = A.Compose([
        # Apply either rotation or horizontal flip with a probability of 20%.
        A.OneOf([
            # Rotate the image by a random angle between -90 and 90 degrees.
            # crop_border=True ensures that parts of the image rotated outside
            # the frame are cropped.
            A.Rotate(p=1.0, limit=(-90, 90), crop_border=True),
            # Horizontally flip the image.
            A.HorizontalFlip(p=1.0,),
            # A.VerticalFlip(always_apply=True),  # Add vertical flip if needed.
        ], p=0.4),

        # Randomly crop and resize the image.
        A.RandomResizedCrop(
            height=img_height,
            width=img_width,
            scale=(0.5, 1.0),  # Crop to a random size between 50% and 100% of the original.
            ratio=(0.75, 1.33), # Allow aspect ratios between 3:4 and 4:3.
            interpolation= 4, #Flag 4 : cv2.INTER_LANCZOS4,  # Use Lanczos interpolation for better quality (optional).
            p=1.0  # Always apply this transformation.
        ),
        
    ])
    return transform

def Blur_transform(img_height=640, img_width=640, scale=2, p=0.5):
    transform = A.Compose([
        
        #Apply one of the blur transformations with a probability of 50%.
        A.OneOf([
            # Apply Gaussian blur with a random kernel size between 3 and 4,
            # and a sigma between 0.1 and 2.0.
            #A.GaussianBlur(blur_limit=(1, 3), sigma_limit=(0.1, 2.0), p=1.0,),
            
            # Apply defocus blur with a random radius between 3 and 5,
            # and an alias blur between 0.1 and 0.3.
            A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.3), p=1.0,),
            A.Compose([
                A.Resize(height=img_height//scale, width=img_width//scale, interpolation=2),
                A.Resize(height=img_height, width=img_width, interpolation=2),
            ]),
        ], p=p),
        #GaussNoise
        #A.GaussNoise(
        #    std_range=(0.05, 0.4),  # tuple[float, float]
        #    per_channel=True,  # bool
        #    noise_scale_factor=1,  # float
        #    p=1.0,  # float
        #),
        
    ])
    return transform


def test_transform(img_height=640, img_width=640,):

    transform = A.Compose([
        A.Resize(height=img_height, width=img_width, interpolation=2)

    ])
    
    return transform

    



def apply_brightness_reduction(image: np.ndarray, k: float, l: float, alpha: float, num_iterations: int=1) -> np.ndarray:
    x = image.copy()
    #exponent = 1 / (1 - alpha + 1e-6)을 사용해도 되지만 0.7~1.0 사이에 급격하게 변화하므로 rescale함
    exponent = 1 / (1 - np.power(alpha,0.2) + 1e-6)
    # 계수 C 계산
    C_denominator = np.power(1 - k, exponent)
    C_denominator = np.maximum(C_denominator, 1e-6)  # 분모가 0이 되는 것을 방지
    C = (1 - l) / C_denominator
    for _ in range(num_iterations):
        x_minus_k = x - k
        x_minus_k = np.maximum(x_minus_k, 0)  # x - k가 음수가 되는 것을 방지
        y = C * np.power(x_minus_k, exponent)
        y = np.clip(y, 0, 1)  # 픽셀 값이 0 ~ 1 사이로 유지되도록 클리핑
        x = y  # 다음 반복을 위해 x 업데이트
    return y

class DynamicRandomResizedCrop(A.DualTransform):
    """
    Dynamically computes the scale range for RandomResizedCrop based on the original image size.
    For example, if the original image is 640x640 and the target size is 128x128,
    then the scale range becomes (128/640/4, 128/640*2), i.e. (0.05, 0.4).

    This transform applies both to images and corresponding bounding boxes.
    """
    def __init__(self, target_height, target_width, lower_scaler=2., upper_scaler=2., 
                 ratio=(0.75, 1.33), interpolation=4, always_apply=True, p=1.0):
        super().__init__(always_apply, p=p)
        assert lower_scaler > 0.99999, f"Lower scaler has to be bigger than 1, but {lower_scaler}"
        assert upper_scaler > 0.99999, f"Upper scaler has to be bigger than 1, but {upper_scaler}"
        
        self.target_height = target_height
        self.target_width = target_width
        self.lower_scaler = lower_scaler
        self.upper_scaler = upper_scaler
        self.ratio = ratio
        self.interpolation = interpolation

        

    def __call__(self, force_apply=False, **data):
        image = data["image"]
        orig_height, orig_width = image.shape[:2]
        if orig_height >= orig_width:
            flag = orig_height
            target_flag = self.target_height
        else:
            flag = orig_width
            target_flag = self.target_width

        scale_lower_threshold = 0.1
        scale_upper_threshold = 1.0
        
        # Compute dynamic scale bounds based on the current image size.
        scale_value_1 = min(max( 
            (target_flag / flag) / self.lower_scaler, 
            scale_lower_threshold), scale_upper_threshold)
        scale_value_2 = min(max(
            (target_flag / flag) * self.upper_scaler, 
            scale_lower_threshold), scale_upper_threshold)
        scale_lower, scale_upper = sorted([scale_value_1, scale_value_2])

        # Create a dynamic RandomResizedCrop transform with computed scale bounds.
        dynamic_crop = A.RandomResizedCrop(
            size=(self.target_height,self.target_width),
            scale=(scale_lower, scale_upper),
            ratio=self.ratio,
            interpolation=self.interpolation,
            mask_interpolation=0,  # 혹은 별도의 값 지정
            p=1.0,
        )
        # Apply the same dynamic_crop to the entire data dict (image, bboxes, etc.)
        return dynamic_crop(force_apply=force_apply, **data)

    def get_transform_init_args_names(self):
        return ("target_height", "target_width", "ratio", "interpolation")







# D: dark, N: Normal, O: Over-exposed
# 0: D->D, 1: D->N, 2: D->O
# 3: N->D, 4: N->N, 5: N->O
# 6: O->D, 7: O->N, 8: O->O
def image_preprocessing(org_image: np.ndarray, change_type: int, image_preprocessing_config: dict, 
                        k = None, l = None, alpha = None,
                        #mask_transform=None, blur_transform = None, Noise_transform=None,
                        ):
    if not isinstance(org_image, np.ndarray):
        raise TypeError(f"org_image type must be np.ndarray, but {type(org_image)}")
    if not isinstance(change_type, (int, np.integer)):
        raise TypeError(f"change_tpe must be int, but {type(change_type)}")
    
    
    alpha_config = image_preprocessing_config.get('alpha')
    k_config = image_preprocessing_config.get('k')
    l_config = image_preprocessing_config.get('l')

    assert alpha_config is not None, "alpha_config is None"
    assert k_config is not None, "k_config is None"
    assert l_config is not None, "l_config is None"

    #alpha = sample_float(min_val=0.00, max_val=0.4, precision=3)
    #k = gaussian_sample_in_range(min_val=0.05, max_val=0.3, mean=-0.1, precision=3)   # Low zero
    #l = gaussian_sample_in_range(min_val=0.2, max_val=0.95, mean=0.7, precision=3)   # High zero

    if alpha is None:
        alpha = sample_float(**alpha_config)
    if k is None:
        k = gaussian_sample_in_range(**k_config)   # Low zero
    if l is None:
        l = gaussian_sample_in_range(**l_config)   # High zero

    

    D_image = apply_brightness_reduction(
        image = org_image,
        k = k, l = l,
        alpha=alpha,
    )
    N_image = org_image

    # N -> D
    if change_type == 3:
        target_image = N_image
        input_image = D_image
        attn_type = -1

    else:
        raise ValueError(f"Check change_type, {change_type} is not in [0~8]")


    input_image = QuantizeNormData(
        float_data = input_image,
        normalize_func = None,
        denormalize_func = None,
    )
    target_image = QuantizeNormData(
        float_data = target_image,
        normalize_func = None,
        denormalize_func = None,
    )
    
    info_dict = dict(
        change_type = change_type,
        alpha = alpha,
        k = k,
        l = l,
        preprocessing_config = dict(alpha = alpha_config, k = k_config, l = l_config)
    )
    return dict(
        target_image = target_image, 
        input_image = input_image, 
        info=info_dict,
    )

def sample_float(min_val, max_val, precision=3):
    scale = 10 ** precision  # 소수점 자리수에 따른 스케일 설정
    int_min = int(min_val * scale)
    int_max = int(max_val * scale)
    random_int = np.random.randint(int_min, int_max)  # 정수 값 선택
    return random_int / scale  # 선택한 정수를 다시 나누어 소수점 값 생성


def gaussian_sample_in_range(min_val, max_val, mean, precision=3):
    std_dev = (max_val - min_val) / 4
    # Standard deviation cannot be zero
    if std_dev == 0:
        return round(mean, precision)
        
    # Calculate bounds in terms of standard deviations
    a, b = (min_val - mean) / std_dev, (max_val - mean) / std_dev
    
    # Generate a single sample from the truncated distribution
    sample = truncnorm.rvs(a, b, loc=mean, scale=std_dev)
    return round(sample, precision)


def get_image_paths_from_folders(folder, extensions=('.png', '.jpg', '.jpeg', '.bmp')):
    """
    주어진 폴더들에서 특정 확장자의 이미지 파일 경로 리스트를 가져오는 함수
    
    Args:
        folder (list): 이미지 파일을 찾을 폴더 경로
        extensions (tuple): 가져올 파일 확장자들 (기본값은 .png, .jpg, .jpeg)
    
    Returns:
        list: 이미지 파일 경로 리스트
    """
    image_paths = []

    image_paths += [os.path.join(folder, fname) for fname in os.listdir(folder) 
                    if fname.endswith(extensions)]
    image_paths_sorted = sorted(image_paths, key=lambda x: os.path.basename(x).lower())
    return image_paths_sorted


def sigmoid_interp(start, end, epoch, total_epochs, steepness=6.0, center_frac=0.5):
    """Sigmoid-based interpolation from start to end over epochs."""
    t = epoch / max(total_epochs - 1, 1)
    c = center_frac
    s = 1.0 / (1.0 + math.exp(-steepness * (t - c)))
    s0 = 1.0 / (1.0 + math.exp(-steepness * (0 - c)))
    s1 = 1.0 / (1.0 + math.exp(-steepness * (1 - c)))
    w = (s - s0) / (s1 - s0 + 1e-8)
    return start + (end - start) * w


def linear_interp(start, end, epoch, total_epochs):
    """Linear interpolation from start to end over epochs."""
    t = epoch / max(total_epochs - 1, 1)
    return start + (end - start) * t


def resolve_pp_cfg(config, epoch):
    """
    Create image preprocessing config for the given epoch using sigmoid schedule.
    """
    base = copy.deepcopy(config.image_preprocessing_config)
    sched = getattr(config, "image_preprocessing_schedule", None)
    if not sched:
        return base

    start_cfg = sched.get("start", base)
    end_cfg = sched.get("end", start_cfg)
    steepness = sched.get("steepness", 6.0)
    center_frac = sched.get("center_frac", 0.5)
    mode = sched.get("mode", "sigmoid")
    total_ep = config.num_epochs

    out = {}
    for name in ("alpha", "k", "l"):
        base_block = base[name]
        start_block = start_cfg.get(name, base_block)
        end_block = end_cfg.get(name, start_block)
        block = copy.deepcopy(base_block)
        precision = block.get("precision", 3)
        for k, v in base_block.items():
            if k == "precision":
                block[k] = v
                continue
            sv = start_block.get(k, v)
            ev = end_block.get(k, sv)
            if isinstance(sv, (int, float)) and isinstance(ev, (int, float)):
                if mode == "linear":
                    interp_val = linear_interp(sv, ev, epoch, total_ep)
                else:
                    interp_val = sigmoid_interp(sv, ev, epoch, total_ep, steepness, center_frac)
                block[k] = round(interp_val, precision)
            else:
                block[k] = sv
        out[name] = block
    return out



class RandomBlackSquares(ImageOnlyTransform):
    def __init__(self, 
                 black_percentage=10.0,      # 전체 이미지 대비 검정색 비율 (%)
                 min_size=1,                # 사각형의 최소 한 변 길이 (픽셀)
                 max_size=3,                # 사각형의 최대 한 변 길이 (픽셀)
                 guass_std_range=(0.05, 0.4),
                 guass_noise_p=0.5,
                 p=0.7):
        super(RandomBlackSquares, self).__init__(p=p)
        self.black_percentage = black_percentage
        self.min_size = min_size
        self.max_size = max_size
        self.guass_std_range = guass_std_range
        self.guass_noise_p = guass_noise_p

    def apply(self, img, **params):
        # 이미지 높이와 너비
        h, w = img.shape[:2]
        total_pixels = h * w

        # 목표 검정색 픽셀 수 계산
        target_black_pixels = total_pixels * self.black_percentage / 100

        # 평균 사각형 면적 계산 (평균 크의 제곱)
        avg_size = (self.min_size + self.max_size) / 2
        avg_area = avg_size ** 2

        # 필요한 사각형 개수 계산
        num_squares = int(target_black_pixels / avg_area)

        if num_squares == 0:
            return img  # 적용할 사각형이 없으면 원본 이미지 반환

        # 랜덤한 사각형 크기 리스트 생성
        sizes = np.random.randint(self.min_size, self.max_size + 1, size=num_squares)

        # 랜덤한 사각형 위치 리스트 생성 (이미지 경계를 넘지 않도록 조정)
        xs = np.random.randint(0, w - sizes + 1)
        ys = np.random.randint(0, h - sizes + 1)

        # 이미지 복사 (원본을 수정하지 않도록)
        img_corrupted = img.copy()

        img_corrupted = A.GaussNoise(
            std_range=self.guass_std_range,
            p=self.guass_noise_p
        )(image=img_corrupted.astype(np.float32))["image"].astype(img.dtype)
        
        # 벡터화된 방식으로 정사각형 적용
        for x, y, size in zip(xs, ys, sizes):
            #img_corrupted[y:y+size, x:x+size] = 0  # 검정색으로 설정
            noise = np.random.rand(size, size, 3)
            if np.random.rand() < 0.3:
                noise = 0
            img_corrupted[y:y+size, x:x+size] = noise  # Noise로 설정
        
        return img_corrupted


def QuantizeNormData(float_data, normalize_func=None, denormalize_func=None):
    if isinstance(float_data, np.ndarray):
        if denormalize_func:
            quantized_data_int = np.floor(denormalize_func(float_data))
        else:
            quantized_data_int = np.floor(np.clip(float_data * 255, 0, 255))
        if normalize_func:
            quantized_data_normalized = normalize_func(quantized_data_int)
        else:
            quantized_data_normalized = quantized_data_int / 255.0
    elif isinstance(float_data, torch.Tensor):
        if denormalize_func:
            quantized_data_int = torch.floor(denormalize_func(float_data))
        else:
            quantized_data_int = torch.floor(torch.clamp(float_data * 255, 0, 255))
        if normalize_func:
            quantized_data_normalized = normalize_func(quantized_data_int)
        else:
            quantized_data_normalized = quantized_data_int / 255.0
    else:
        print("Error: Input must be Numpy Array or PyTorch Tensor.")
        raise TypeError(f"Input must be a NumPy array or PyTorch Tensor, But {type(float_data)}.")

    return quantized_data_normalized


def FillNoise(image, noise_rate=1e-2):
    noise = np.random.rand(*image.shape)
    results = (1.0-noise_rate)*image + noise_rate*noise
    return results


class FillNoise_transform(ImageOnlyTransform):
    def __init__(self, noise_rate=1e-2, p=0.5):
        super(FillNoise_transform, self).__init__(p=p)
        self.noise_rate = noise_rate

    def apply(self, img, **params):
        return FillNoise(img, noise_rate=self.noise_rate)



#def normalize_one_to_one(image: Union[torch.Tensor, np.ndarray])  -> Union[torch.Tensor, np.ndarray]:
#    _norm = image / 255.0
#    return (_norm - 0.5) * 2

def normalize_one_to_one(image: Union[torch.Tensor, np.ndarray],data_max=False)  -> Union[torch.Tensor, np.ndarray]:
    if data_max:
        _norm = image / image.max()
    else:
        _norm = image / 255.0
    return (_norm - 0.5) * 2

def denormalize_one_to_one(image: Union[torch.Tensor, np.ndarray])  -> Union[torch.Tensor, np.ndarray]:
    if isinstance(image, torch.Tensor):
        _image = (image.float() + 1.) / 2
        _image = _image.clamp(0.0, 1.0) * 255
    elif isinstance(image, np.ndarray):
        _image = (image + 1.) / 2
        _image = np.clip(_image, 0.0, 1.0) * 255
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    #return _image.to(dtype=torch.uint8) if isinstance(image, torch.Tensor) else _image.astype(np.uint8)
    return _image


class image_normalization_func:
    def __init__(self, set_max:float=1., set_min:float=0., data_max=False, data_min=False):
        self.norm_max = set_max
        self.norm_min = set_min
        self.data_max = data_max
        self.data_min = data_min
        self.eps = 1e-8

        self.img_max = 255.
        self.img_min = 0.


    def _norm_func(self, image: Union[torch.Tensor, np.ndarray],max_val=255.0, min_val=0.0):
        result = self.norm_min + (image - min_val) / (max_val - min_val + self.eps) * (self.norm_max - self.norm_min)
        return result
    
    def _denorm_func(self, image: Union[torch.Tensor, np.ndarray], max_val=255.0, min_val=0.0):
        result = min_val + (image - self.norm_min) / (self.norm_max - self.norm_min) * (max_val - min_val)
        return result

    def norm_input(self, image: Union[torch.Tensor, np.ndarray]):
        if self.data_max:
            max_val = image.max()
        else:
            max_val = self.img_max
        if self.data_min:
            min_val = image.min()
        else:
            min_val = self.img_min
        return self._norm_func(image, max_val, min_val)
    
    def norm_target(self, image: Union[torch.Tensor, np.ndarray]):
        return self._norm_func(image, self.img_max, self.img_min)

    def denorm_func(self, image: Union[torch.Tensor, np.ndarray]):
        return self._denorm_func(image, max_val = self.img_max, min_val = self.img_min)

    def denorm_input(self, image: Union[torch.Tensor, np.ndarray]):
        return self.denorm_func(image)
    
    def denorm_output(self, image: Union[torch.Tensor, np.ndarray]):
        return self.denorm_func(image)
