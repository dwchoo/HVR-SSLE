
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
import ujson as json # For loading metadata
import os
import copy


PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    name: str
    project_path: Path = field(default=PROJECT_ROOT)
    wandb_project_name: str= "HVR-SSLE"
    width: int=224
    height: int=224
    sample_width: int= 256
    sample_height: int= 256

    autocast_dtype: str='bf16' # ['no', 'fp8', 'fp16', 'bf16']
    checkpoint: bool = True
    
    batch_size: int=48
    num_workers: int=8
    dataloader_prefetch_factor: int=2
    num_epochs: int=300
    gradient_accumulation_steps: int= 1
    learning_rate_static: bool=False
    learning_rate: float=5e-4
    learning_rate_min: float=1e-9
    lr_cosine_warmup_rate: float = 0.05
    lr_cosine_cycles: int = 2
    lr_scaling_factor: float=0.2

    normalize_one_to_one: bool = True
    normalize_max:bool = False
    
    loss_weight: Dict[str, float]=field(default_factory=lambda: {
        'mae' : 1.0,
        'sat' : 1.0,
        'lpips' : 0.1,
    })
    exposure_mean: float=0.4
    exposure_loss_patch_size: int=16

    augument_rate: float=0.1
    image_mask: Dict[str, Any] = field(default_factory=lambda: {
        'black_percentage': 10.0,
        'min_size': 1,
        'max_size': 2,
    })

    blur: bool=True
    fill_noise: bool=True
    noise_rate: float=0.1

    image_preprocessing_config: Dict[str, Any] = field(default_factory=lambda:{
        'alpha' : {
            'min_val' : 0.0,
            'max_val' : 0.3,
            'precision' : 3,
        },
        'k' : {
            'min_val' : 0.0,
            'max_val' : 0.4,
            'mean' : 0.0,
            'precision' : 3,
        },
        'l' : {
            'min_val' : 0.0,
            'max_val' : 0.95,
            'mean' : 0.0,
            'precision' : 3,
        },
    })
    image_preprocessing_schedule: Optional[Dict[str, Any]] = None  # None이면 start=end=고정 분포가 자동 설정됩니다.

    pre_trained: bool=False


    z_distribution: Dict[str, Any] = field(default_factory=lambda: {
        'mean': 0.0,
        'std': 1.0,
        'a' : -1.0,
        'b' : 1.0,
        'zeros' : False,
    })
    
    HVR_T: int = 2
    HVR_C: int = 2
    HVR_N_supervision: int = 2
    HVR_N_supervision_inference_factor: int = 1

    ###### DIMS ######
    embed_dims: int=24
    z_dims: int = 24
    
    LightUnetPP: Dict[str, Any]=field(default_factory=lambda: dict(
        init_features = 24,
        num_groups = 8,
    ))

    Swin2SR: Dict[str, Any]=field(default_factory=lambda: dict(
        depths=[2, 2,],
        num_heads=[4, 4,],
        window_size=8,
        mlp_ratio=2.,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        resi_connection = '1conv' ## 1conv / 3conv
    ))

    save_every: int= 10
    device: str='cuda'
    progress_bar: bool=True
    coco_train_dir: Path= Path('dataset/coco/train2017')
    coco_val_dir: Path = Path('dataset/coco/val2017')
    coco_train_annotation_file_path: Path=Path("dataset/coco/annotations/instances_train2017.json")
    coco_val_annotation_file_path: Path=Path("dataset/coco/annotations/instances_val2017.json")
    train_image_ids: List[int]=None
    validation_image_ids: List[int]=None
    test_image_ids: List[int]=None
    output_dir: str= './checkpoint'      # 모델 저장 디렉토리 경로로 변경
    sample_num: int= 20
    num_sample_batch: int=1
    eval_dataset: Dict[str, Dict[str, Union[str, int, Path]]] = field(default_factory=lambda: {
        'LOLv1' : {'input_dir' : Path('dataset/LOLv1/eval15/low'),
                  'target_dir' : Path('dataset/LOLv1/eval15/high'),
                  'width' : 600,
                  'height' : 400},
        'LOLv2_real' : {'input_dir' : Path('dataset/LOL-v2/Real_captured/Test/Low'),
                        'target_dir' : Path('dataset/LOL-v2/Real_captured/Test/Normal'),
                        'width' : 600,
                        'height' : 400},
        'LOLv2_synthetic' : {'input_dir' : 'dataset/LOL-v2/Synthetic/Test/Low',
                  'target_dir' : 'dataset/LOL-v2/Synthetic/Test/Normal',
                    'width' : 384,
                    'height' : 384,},
        'LSRW_Huawei' : {'input_dir' : Path('dataset/LSRW/Eval/Huawei/low'),
                        'target_dir' : Path('dataset/LSRW/Eval/Huawei/high'),
                        'width' : 960,
                        'height' : 720},
        'LSRW_Nikon' : {'input_dir' : Path('dataset/LSRW/Eval/Nikon/low'),
                        'target_dir' : Path('dataset/LSRW/Eval/Nikon/high'),
                        'width' : 960,
                        'height' : 640},
        #'LIME' : {'input_dir' : Path('dataset/LIME'),
        #          'target_dir' : Path('dataset/LIME'),
        #          'width' : 600,
        #          'height' : 400,
        #          'NoTarget' : True,}
        
    })
    save_train_samples: bool=True
    eval_batch_scale_factor: float = 0.1
    logging: bool=True

    tqdm_length: int = 80
    sampling_train_dataset: float = 1.0
    debug: int=0

    info:str = ''
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def keys(self):
        """
        Config 객체의 모든 attribute 이름을 반환합니다.
        """
        return [f.name for f in self.__dataclass_fields__.values()]
    
    def get(self, key, default_value=None):
        return getattr(self, key, default_value)
    
    def to_dict(self):
        """
        dataclass를 dictionary로 변환하며, Path 객체를 문자열로 변환합니다.
        """
        data = asdict(self)
        
        def _convert_paths_recursive(item):
            if isinstance(item, dict):
                return {k: _convert_paths_recursive(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [_convert_paths_recursive(v) for v in item]
            elif isinstance(item, Path):
                return str(item)
            else:
                return item
        
        return _convert_paths_recursive(data)

    def from_dict(self, data):
        """
        딕셔너리에서 값을 가져와 Config 객체를 업데이트합니다.
        """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_json(self, file_path: Union[str, Path], **kwargs):
        """Config 객체를 JSON 파일로 저장하고, 추가 인자를 포함합니다."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        
        config_dict.pop('resume_from_checkpoint', None)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)


    def __post_init__(self):
        """객체의 모든 속성을 재귀적으로 순회하며 Path 객체를 절대 경로로 변환합니다."""

        if self.image_preprocessing_schedule is None:
            # 스케줄이 None이면 start=end=기존 분포로 설정되어 epoch 동안 값이 고정됩니다.
            self.image_preprocessing_schedule = dict(
                mode = "sigmoid",
                start = copy.deepcopy(self.image_preprocessing_config),
                end = copy.deepcopy(self.image_preprocessing_config),
                steepness = 6.0,
                center_frac = 0.5,
            )

        def _resolve_paths_recursive(item: Any) -> Any:
            if isinstance(item, Path):
                # project_path가 상대경로 './' 같은 것을 포함할 수 있으므로 resolve()로 정규화
                if not item.is_absolute():
                    return (self.project_path / item).resolve()
                return item.resolve()
            elif isinstance(item, dict):
                return {key: _resolve_paths_recursive(value) for key, value in item.items()}
            elif isinstance(item, list):
                return [_resolve_paths_recursive(elem) for elem in item]
            else:
                return item

        for key, value in self.__dict__.items():
            if key == 'project_path':
                continue
            setattr(self, key, _resolve_paths_recursive(value))

    @classmethod
    def init_from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """
        Creates a Config instance from a dictionary, ignoring any keys that don't match.
        """
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)

    @classmethod
    def default(cls, name:str):
        config = cls(name=name)
        return config

    @classmethod
    def short_test(cls, name:str):
        config = cls(name=name)
        config.sampling_train_dataset = 0.02
        return config


    @classmethod
    def base_model(cls, name: str):
        config = cls(name=name)

        config.normalize_max = False
        config.fill_noise = True
        config.noise_rate = 0.1

        config.num_epochs = 300
        config.lr_cosine_cycles= 2
        config.learning_rate = 5e-4
        config.learning_rate_static = False
        config.lr_scaling_factor = 0.2
        config.learning_rate_min = 1e-9
        config.batch_size=64
        config.gradient_accumulation_steps=1
        config.loss_weight = {
                'mae' : 1.0,
                'sat' : 1.0,
                'lpips' : 0.1,
            }
        config.exposure_mean=0.5
        config.augument_rate=0.1
        config.image_preprocessing_config = {
                'alpha' : {
                    'min_val' : 0.0,
                    'max_val' : 0.3,
                    'precision' : 3,
                },
                'k' : {
                    'min_val' : 0.0,
                    'max_val' : 0.4,
                    'mean' : 0.0,
                    'precision' : 3,
                },
                'l' : {
                    'min_val' : 0.0,
                    'max_val' : 0.95,
                    'mean' : 0.0,
                    'precision' : 3,
                },
            }

        config.image_preprocessing_schedule = None

        config.z_distribution= {
                'mean': 0.0,
                'std': 1.0,
                'a' : -1.0,
                'b' : 1.0,
                'zeros' : False,
            }
        config.embed_dims=24
        config.z_dims= 24
            
        config.LightUnetPP = dict(
                init_features = 24,
                num_groups = 2,
            )

        config.Swin2SR = dict(
                depths=[2,2],
                num_heads=[4,4],
                window_size=8,
                mlp_ratio=2.,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                resi_connection = '1conv' ## 1conv / 3conv
            )
        config.info = "ai-server-a100"

        config.eval_dataset =  {
            'LOLv1' : {'input_dir' : Path('dataset/LOLv1/eval15/low'),
                    'target_dir' : Path('dataset/LOLv1/eval15/high'),
                    'width' : 600,
                    'height' : 400},
            'LOLv2_real' : {'input_dir' : Path('dataset/LOL-v2/Real_captured/Test/Low'),
                            'target_dir' : Path('dataset/LOL-v2/Real_captured/Test/Normal'),
                            'width' : 600,
                            'height' : 400},
            'LOLv2_synthetic' : {'input_dir' : 'dataset/LOL-v2/Synthetic/Test/Low',
                                'target_dir' : 'dataset/LOL-v2/Synthetic/Test/Normal',
                                'width' : 384,
                                'height' : 384,},
            'LSRW_Huawei' : {'input_dir' : Path('dataset/LSRW/Eval/Huawei/low'),
                            'target_dir' : Path('dataset/LSRW/Eval/Huawei/high'),
                            'width' : 960,
                            'height' : 720},
            'LSRW_Nikon' : {'input_dir' : Path('dataset/LSRW/Eval/Nikon/low'),
                            'target_dir' : Path('dataset/LSRW/Eval/Nikon/high'),
                            'width' : 960,
                            'height' : 640},
            
        }

        return config
    