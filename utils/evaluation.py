import sys
import os
import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from typing import Union, Optional

from tqdm.auto import tqdm

# Accelerate 임포트
from accelerate import Accelerator

from dataset.preprocessing import image_normalization_func
from .metrics import Metric_PSNR, Metric_SSIM, Metric_LPIPS, Metric_NIQE, scipy_PSNR, scipy_SSIM
from .loss import Loss_hybrid, WeightTotalLoss

try:
    from ..dataset.preprocessing import LOLDataset, test_transform
    #print("✅ Successfully imported using relative paths (as a package module).")

except ImportError:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    from dataset.preprocessing import LOLDataset, test_transform

from icecream import ic


class Evaluation:
    def __init__(
        self, 
        model,
        accelerator,
        config,
    ):
        self.device = accelerator.device
        self.accelerator = accelerator
        self.model = model
        self.config = config

        self.model.eval()
    
    @torch.inference_mode()
    def generate_sample(self,input_images,):
        with self.accelerator.autocast():
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            (zH, zL), generated_images = unwrapped_model.sample(
                x = input_images,
                T = self.config.HVR_T,
                C = self.config.HVR_C,
                N_supervision=max(int(self.config.HVR_N_supervision_inference_factor*self.config.HVR_N_supervision),1)
            )

        return (zH, zL), generated_images


    @classmethod
    @torch.inference_mode()
    def evaluate(
        cls,
        config, 
        model, 
        accelerator, 
        dataloader, 
        data_name = 'LOLv1',
        save_sample=False,
        context={},
        now_epoch=0,
        preprocessing_func: Optional[image_normalization_func]=None,
        NoTarget = False,
    ):
        device = accelerator.device
        max_epoch = config.num_epochs

        evaluate_latent_model = cls(
            model = model,
            accelerator = accelerator,
            config = config,
        )
        metrics = evaluation_metrics(
            device=device,
            NoTarget=NoTarget,
            use_scipy=True,
        )

        if preprocessing_func is None:
            preprocessing_func = image_normalization_func()

        if config.progress_bar:
            loop = tqdm(
                dataloader,
                total = len(dataloader),
                leave=True,
                ncols=config.tqdm_length,
                desc=f"{data_name} | EPOCH {now_epoch+1}/{max_epoch}",
                bar_format="[Bench] | {desc} | {percentage:3.0f}% | {bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                #position=0,
            )
        else:
            loop = dataloader

        input_images_array = []
        target_images_array = []
        generated_images_array = []

        for step, batch in enumerate(loop):
            input_images = batch['input'].to(device)
            target_images = batch['target'].to(device)

            (zH, zL), generated_images = evaluate_latent_model.generate_sample(input_images)
            _denorm_generated_images = preprocessing_func.denorm_func(generated_images)
            _denorm_target_images = preprocessing_func.denorm_func(target_images)
            _generated_images_zero_one_norm = _denorm_generated_images / 255.0
            _target_image_zero_one_norm = _denorm_target_images / 255.0

            metrics.eval(_generated_images_zero_one_norm, _target_image_zero_one_norm)

            input_images_array.append(input_images.permute(0,2,3,1).detach().cpu().numpy())
            target_images_array.append(target_images.permute(0,2,3,1).detach().cpu().numpy())
            generated_images_numpy = generated_images.permute(0,2,3,1).float().detach().cpu().numpy()
            generated_images_array.append(generated_images_numpy)
            #generated_images_array.append(generated_images.permute(0,2,3,1).detach().cpu().numpy())

        input_images_array     = preprocessing_func.denorm_func(np.concatenate(input_images_array)).astype(np.uint8)
        target_images_array    = preprocessing_func.denorm_func(np.concatenate(target_images_array)).astype(np.uint8)
        generated_images_array = preprocessing_func.denorm_func(np.concatenate(generated_images_array)).astype(np.uint8)
        image_dict = {
            'input_images'     : input_images_array    ,
            'target_images'    : target_images_array   ,
            'generated_images' : generated_images_array,
        }
        return metrics.get_results(), image_dict

    @classmethod
    def denormalize_from_zero_to_one(cls, image):
        if isinstance(image, torch.Tensor):
            _image = image.float().clamp(0.0, 1.0) * 255
        elif isinstance(image, np.ndarray):
            _image = np.clip(image, 0.0, 1.0) * 255
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        #return _image.to(dtype=torch.uint8) if isinstance(image, torch.Tensor) else _image.astype(np.uint8)
        return _image



class evaluation_metrics:
    def __init__(self, device, verbose=False, NoTarget=False, use_scipy=False):
        self.device = device
        self.verbose = verbose
        self.use_scipy = use_scipy
        
        self.target_metrics = {
            "lpips": dict(results=np.array([]), metric=Metric_LPIPS(device=device)),
        }
        if self.use_scipy:
            self.target_metrics["psnr"] = dict(results=np.array([]), metric=scipy_PSNR(data_range=1.0,), flag='scipy')
            self.target_metrics["ssim"] = dict(results=np.array([]), metric=scipy_SSIM(data_range=1.0, channel_axis=-1, win_size=3), flag='scipy')
        else:
            self.target_metrics["psnr"] = dict(results=np.array([]), metric=Metric_PSNR(data_max=1.0, eps=1e-8, device=device))
            self.target_metrics["ssim"] = dict(results=np.array([]), metric=Metric_SSIM(device=device))
        self.NoTarget_metrics = {
            "niqe": dict(results=np.array([]), metric=Metric_NIQE(device=device), NoTarget=True),
        }
        #self.scipy_metrics = {
        #    "psnr": dict(results=np.array([]), metric=scipy_PSNR(data_range=1.0,)),
        #    "ssim": dict(results=np.array([]), metric=scipy_SSIM(data_range=1.0, channel_axis=-1, win_size=3)),
        #}
        if NoTarget:
            self.metrics = self.NoTarget_metrics
        else:
            self.metrics = {**self.target_metrics, **self.NoTarget_metrics}


    def calculate_metrics(self, metric, predicted_image_batch, target_image_batch, NoTarget, flag=None):
        if flag == 'scipy':
            predicted_image_batch_np = predicted_image_batch.permute(0,2,3,1).detach().cpu().numpy()
            target_image_batch_np = target_image_batch.permute(0,2,3,1).detach().cpu().numpy()
            if NoTarget:
                result_numpy = metric(pred = predicted_image_batch_np)
            else:
                result_numpy = metric(pred = predicted_image_batch_np, target = target_image_batch_np)

        else:
            predicted_image_batch = predicted_image_batch.to(self.device)
            target_image_batch = target_image_batch.to(self.device)
            if NoTarget:
                result_tensor = metric(predicted_image_batch)
            else:
                result_tensor = metric(predicted_image_batch, target_image_batch)
            result_numpy = result_tensor.detach().cpu().numpy()
        result_numpy = np.squeeze(result_numpy)
        
        return result_numpy

    def eval(self, x: torch.Tensor, y: torch.Tensor,) -> None:
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: predicted {x.shape} vs target {y.shape}")
        
        for metric_name, metric in self.metrics.items():
            try:
                result_numpy = self.calculate_metrics(
                    metric=metric.get('metric'),
                    predicted_image_batch=x.float(),
                    target_image_batch=y.float(),
                    NoTarget=metric.get('NoTarget', False),
                    flag=metric.get('flag', None),
                )
                # Ensure result_numpy is always at least 1D for concatenation
                metric['results'] = np.concatenate((metric.get('results'), np.atleast_1d(result_numpy)), axis=0)
                
                if self.verbose:
                    print(f"Metric: {metric_name}, Result: {result_numpy}")
            except Exception as e:
                print(f"Error calculating {metric_name}: {str(e)}")

    def get_results(self,):
        return {
            metric_name: {
                'all': np.array(m['results']),
                'mean': np.mean(m['results']),
            } for metric_name, m in self.metrics.items()
        }
        
    def reset_metrics(self) -> None:
        """Reset all metric results."""
        for metric in self.metrics.values():
            metric['results'] = np.array([])


def sampling_policy(data_length, sample_num, logger=None):
    if sample_num > data_length:
        #raise ValueError("샘플링 수가 데이터 길이보다 클 수 없습니다.")
        return [i for i in range(data_length)]
    
    # 간격 계산: (데이터 길이 - 1) / (샘플 개수 - 1)로 하여 적절한 간격 계산
    step = (data_length - 1) / (sample_num - 1)
    
    # 샘플링할 인덱스 생성
    sampled_indices = [round(i * step) for i in range(sample_num)]
    
    return sampled_indices



@torch.inference_mode()
def Run_benchmark_evaluation(
    config,
    model,
    benchmark_data_list,
    now_epoch,
    accelerator,
    save_sample = False,
    context = {},
    preprocessing_func : Optional[image_normalization_func]=None,
    *args,
    **kwargs,
):
    benchmark_results = {
        'metrics' : {},
        'images' : {},
    }
    
    for benchmark_data in benchmark_data_list:
        dataset_name = benchmark_data['name']
        dataloader = benchmark_data['dataloader']
        NoTarget = benchmark_data.get('NoTarget', False)
        metrics, image_dict = Evaluation.evaluate(
            config = config,
            model=model,
            accelerator=accelerator,
            dataloader=dataloader,
            data_name=dataset_name,
            save_sample=save_sample,
            context=context,
            now_epoch=now_epoch,
            preprocessing_func=preprocessing_func,
            NoTarget=NoTarget,
        )
        benchmark_results['metrics'][dataset_name] = metrics
        benchmark_results['images'][dataset_name] = image_dict
    
    return benchmark_results


def Prepare_benchmark_datasets(config, accelerator, preprocessing_func):
    """벤치마크용 데이터셋과 데이터로더를 초기화하는 함수
    
    Args:
        config: 설정값들이 포함된 Config 객체
        accelerator: Accelerator 인스턴스 (is_main_process 체크용)
    
    Returns:
        list: benchmark dataset과 dataloader 정보를 담은 딕셔너리 리스트
    """
    benchmark_data_list = []
    
    if accelerator.is_main_process:
        for name, dataset_info in config['eval_dataset'].items():
            dataset = LOLDataset(
                ll_image_dir=dataset_info['input_dir'],
                target_image_dir=dataset_info['target_dir'],
                preprocessing_func=preprocessing_func,
                transform=test_transform(
                    img_width=dataset_info.get('width', config['width']), 
                    img_height=dataset_info.get('height', config['height']),
                ),
                fill_noise = False,
                noise_rate = config['noise_rate'],
            )
            dataloader = DataLoader(
                dataset,
                batch_size=int(max(config['batch_size']* config['eval_batch_scale_factor'], 1)),
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=LOLDataset.collate_fn
            )
            benchmark_data_list.append({
                'name': name, 
                'dataloader': dataloader,
                'NoTarget' : dataset_info.get('NoTarget', False),
            })
    
    return benchmark_data_list