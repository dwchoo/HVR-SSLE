import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from accelerate import Accelerator # Import Accelerate
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs  # DDP configuration options (optional)
import numpy as np
import multiprocessing as mp


from models.HVR import HVR

from utils.lr_scheduler import CosineWithScaledRestartsLR
from utils.loss import Loss_lpips, Loss_hybrid
from utils.mp_forkserver import ensure_start_method


from dataset.preprocessing import (
    COCODataset, 
    LOLDataset, 
    coco_transform, 
    image_transform, 
    test_transform, 
    RandomBlackSquares, 
    normalize_one_to_one, 
    denormalize_one_to_one,
    image_normalization_func,
    resolve_pp_cfg,
)
from utils.evaluation import Run_benchmark_evaluation, Prepare_benchmark_datasets
from utils.config_loader import load_config_from_path


import wandb
import logging # Logging 추가
import random
import ujson as json # For loading metadata, use ujson for consistency
import glob # glob 모듈 추가

import gc


from icecream import ic


# --- Validation Function ---
#@torch.inference_mode()
def validate(model, dataloader, criterion, accelerator, config):
    """Performs validation on the validation dataset."""
    model.eval() # Set model to evaluation mode
    total_val_loss = 0.0
    device = accelerator.device
    #num_val_batches = 0 # 검증 배치 수 카운트
    accelerator.unwrap_model(criterion).reset_track_loss()

    step = 0
    last_step = int(len(dataloader) * config.get("sampling_train_dataset",1.0))
    with torch.inference_mode(): ## ADD
        for batch in tqdm(dataloader, desc="Validation", disable=not accelerator.is_local_main_process, leave=False, ncols=config.tqdm_length):
            input_images  = batch["input"].to(device, non_blocking=True)
            target_images = batch["target"].to(device, non_blocking=True)
            (zH, zL), pred_images = accelerator.unwrap_model(model).sample(
                input_images,
                N_supervision=max(int(config.HVR_N_supervision_inference_factor*config.HVR_N_supervision),1),
            )

            loss, loss_dict = criterion(
                pred_image = pred_images.float(),
                target_image = target_images.float(),
                input_image = input_images.float(),
                input_zero_to_one=False if not config.normalize_one_to_one else True,
            )
            step += 1
            if step > last_step:
                break

        mean_losses_dict, _ = accelerator.unwrap_model(criterion).compute_mean_losses(return_tensor=True)
    #total_loss = accelerator.gather(mean_losses_dict['total']).mean().item()
    #avg_loss_dict = {}
    #for _key, _value in mean_losses_dict.items():
    #    avg_loss_dict[_key] = accelerator.gather(_value).mean().item()
    with torch.no_grad():
        def _gather_scalar(x):
            # x: 0-dim tensor (inference에서 나온 값). float로 꺼내 '새 텐서' 생성
            t = torch.tensor(float(x.item()), device=accelerator.device)
            # all_gather 출력 텐서도 '일반 텐서'로 생성되도록 이 컨텍스트(비-inference)에서 호출
            return accelerator.gather_for_metrics(t).float().mean().item()

        total_loss = _gather_scalar(mean_losses_dict['total'])

        avg_loss_dict = {}
        for k, v in mean_losses_dict.items():
            avg_loss_dict[k] = _gather_scalar(v)

    accelerator.unwrap_model(criterion).reset_track_loss()
    return total_loss, avg_loss_dict
    
    #avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0 # 배치 수로 나누어 평균 계산
    #return avg_val_loss

# --- Main Training Function ---
def train(config):
    """
    Loads latent space data from WebDataset and trains a simple model.
    Uses Accelerate for DDP and mixed-precision training.
    """
    # --- 로깅 설정 ---
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    resume_from_checkpoint = getattr(config, 'resume_from_checkpoint', None)
    start_epoch = getattr(config, 'start_epoch', 0)
    best_val_loss = getattr(config, 'best_val_loss', float('inf'))

    # --- Setup ---
    _date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if resume_from_checkpoint:
        # 체크포인트가 있는 경우, 해당 실험 디렉토리를 그대로 사용합니다.
        EXP_RESULT_DIR = os.path.dirname(resume_from_checkpoint)
        logger.info(f"Resuming training from checkpoint. Experiment directory: {EXP_RESULT_DIR}")
    else:
        # 새로운 실험을 위한 디렉토리를 생성합니다.
        EXP_RESULT_DIR = os.path.join(config.output_dir, f"{config.wandb_project_name}_{config.name}_{_date_time}")
        logger.info(f"Starting new training. Experiment directory: {EXP_RESULT_DIR}")


    wandb_mode = os.getenv("WANDB_MODE", "offline")
    use_wandb_tracker = True

    EXP_RESULT_DIR = os.path.abspath(EXP_RESULT_DIR)

    COCO_TRAIN_DIR = config.coco_train_dir
    COCO_TRAIN_ANN = config.coco_train_annotation_file_path
    COCO_VAL_DIR = config.coco_val_dir
    COCO_VAL_ANN = config.coco_val_annotation_file_path

    BATCH_SIZE = config.batch_size # Effective batch size (accelerator handles gradient accumulation)
    NUM_WORKERS = config.num_workers
    EPOCHS = config.num_epochs
    LEARNING_RATE = config.learning_rate
    
    GRADIENT_ACCUMULATION_STEPS = config.get('gradient_accumulation_steps',1) # Gradient accumulation steps
    
    SAVE_EVERY = config.save_every # 체크포인트 저장 주기

    AUGMENT_RATE = config.augument_rate

    IMAGE_MASK = config.image_mask
    IMAGE_MASK['p'] = AUGMENT_RATE
    FILL_NOISE = AUGMENT_RATE if config.fill_noise is True else False

    SEED = 12345
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    # Kwargs handler for DDP related settings (optional, use for settings like find_unused_parameters)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False) # May need to be set to True depending on the model

    # Initialize Accelerator (with mixed precision enabled)
    # gradient_accumulation_steps: Number of steps to accumulate gradients (use if memory is limited)
    init_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(minutes=120))
    accelerator = Accelerator(
        mixed_precision=config.autocast_dtype, # 'fp16' or 'bf16' can be used
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        kwargs_handlers=[ddp_kwargs, init_kwargs], # Apply DDP settings
        log_with = "wandb" if use_wandb_tracker else None,
        project_dir=EXP_RESULT_DIR, # Accelerate 프로젝트 설정 및 체크포인트 저장 경로
        #even_batches=True,
    )


    # Accelerator 초기화 시 로깅 설정
    if accelerator.is_main_process:
        # wandb tracker 초기화
        os.makedirs(EXP_RESULT_DIR, exist_ok=True)
        wandb_id = None
        wandb_id_path = os.path.join(EXP_RESULT_DIR, "wandb_id.txt")

        if resume_from_checkpoint and os.path.exists(wandb_id_path):
            with open(wandb_id_path, "r") as f:
                wandb_id = f.read().strip()

        wandb_init_kwargs = {
            'mode' : wandb_mode,
            "dir": EXP_RESULT_DIR
        }
        if wandb_id:
            wandb_init_kwargs["id"] = wandb_id
            wandb_init_kwargs["resume"] = "must"

        accelerator.init_trackers(project_name=config.wandb_project_name, config=config.to_dict(),
                                  init_kwargs={"wandb": wandb_init_kwargs})

        if not wandb_id and accelerator.get_tracker("wandb").run is not None:
            with open(wandb_id_path, "w") as f:
                f.write(accelerator.get_tracker("wandb").run.id)

    accelerator.print(f"Accelerator setup complete: {accelerator.state}")
    accelerator.print(f"Using device: {accelerator.device}")
    accelerator.print(f"Mixed precision: {accelerator.mixed_precision}")
    accelerator.print(f"Output directory: {EXP_RESULT_DIR}")
    
    train_dataset = None
    validation_dataset = None

    normalize_func = None
    #denormalize_func = None

    if config.normalize_one_to_one:
        pp_set_max = 1.
        pp_set_min = -1.
        pp_data_max = config.normalize_max
    else:
        pp_set_max = 1.
        pp_set_min = 0.
        pp_data_max = False
    preprocessing_func = image_normalization_func(
        set_max = pp_set_max,
        set_min = pp_set_min,
        data_max = pp_data_max,
        data_min = False,
    )

    mp_start_method = mp.get_start_method(allow_none=True) or "spawn"
    mp_ctx = mp.get_context(mp_start_method)
    pp_manager = mp_ctx.Manager()
    shared_pp_cfg = pp_manager.dict(resolve_pp_cfg(config, start_epoch))

    try:

        data_transform = coco_transform(img_height=config.height, img_width=config.width)
        mask_transform = RandomBlackSquares(
            #black_percentage=5.0, min_size=1, max_size=5, p=AUGMENT_RATE,
            **IMAGE_MASK,
        )
        blur_transform = True
        
        # Load dataset
        train_dataset = COCODataset(
            image_dir=COCO_TRAIN_DIR,
            annotation_file=COCO_TRAIN_ANN,
            transform=data_transform,
            width=config.width,
            height=config.height,
            image_ids=None, # COCODataset 내부에서 ID를 가져오도록 함
            coco_instance=None, # 매번 새로 COCO 인스턴스 생성 (seed별 독립성)
            preprocessing_func = preprocessing_func,
            image_preprocessing_config=config.image_preprocessing_config,
            shared_image_preprocessing_config=shared_pp_cfg,
            mask_transform=mask_transform,
            blur_transform=blur_transform,
            blur_rate=AUGMENT_RATE,
            fill_noise=FILL_NOISE,
            noise_rate=config.noise_rate,
            seed = SEED,
        )
        validation_dataset = COCODataset(
            image_dir=COCO_VAL_DIR,
            annotation_file=COCO_VAL_ANN,
            transform=test_transform(img_height=config.height, img_width=config.width),
            width=config.width,
            height=config.height,
            image_ids=None, # COCODataset 내부에서 ID를 가져오도록 함
            coco_instance=None, # 매번 새로 COCO 인스턴스 생성 (seed별 독립성)
            preprocessing_func = preprocessing_func,
            image_preprocessing_config=config.image_preprocessing_config,
            mask_transform=None,
            blur_transform=False,
            blur_rate=0.0,
            fill_noise=False,
            noise_rate=0.01,
            seed = SEED,
        )

        benchmark_data_list = None
        if accelerator.is_main_process:
            benchmark_data_list = Prepare_benchmark_datasets(
                config = config,
                accelerator= accelerator,
                preprocessing_func = preprocessing_func,
            )

        # Dataloader
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE,
            num_workers = NUM_WORKERS,
            shuffle = True,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=config.get('dataloader_prefetch_factor',2),
            worker_init_fn=COCODataset.worker_init_fn if NUM_WORKERS > 0 else None,
            persistent_workers= True if NUM_WORKERS > 0 else False,
        )
        val_dataloader = DataLoader(
            validation_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS//2,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=COCODataset.worker_init_fn if NUM_WORKERS > 0 else None,
            persistent_workers=True if NUM_WORKERS > 0 else False,
        )

        if accelerator.is_main_process:
            print("DataLoader complete")
            wandb_tracker = accelerator.get_tracker("wandb")
            # 실험 관련 추가 정보를 wandb config에 한 번에 업데이트합니다.
            config_updates = {
                'exp_result_dir': EXP_RESULT_DIR,
                'coco_preprocessing_config': train_dataset.image_preprocessing_config,
            }
            wandb_tracker.run.config.update(config_updates, allow_val_change=True )

        else:
            wandb_tracker = None
        


        # --- 모델, 손실 함수, 옵티마이저 초기화 ---
        accelerator.print("Initializing model, loss function, and optimizer...")
        
        HVR_model = HVR(
            config = config,
            checkpoint = config.checkpoint,
        )
        # Gradient checkpoint
        HVR_model.checkpoint = True
        criterion = Loss_hybrid(
            device = accelerator.device, 
            accelerator=accelerator,
            loss_weights = config.loss_weight,
            exposure_mean=config.get('exposure_mean', 0.6),
        )

        optimizer = optim.AdamW(HVR_model.parameters(), lr=LEARNING_RATE)
        #optimizer = AdamW8bit(swin_model.parameters(), lr=LEARNING_RATE)
        
        # Learning rate scheduler
        num_warmup_steps = max(int(EPOCHS * config.lr_cosine_warmup_rate), 1)
        lr_scheduler = CosineWithScaledRestartsLR(
            optimizer= optimizer,
            num_warmup_epochs= num_warmup_steps,
            num_training_epochs= EPOCHS,
            num_cycles= config.lr_cosine_cycles,
            scaling_factor= config.lr_scaling_factor,
            warmup_epochs_per_cycle= None,
            last_epoch= -1,
            learning_rate_min= config.learning_rate_min,
            learning_rate_static= config.learning_rate_static,
        )

        accelerator.print("Initialization model, criterion, optimizer complete.")

        # --- Compile ---
        HVR_model = torch.compile(HVR_model)

        # --- Accelerator 준비 ---
        HVR_model, optimizer, train_dataloader, val_dataloader, lr_scheduler  = accelerator.prepare(
            HVR_model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
        accelerator.print("Accelerator preparation complete (model, optimizer, train_dataloader, val_dataloader).")

        # --- 학습 루프 ---
        accelerator.print(f"Starting training for {EPOCHS} epochs...")

        try:
            num_batches_per_epoch_this_process = len(train_dataloader)
            total_steps = (num_batches_per_epoch_this_process // GRADIENT_ACCUMULATION_STEPS) * EPOCHS
        except TypeError: 
            if accelerator.is_main_process:
                print("Could not determine len(train_dataloader). Overall progress bar might not show total steps.")
            total_steps = 0 

        completed_steps = getattr(config, 'completed_steps', 0)

        if resume_from_checkpoint:
            accelerator.print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            # `load_state`는 prepare된 모든 객체(모델, 옵티마이저, 스케줄러 등)의 상태를 로드합니다.
            accelerator.load_state(resume_from_checkpoint)
            accelerator.print(f"Resumed from epoch {start_epoch}. Completed steps: {completed_steps}")


        progress_bar = tqdm(
            range(total_steps) if total_steps > 0 else 0, 
            initial=completed_steps,
            disable=not accelerator.is_local_main_process, 
            desc="Overall Training Progress", 
            ncols=config.tqdm_length,
        )

        device = accelerator.device

        # Init criterion
        _ = criterion.to(device)
        accelerator.unwrap_model(criterion).reset_track_loss()

        # Init Swin model
        _ = HVR_model.to(device)


        for epoch in range(start_epoch, EPOCHS):
            HVR_model.train()
            accelerator.unwrap_model(criterion).reset_track_loss()
            if shared_pp_cfg is not None:
                new_pp_cfg = resolve_pp_cfg(config, epoch)
                shared_pp_cfg.clear()
                shared_pp_cfg.update(new_pp_cfg)
                if accelerator.is_main_process:
                    cfg_log = {}
                    for _name in ("alpha", "k", "l"):
                        for _key, _val in new_pp_cfg[_name].items():
                            cfg_log[f"preproc/{_name}/{_key}"] = _val
                    accelerator.log(cfg_log, step=completed_steps)
            #epoch_loss = 0.0
            #epoch_mse_loss = 0.0
            #epoch_lpips_loss = 0.0

            last_step = int(len(train_dataloader) * config.get("sampling_train_dataset",1.0))

            inner_pbar = tqdm(
                train_dataloader, 
                desc=f"Epoch {epoch+1}/{EPOCHS} Training", 
                disable=not accelerator.is_local_main_process, 
                ncols=config.tqdm_length,
            )

            for step, batch in enumerate(inner_pbar):
                input_images = batch["input"].to(device, non_blocking=True)
                target_images = batch["target"].to(device, non_blocking=True)
                b, c, h, w = input_images.shape
                
                with accelerator.accumulate(HVR_model):
                    with accelerator.autocast():
                        z = None
                        for _s in range(config.HVR_N_supervision):
                            z, pred_images = HVR_model(
                                x = input_images, 
                                z = z, 
                            )    
                            loss, loss_dict = criterion(
                                pred_image = pred_images.float(),
                                target_image = target_images.float(),
                                input_image = input_images.float(),
                                input_zero_to_one= False if config.normalize_one_to_one else True,
                            )
                            zH, zL = z
                            z = zH.detach(), zL.detach()
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                optimizer.step()
                                optimizer.zero_grad()
                        
                        accelerator.unwrap_model(criterion).add_value_to_track(dict_value = loss_dict, save_custom = True)
                        completed_steps += 1 
                        progress_bar.update(1) 

                        step_loss_dict = {}
                        for _loss_name in loss_dict.keys():
                            step_loss_dict[_loss_name] = accelerator.gather(loss_dict[_loss_name]).mean().item()
                                
                        if accelerator.is_main_process:
                            __metric = {}
                            for _key, _value in step_loss_dict.items():
                                __metric[f"train_{_key}_loss_step"] = _value
                            accelerator.log(__metric, step=completed_steps)
                            inner_pbar.set_postfix({
                                "step_loss": step_loss_dict['total'],
                            })

                if step > last_step:
                    accelerator.print(f"\nSampled train dataset, MAX steps reached. Current step:{step}, MAX steps:{last_step}\n")
                    break
            if accelerator.is_main_process:
                inner_pbar.close()
            

            # Record train losses
            _mean_losses_dict, _ = accelerator.unwrap_model(criterion).compute_mean_losses(
                accelerator.unwrap_model(criterion).custom_track_loss,
                return_tensor=True
            )
            mean_losses_dict = {}
            for _key, _value in _mean_losses_dict.items():
                mean_losses_dict[_key] = accelerator.gather(_value).mean().item()
            
            lr_scheduler.step(epoch + 1)
            current_learning_rate = round(optimizer.param_groups[0]['lr'],8)
            accelerator.print(f"TEST VALIDATION DATASET ON EPOCH {epoch+1}")
            accelerator.wait_for_everyone() 
            # validation아직 loss 안함
            avg_val_loss, avg_loss_dict = validate(
                model = HVR_model, 
                dataloader = val_dataloader,
                criterion= criterion, 
                accelerator = accelerator, 
                config = config
            )

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                #num_actual_train_steps = (step + 1) // GRADIENT_ACCUMULATION_STEPS
                #avg_epoch_loss = epoch_loss / num_actual_train_steps if num_actual_train_steps > 0 else 0.0

                metrics = {
                    "epoch": epoch + 1, 
                    'learning_rate': current_learning_rate,
                }
                for _key in mean_losses_dict.keys():
                    metrics[f"train_{_key}_loss_epoch"] = mean_losses_dict[_key]
                    metrics[f"val_{_key}_loss"] = avg_loss_dict[_key]
                accelerator.log(metrics, step=completed_steps) 
                accelerator.print(f"\nEpoch {epoch+1} complete. Avg Train Loss: {mean_losses_dict['total']:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

                if (epoch + 1) % SAVE_EVERY == 0:
                    epoch_str = str(epoch + 1).zfill(len(str(EPOCHS)))
                    #save_path = os.path.join(EXP_RESULT_DIR, f"checkpoint_epoch_{epoch_str}")
                    save_path = os.path.join(EXP_RESULT_DIR, f"checkpoint_epoch_latest")
                    accelerator.save_state(output_dir=save_path)
                    # config와 학습 상태 저장
                    config.save_json(
                        os.path.join(EXP_RESULT_DIR, "config.json"),
                        start_epoch=epoch + 1,
                        completed_steps=completed_steps,
                        best_val_loss=best_val_loss
                    )
                    accelerator.print(f"Checkpoint saved to {save_path}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_save_path = os.path.join(EXP_RESULT_DIR, "best_model")
                    accelerator.save_state(output_dir=best_model_save_path) 
                    config.save_json(
                        os.path.join(EXP_RESULT_DIR, "config.json"),
                        start_epoch=epoch + 1,
                        completed_steps=completed_steps,
                        best_val_loss=best_val_loss
                    )
                    accelerator.print(f"New best model saved to {best_model_save_path} with validation loss: {best_val_loss:.4f}")

            accelerator.wait_for_everyone() 
            _save_epoch_bool = (epoch == 0) or ((epoch + 1) % config.save_every == 0) or (epoch+1==EPOCHS)
            if accelerator.is_main_process and (benchmark_data_list is not None) and _save_epoch_bool:

                try:
                    del batch, input_images, target_images, loss
                except NameError:
                    pass # 변수가 할당되지 않은 경우 무시
                gc.collect()
                torch.cuda.empty_cache()

                accelerator.print(f"Calculate benchmark dataset | epoch: {epoch}")
                benchmark_results = Run_benchmark_evaluation(
                    config = config,
                    model= HVR_model,
                    benchmark_data_list= benchmark_data_list,
                    now_epoch= epoch,
                    accelerator= accelerator,
                    save_sample = False,
                    context = {},
                    preprocessing_func = preprocessing_func,
                )
                benchmark_results_means = {}
                for __dataset, __dataset_results in benchmark_results['metrics'].items():
                    benchmark_results_means[f"{__dataset}"] = {}
                    for __metric, __results in __dataset_results.items():
                        benchmark_results_means[f"{__dataset}"][f"{__metric}"] = __results['mean']
                accelerator.log(benchmark_results_means, step=completed_steps)

                log_images_dict = {}
                if epoch == 0:
                    image_name_list = ['input_images', 'target_images', 'generated_images']
                else:
                    image_name_list = ['generated_images']
                if len(image_name_list) > 0:
                    accelerator.print(f"Make wandb image logs | epoch: {epoch} | list: {image_name_list}")
                    for _dataset_name, _image_dict in benchmark_results['images'].items():
                        for _image_name in image_name_list:
                            __wandb_image_list = []
                            for __idx, __image in enumerate(_image_dict[_image_name]):
                               __wandb_image = wandb.Image(
                                   __image, 
                                   caption=f"{_dataset_name}/{_image_name}/{__idx}",
                                   file_type="jpg",
                                )
                               __wandb_image_list.append(__wandb_image)
                            log_images_dict[f"{_dataset_name}/{_image_name}"] = __wandb_image_list
                    accelerator.log(log_images_dict, step=completed_steps)
                    accelerator.print(f"Finish upload wandb image logs")

            accelerator.wait_for_everyone()

        accelerator.print("\nTraining complete.")

    finally:
        accelerator.end_training()



# --- Argument Parser ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and validate a model")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--config_name", type=str, default='Config', help="Config name")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to the checkpoint directory to resume training from.")
    parser.add_argument("--forkserver", action='store_true', help="Use 'forkserver' as the multiprocessing start method.")
    args = parser.parse_args()

    if args.forkserver:
        method = ensure_start_method("forkserver", fallback="spawn", force=True)
        print(f"[mp] start method = {method}")
        


    if args.resume_from_checkpoint:
        # 학습 재개 시, 저장된 config.json에서 설정을 불러옵니다.
        ic(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        exp_dir = os.path.dirname(args.resume_from_checkpoint)
        config_json_path = os.path.join(exp_dir, "config.json")
        if not os.path.exists(config_json_path):
            raise FileNotFoundError(f"config.json not found in experiment directory: {exp_dir}")

        with open(config_json_path, 'r') as f:
            config_dict = json.load(f)

        # Config 클래스를 동적으로 로드합니다.
        module_path = args.config_path.replace(os.path.sep, '.').replace('.py', '')
        try:
            config_module = __import__(module_path, fromlist=[args.config_name])
            ConfigClass = getattr(config_module, args.config_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not load config class '{args.config_name}' from '{args.config_path}': {e}")

        config = ConfigClass.init_from_dict(config_dict)
        # 재개를 위한 추가 정보들을 config 객체에 설정합니다.
        config.resume_from_checkpoint = args.resume_from_checkpoint
        config.start_epoch = config_dict.get('start_epoch', 0)
        config.completed_steps = config_dict.get('completed_steps', 0)
        config.best_val_loss = config_dict.get('best_val_loss', float('inf'))
    else:
        # 새로운 학습 시작
        config_factory = load_config_from_path(
            config_path = args.config_path,
            config_name = args.config_name,
            config_exp_name= args.exp_name,
        )

        if args.exp_name is None:
            name = 'test'
        else:
            name = args.exp_name
        
        config = config_factory(name=name)

    train(config)
