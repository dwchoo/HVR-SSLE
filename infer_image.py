#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm

from exp_config.config import Config
from models.HVR import HVR


class SimpleImageNormalization:
    def __init__(self, set_max: float = 1.0, set_min: float = 0.0, data_max=False, data_min=False):
        self.norm_max = set_max
        self.norm_min = set_min
        self.data_max = data_max
        self.data_min = data_min
        self.eps = 1e-8
        self.img_max = 255.0
        self.img_min = 0.0

    def _norm_func(self, image, max_val=255.0, min_val=0.0):
        return self.norm_min + (image - min_val) / (max_val - min_val + self.eps) * (self.norm_max - self.norm_min)

    def _denorm_func(self, image, max_val=255.0, min_val=0.0):
        return min_val + (image - self.norm_min) / (self.norm_max - self.norm_min) * (max_val - min_val)

    def norm_input(self, image):
        max_val = image.max() if self.data_max else self.img_max
        min_val = image.min() if self.data_min else self.img_min
        return self._norm_func(image, max_val=max_val, min_val=min_val)

    def norm_target(self, image):
        return self._norm_func(image, max_val=self.img_max, min_val=self.img_min)

    def denorm_func(self, image):
        return self._denorm_func(image, max_val=self.img_max, min_val=self.img_min)

    def denorm_input(self, image):
        return self.denorm_func(image)

    def denorm_output(self, image):
        return self.denorm_func(image)


def load_config(config_path: Path) -> Config:
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    cfg = Config.init_from_dict(cfg_dict)
    cfg.project_path = Path(".").resolve()
    return cfg


def build_preprocess(config: Config):
    if config.normalize_one_to_one:
        set_max, set_min, data_max = 1.0, -1.0, config.normalize_max
    else:
        set_max, set_min, data_max = 1.0, 0.0, config.normalize_max
    return SimpleImageNormalization(
        set_max=set_max,
        set_min=set_min,
        data_max=data_max,
        data_min=False,
    )


def load_image(path: Path, preprocess) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    norm_arr = preprocess.norm_input(arr)
    return torch.from_numpy(norm_arr).permute(2, 0, 1)


def save_image(tensor: torch.Tensor, path: Path, preprocess):
    denorm = preprocess.denorm_func(tensor)
    arr = (
        denorm.clamp(0, 255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(arr).save(path)


def gather_images(input_path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if input_path.is_file():
        return [input_path]
    files: List[Path] = []
    for ext in exts:
        files.extend(input_path.glob(f"*{ext}"))
    return sorted(files)


def clean_state_dict_keys(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        nk = k[10:] if k.startswith("_orig_mod.") else k
        cleaned[nk] = v

    def alias(src_prefix: str, dst_prefix: str):
        src_w, src_b = f"{src_prefix}.weight", f"{src_prefix}.bias"
        dst_w, dst_b = f"{dst_prefix}.weight", f"{dst_prefix}.bias"
        if src_w in cleaned and dst_w not in cleaned:
            cleaned[dst_w] = cleaned[src_w].clone()
        if src_b in cleaned and dst_b not in cleaned:
            cleaned[dst_b] = cleaned[src_b].clone()

    alias("embed_layer.embedding", "embed_layer.layers.0")
    alias("head_layer.embedding", "head_layer.layers.0")
    return cleaned


def load_state_dict_with_fallback(model: torch.nn.Module, state_dict):
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        cleaned = clean_state_dict_keys(state_dict)
        model.load_state_dict(cleaned)


def measure_gpu_memory_with_dummy(model, cfg, device,
                                  height: int = 400, width: int = 600) -> None:
    """더미 (width x height) 이미지를 1장 넣었을 때 GPU 메모리 사용량을 출력."""
    if device.type != "cuda":
        print("[GPU MEM] CUDA 장치가 아니라서 GPU 메모리를 측정할 수 없습니다.")
        return

    # 이전 연산과 섞이지 않도록 동기화 + peak 통계 리셋
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    # 현재까지(모델 로딩까지)의 메모리
    base_mem = torch.cuda.memory_allocated(device)

    # 더미 입력 (B, C, H, W) = (1, 3, 400, 600)
    dummy = torch.zeros(1, 3, height, width, device=device)

    # 실제 추론과 같은 N_supervision 사용
    n_supervision = max(
        int(cfg.HVR_N_supervision_inference_factor * cfg.HVR_N_supervision),
        1,
    )

    with torch.inference_mode():
        # HVR 인터페이스에 맞춰서 한 번 샘플링
        (_, _), _ = model.sample(
            dummy,
            T=cfg.HVR_T,
            C=cfg.HVR_C,
            N_supervision=n_supervision,
        )

    torch.cuda.synchronize(device)

    peak_mem = torch.cuda.max_memory_allocated(device)
    extra_mem = peak_mem - base_mem

    print(
        f"[GPU MEM] Dummy {width}x{height} 1장 기준 GPU 메모리 사용량"
        f"\n  - base (모델만): {base_mem / (1024**2):.2f} MB"
        f"\n  - peak:         {peak_mem / (1024**2):.2f} MB"
        f"\n  - extra:        {extra_mem / (1024**2):.2f} MB"
    )

    # 참조 제거 (선택 사항)
    del dummy





@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Run single-image inference with HVR.")
    parser.add_argument(
        "--config_path",
        required=True,
        help="config.json 경로",
    )
    parser.add_argument(
        "--weights_path",
        required=True,
        help="가중치 경로 (.safetensors)",
    )
    parser.add_argument("--input_path", required=True, help="단일 파일 또는 폴더 경로")
    parser.add_argument("--output_dir", required=True, help="결과 저장 폴더")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["auto", "cuda", "cpu"],
        help="추론 디바이스",
    )
    parser.add_argument("--hvr_t", type=int, default=None, help="샘플링용 HVR_T override")
    parser.add_argument("--hvr_c", type=int, default=None, help="샘플링용 HVR_C override")
    parser.add_argument("--hvr_n_supervision", type=int, default=None, help="샘플링용 HVR_N_supervision override")
    parser.add_argument("--hvr_n_sup_factor", type=float, default=None, help="샘플링용 HVR_N_supervision_inference_factor override")
    parser.add_argument(
        "--check_gpu_mem",
        action="store_true",
        help="더미 600x400 이미지를 한 번 넣어서 GPU 메모리 사용량만 출력하고 종료",
    )

    args = parser.parse_args()

    config_path = Path(args.config_path)
    weights_path = Path(args.weights_path)
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    if args.hvr_t is not None:
        cfg.HVR_T = args.hvr_t
    if args.hvr_c is not None:
        cfg.HVR_C = args.hvr_n
    if args.hvr_n_supervision is not None:
        cfg.HVR_N_supervision = args.hvr_n_supervision
    if args.hvr_n_sup_factor is not None:
        cfg.HVR_N_supervision_inference_factor = args.hvr_n_sup_factor
    preprocess = build_preprocess(cfg)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = HVR(cfg, checkpoint=False).to(device)
    # safetensors의 device 인자는 문자열을 기대할 수 있으므로 호환성을 위해 문자열로 전달
    load_device = str(device)
    state_dict = load_file(weights_path, device=load_device)
    load_state_dict_with_fallback(model, state_dict)
    model.eval()

    # ---- 여기 추가: GPU 메모리 측정 모드 ----
    if args.check_gpu_mem:
        measure_gpu_memory_with_dummy(model, cfg, device, height=400, width=600)
        return
    # -----------------------------------

    image_paths = gather_images(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {input_path}")

    n_supervision = max(
        int(cfg.HVR_N_supervision_inference_factor * cfg.HVR_N_supervision), 1
    )
    pbar = tqdm(image_paths, desc="Inference", ncols=80)
    for img_path in pbar:
        img_tensor = load_image(img_path, preprocess).unsqueeze(0).to(device)
        with torch.inference_mode():
            (_, _), output = model.sample(
                img_tensor,
                T=cfg.HVR_T,
                C=cfg.HVR_C,
                N_supervision=n_supervision,
            )
        output_single = output.squeeze(0).cpu()
        out_path = output_dir / img_path.name
        save_image(output_single, out_path, preprocess)
        pbar.write(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
