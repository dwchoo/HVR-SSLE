#!/usr/bin/env python
import argparse
from pathlib import Path

from safetensors.torch import load_file, save_file


def convert_state_dict(state_dict):
    new_sd = {}

    for k, v in state_dict.items():
        nk = k[10:] if k.startswith("_orig_mod.") else k
        new_sd[nk] = v

    def alias(src_prefix: str, dst_prefix: str):
        src_w, src_b = f"{src_prefix}.weight", f"{src_prefix}.bias"
        dst_w, dst_b = f"{dst_prefix}.weight", f"{dst_prefix}.bias"
        if src_w in new_sd and dst_w not in new_sd:
            new_sd[dst_w] = new_sd[src_w].clone()
        if src_b in new_sd and dst_b not in new_sd:
            new_sd[dst_b] = new_sd[src_b].clone()

    alias("embed_layer.embedding", "embed_layer.layers.0")
    alias("head_layer.embedding", "head_layer.layers.0")

    return new_sd


def main():
    parser = argparse.ArgumentParser(
        description="Convert safetensors checkpoint to remove _orig_mod prefix and add alias keys."
    )
    parser.add_argument("--input", required=True, help="입력 safetensors 경로")
    parser.add_argument(
        "--output",
        default=None,
        help="출력 경로 (미지정 시 <input>_fixed.safetensors)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output is not None
        else input_path.with_name(input_path.stem + "_fixed.safetensors")
    )
    state_dict = load_file(input_path, device="cpu")
    new_sd = convert_state_dict(state_dict)
    save_file(new_sd, output_path)
    print(f"Converted checkpoint saved to: {output_path}")
    print(f"Keys before: {len(state_dict)}, after: {len(new_sd)}")


if __name__ == "__main__":
    main()
