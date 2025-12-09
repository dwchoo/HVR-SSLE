python3 ./infer_image.py \
    --config_path ./checkpoint/config.json \
    --weights_path ./checkpoint/HVR.safetensors \
    --input_path dataset/LOLv1/eval15/low \
    --output_dir ./results/LOLv1 \
    --device auto
