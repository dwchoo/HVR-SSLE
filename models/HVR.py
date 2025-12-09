import os
import sys
import torch
import torch.nn as nn

from typing import Tuple, Optional


try:
    from .swin2sr import Swin2SR_DFE
    from .LightUnetPP import LightUNetPlusPlus
    from .utils import pad_to_multiple_calculator, pad_to_multiple, crop_to_original, CnnEmbeddingLayer, lcm_of_list
except ImportError:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    from models.swin2sr import Swin2SR_DFE
    from models.LightUnetPP import LightUNetPlusPlus
    from models.utils import pad_to_multiple_calculator, pad_to_multiple, crop_to_original, CnnEmbeddingLayer, lcm_of_list




class HVR(nn.Module):
    def __init__(self, config, checkpoint: bool=True):
        super(HVR, self).__init__()
        self.config = config

        self.checkpoint = checkpoint
        self.T = config.HVR_T
        self.C = config.HVR_C
        self.N_supervision = config.HVR_N_supervision

        self.image_input_channels = 3
        self.embed_layer = CnnEmbeddingLayer(
            in_channels = self.image_input_channels,
            output_channels= config.embed_dims,
        )

        self.L_net_model = LightUNetPlusPlus(
            in_channels   = config.embed_dims,
            z_channels    = config.z_dims,
            out_channels  = config.z_dims,
            init_features = config.LightUnetPP.get('init_features', 64),
            num_groups    = config.LightUnetPP.get('num_groups', 16),
        )

        # calculate padding size
        input_image_h = config.height
        input_image_w = config.width
        self.padding_multiply_factor = lcm_of_list(
            [self.L_net_model.padding_multiply_factor, config.Swin2SR.get('window_size', 8)]
        )
        (_,_), (h_padded, w_padded) = pad_to_multiple_calculator(
            h = input_image_h,
            w = input_image_w,
            multiple=self.padding_multiply_factor
        )
        

        self.H_net_model = Swin2SR_DFE(
            img_size = (h_padded, w_padded),
            patch_size = 1,
            embed_dim = config.z_dims,
            depths = config.Swin2SR.get('depths', [3,3,3]),
            num_heads = config.Swin2SR.get('num_heads', [4,4,4]),
            window_size = config.Swin2SR.get('window_size', 8),
            mlp_ratio = config.Swin2SR.get('mlp_ratio', 4.0),
            qkv_bias = True,
            drop_rate = config.Swin2SR.get('drop_rate', 0.1),
            attn_drop_rate = config.Swin2SR.get('attn_drop_rate', 0.1),
            drop_path_rate = config.Swin2SR.get('drop_path_rate', 0.1),
            norm_layer = nn.LayerNorm,
            ape= False,
            patch_norm = True,
            use_checkpoint = self.checkpoint,
            img_range= 1.,
            resi_connection= config.Swin2SR.get('resi_connection', '1conv'),
        )


        self.head_layer = CnnEmbeddingLayer(
            in_channels = config.z_dims,
            output_channels= self.image_input_channels,
            enable_activation=True,
            last_one_to_one= True if config.normalize_one_to_one else False,
        )
    
    def L_net(self, x_embed: torch.Tensor, zL: torch.Tensor, zH: torch.Tensor):
        zL_in = zL
        zL_out = self.L_net_model(x_embed, zL + zH)
        return zL_in + zL_out

    def H_net(self, zH: torch.Tensor, zL: torch.Tensor):
        return self.H_net_model(zH + zL)


    @classmethod
    def z_init(cls, shape, zeros=False, a=-1.0, b=1.0, mean=0., std=1., device=None, dtype=None):
        result = torch.zeros(shape, device=device, dtype=dtype)
        if zeros:
            return result

        torch.nn.init.trunc_normal_(result, a=a, b=b, mean=mean, std=std)
        return result

    def forward(self, x, z: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,):
        batch_size, _, h, w = x.shape

        x_padded_org, (_, _) = pad_to_multiple(x, multiple=self.padding_multiply_factor)
        
        if z is None:
            zH = self.z_init(
                shape=(batch_size, self.config.z_dims, h, w),
                device=x.device, dtype=x.dtype,
                **self.config.z_distribution,
            )
            zL = self.z_init(
                shape=(batch_size, self.config.z_dims, h, w),
                device=x.device, dtype=x.dtype,
                **self.config.z_distribution,
            )
            z = (zH, zL)
        zH, zL = z
        zH, (_, _) = pad_to_multiple(zH, multiple=self.padding_multiply_factor)
        zL, (_, _) = pad_to_multiple(zL, multiple=self.padding_multiply_factor)

        with torch.no_grad():
            x_embed = self.embed_layer(x_padded_org)
            for _i in range(self.T * self.C -1):
                zL = self.L_net(x_embed, zL, zH)
                if (_i + 1) % self.T == 0:
                    zH = self.H_net(zH, zL)
        
        zH = zH.detach()
        zL = zL.detach()

        x_embed = self.embed_layer(x_padded_org)
        zL = self.L_net(x_embed, zL, zH)
        zH = self.H_net(zH, zL)

        output = self.head_layer(zH)
        output = crop_to_original(output, (h, w))
        zH = crop_to_original(zH, (h, w))
        zL = crop_to_original(zL, (h, w))

        return (zH, zL), output
        


    @torch.inference_mode()
    def sample(self, 
               x: torch.Tensor, 
               z: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
               T: Optional[int]=None, 
               C: Optional[int]=None, 
               N_supervision: Optional[int]=None,
               padding_multiply_factor: Optional[int]=None,
            ):
        T = self.T if T is None else T
        C = self.C if C is None else C
        N_supervision = self.N_supervision if N_supervision is None else N_supervision

        if padding_multiply_factor is None:
            padding_multiply_factor = self.padding_multiply_factor

        batch_size, _, h, w = x.shape

        if z is None:
            zH = self.z_init(
                shape=(batch_size, self.config.z_dims, h, w),
                device=x.device, dtype=x.dtype,
                **self.config.z_distribution,
            )
            zL = self.z_init(
                shape=(batch_size, self.config.z_dims, h, w),
                device=x.device, dtype=x.dtype,
                **self.config.z_distribution,
            )
            z = (zH, zL)

        zH, zL = z
        zH, (_, _) = pad_to_multiple(zH, multiple=padding_multiply_factor)
        zL, (_, _) = pad_to_multiple(zL, multiple=padding_multiply_factor)

        x_padded, (_, _) = pad_to_multiple(x, multiple=padding_multiply_factor)
        x_embed = self.embed_layer(x_padded)

        if N_supervision > 0 and C > 0 and T >0:
            for _s in range(N_supervision):
                for _i in range(T*C):
                    zL = self.L_net(x_embed, zL, zH)
                    if (_i + 1) % T == 0:
                        zH = self.H_net(zH, zL)
        elif N_supervision == 0:
            if T == 0:
                zL = self.L_net(x_embed, zL, zH)
                for _ in range(C):
                    zH = self.H_net(zH, zL)
            if C == 0:
                for _ in range(T):
                    zL = self.L_net(x_embed, zL, zH)
                zH = self.H_net(zH, zL)
            if C > 0 and T > 0:
                for _i in range(T*C):
                    zL = self.L_net(x_embed, zL, zH)
                    if (_i + 1) % T == 0:
                        zH = self.H_net(zH, zL)
        else:
            zL = self.L_net(x_embed, zL, zH)
            zH = self.H_net(zH, zL)
        
        output = self.head_layer(zH)
        output = crop_to_original(output, (h, w))
        zH = crop_to_original(zH, (h, w))
        zL = crop_to_original(zL, (h, w))

        return (zH, zL), output
        

if __name__ == '__main__':
    from torchinfo import summary  # torchinfo를 사용하여 모델 요약 출력
    import time
    from datetime import datetime
    from icecream import ic

    import sys
    from pathlib import Path

    FILE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = FILE_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from exp_config.config import Config


    batch_size = 8       # Example batch size
    dummy_dim = 3
    img_height = 256
    img_width = 256
    print(ic(str(datetime.now()))) # Removed ic here, simple print is fine for main script start
    # --- Parameters for Swin2SR_DFE Test ---
    
    # Config
    config = Config(name='test')

    config.normalize_max = False

    config.loss_weight = {
            'mae' : 1.0,
            'lpips' : 0.1,
            #'exposure' : 0.05,
        }
    config.HVR_T = 2
    config.HVR_C = 2
    config.HVR_N_supervision = 2
    config.HVR_N_supervision_inference_factor = 1
    config.embed_dims=24
    config.z_dims= 24
        
    config.LightUnetPP = dict(
            init_features = 24,
            num_groups = 3,
        )

    config.Swin2SR = dict(
            depths=[2,2],
            num_heads=[3,3],
            window_size=8,
            mlp_ratio=2.,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            resi_connection = '1conv' ## 1conv / 3conv
        )





    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        autocast_dtype = torch.float16 # Or torch.bfloat16 if supported
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        autocast_dtype = torch.float32 # Autocast typically used on GPU
        print("Using CPU")

    # --- Instantiate the Swin2SR_DFE model ---
    model_dfe = HVR(
        config,
    ).to(device) # Move model to the selected device

    # --- Create a dummy input tensor ---
    # Input shape: (Batch_Size, Channels=embed_dim, Height, Width)
    dummy_input = torch.randn((batch_size, dummy_dim, img_height, img_width)).to(device)
    dummy_zH = model_dfe.z_init(shape = (batch_size, config.z_dims, img_height, img_width), device=device)
    dummy_zL = model_dfe.z_init(shape = (batch_size, config.z_dims, img_height, img_width), device=device)
    dummy_z = (dummy_zH, dummy_zL)

    # --- Model Summary using torchinfo ---
    if summary:
        print("\n--- Swin2SR_DFE Model Summary ---")
        # Input_data should be a tuple containing one element: the input tensor shape
        # Or provide the input tensor directly
        summary(
            model_dfe,
            input_data=(dummy_input, dummy_z), # Pass the actual tensor or its shape
            # input_size=(batch_size, embed_dim, img_height, img_width), # Alternative way
            depth=5,  # Adjust depth as needed (e.g., 5 shows reasonable detail)
            col_names=["input_size", "output_size", "num_params", "trainable"],
            # row_settings=["var_names"] # Optional: show variable names
        )
    else:
        print("\n--- Swin2SR_DFE Model Structure (torchinfo not available) ---")
        print(model_dfe)


    # --- Check Model Parameter Strides ---
    #print("\n=== Check model parameter strides ===")
    #for name, param in model_dfe.named_parameters():
    #    # Only print for parameters that have strides (weights, not biases usually)
    #    if param.dim() > 1: # Check if dimension is > 1 to have stride concept
    #         print(f"{name}: shape = {param.shape}, stride = {param.stride()}")
    #    else:
    #         print(f"{name}: shape = {param.shape}")


    # --- GPU Memory Check (Before Forward Pass) ---
    if torch.cuda.is_available():
        print("\n=== GPU Memory Before Forward Pass ===")
        # Clear cache before checking memory (optional but good practice)
        # torch.cuda.empty_cache()
        print(torch.cuda.memory_summary(device=device, abbreviated=True))


    # --- Perform a forward pass ---
    print(f"\n--- Performing Forward Pass on {device} ---")
    print(f"Input Tensor Shape: {dummy_input.shape}")
    print(f"Input_time Tensor Shape: {dummy_zL.shape}")
    model_dfe.eval() # Set model to evaluation mode

    # Use autocast for potential speedup/memory saving on GPU
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=autocast_dtype):
         with torch.no_grad(): # Disable gradient calculation for testing
             (zH, zL), output = model_dfe(dummy_input, dummy_z)

    print(f"\n--- Forward Pass Output ---")
    print(f"Output Tensor Shape: {output.shape}")


    # --- GPU Memory Check (After Forward Pass) ---
    if torch.cuda.is_available():
        print("\n=== GPU Memory After Forward Pass ===")
        print(torch.cuda.memory_summary(device=device, abbreviated=True))

    # --- Verification ---
    # Check if output shape matches the original input H, W
    if output.shape == (batch_size, dummy_dim, img_height, img_width):
        print("\nOutput shape matches original input H, W. Basic Test Passed!")
    else:
        print(f"\n[Warning] Output shape {output.shape} does not match expected shape {(batch_size, dummy_dim, img_height, img_width)}. Check model padding/logic.")

    time.sleep(1) # Reduced sleep time

    # --- Different Resolution Test ---
    infer_h, infer_w = 400, 600 # Example different resolution
    print(f"\n=== Different Resolution Input Test (e.g., {infer_h}x{infer_w}) ===")
    # Note: Input channels must still be embed_dim
    x_infer = torch.randn(batch_size, dummy_dim, infer_h, infer_w).to(device)
    zH_infer = model_dfe.z_init(shape = (batch_size, config.z_dims, infer_h, infer_w)).to(device)
    zL_infer = model_dfe.z_init(shape = (batch_size, config.z_dims, infer_h, infer_w)).to(device)
    z_infer = (zH_infer, zL_infer)

    print(f"Input shape : {x_infer.shape}")
    print(f"Inference Input Timestep Shape : {zL_infer.shape}")

    # No need for t_infer as Swin2SR_DFE doesn't take time embeddings
    # t_infer = torch.rand(x_infer.shape[0]).to(device)
    # print(f"Input time emb shape : {t_infer.shape}")

    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=autocast_dtype):
        #with torch.no_grad():
        with torch.inference_mode():
            (_,_),output_infer = model_dfe.sample(x_infer, z_infer)

    print("\nInference data test results:")
    print("Input shape: ", x_infer.shape)
    print("Output shape:", output_infer.shape)

    # Verification for inference test
    if output_infer.shape == (batch_size, dummy_dim, infer_h, infer_w):
        print("\nInference output shape matches inference input shape. Test Passed!")
    else:
        print(f"\n[Warning] Inference output shape {output_infer.shape} does not match expected shape {(batch_size, dummy_dim, infer_h, infer_w)}. Check model padding/logic.")

    time.sleep(1) # Reduced sleep time

    print("\n=== Test Completed ===")
