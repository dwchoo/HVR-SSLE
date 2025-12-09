from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce


class CnnEmbeddingLayer(nn.Module):
    """
    Converts the input image channels (e.g., 3 for RGB) to the desired
    embedding dimension for the model's main body.
    This is achieved using a 1x1 Convolution (Pointwise Convolution).
    
    Args:
        in_channels (int): Number of channels in the input image (e.g., 3).
        embedding_dim (int): The number of feature channels the model will use internally.
    """
    def __init__(self,
                 in_channels: int,
                 output_channels: int,
                 kernel_size: int=1,
                 stride:int=1,
                 padding: int=0,
                 enable_activation: bool=False,
                 last_one_to_one: bool=False,
                 custom_activation: Optional[nn.Module]=None,
                 ):
        super(CnnEmbeddingLayer, self).__init__()
        module_list = []
        self.embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        module_list.append(self.embedding)
        if enable_activation:
            if custom_activation is not None:
                module_list.append(custom_activation)
            elif last_one_to_one:
                module_list.append(nn.Tanh())
            else:
                module_list.append(nn.Sigmoid())
        self.layers = nn.ModuleList(module_list)
        
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x





def pad_to_multiple_calculator(h, w, multiple=4):
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return (pad_h, pad_w), (h + pad_h, w + pad_w)


def pad_to_multiple(tensor, multiple=4, mode='reflect'):
    """
    Pad the input tensor so that H and W are multiples of 'multiple'.
    Returns padded tensor and original sizes for later cropping.
    """
    _, _, h, w = tensor.shape
    #pad_h = (multiple - h % multiple) % multiple
    #pad_w = (multiple - w % multiple) % multiple
    (pad_h, pad_w), (_, _) = pad_to_multiple_calculator(h, w, multiple)
    if pad_h == 0 and pad_w == 0:
        return tensor, (h, w)
    
    # Pad: (left, right, top, bottom)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
    return padded, (h, w)

def crop_to_original(tensor, original_size):
    """
    Crop the tensor back to original H, W.
    """
    h, w = original_size
    return tensor[:, :, :h, :w]


def lcm_two_numbers(a, b):
    return abs(a * b) // math.gcd(a, b) if a != 0 and b != 0 else 0

def lcm_of_list(numbers):
    non_zero_numbers = [num for num in numbers if num != 0]
    
    if not non_zero_numbers:
        return 1
        
    return reduce(lcm_two_numbers, non_zero_numbers)
