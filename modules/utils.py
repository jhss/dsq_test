import torch
from torch.autograd import Function

class floor_ste(Function):

    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

def quantize(x, scale, bit, per_channel = True):
    min_int, max_int = - (2**(bit-1)), (2**(bit-1)-1)
    x_int = torch.clamp(torch.round(x/scale), min_int, max_int)
    if bit == 8:
        x_int = x_int.to(torch.int8)
    elif bit == 16:
        x_int = x_int.to(torch.int16)
    
    return x_int

def get_scale(x, bit=8, is_tensor=True):
    min_int, max_int = -(2**(bit-1)), 2**(bit-1)-1
    scale = 2*x.abs().max() / (max_int - min_int)
    if not is_tensor:
        scale = scale.item()
    return scale

def get_min_max(x: torch.Tensor):
    return x.min().item(), x.max().item()

def get_max_idx(x: torch.Tensor):
    return (x == x.max().item()).nonzero()[0]
