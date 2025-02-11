import torch

from modules.triton.exp import _exp, int_exp_poly
from modules.utils import *

triton_exp = _exp.apply
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor = torch.randn((4,8), dtype=torch.float32, device=device)
input_bit = 8
subtract_bit = 16

input_scale = get_scale(input_tensor, input_bit)
input_int = quantize(input_tensor, input_scale, input_bit)
input_float = input_int * input_scale
input_float_max = torch.amax(input_float, dim=-1, keepdim=True)
input_float_subtract = input_float - input_float_max

subtract_scale = get_scale(input_float_subtract, subtract_bit)
subtract_int = quantize(input_float_subtract, subtract_scale, subtract_bit)
subtract_int = subtract_int.to(torch.int16)

# torch exp
torch_exp_int, torch_exp_scale = int_exp_poly(subtract_int, subtract_scale)
torch_exp_quant = torch_exp_int * torch_exp_scale
torch_exp_float = torch.exp(input_float_subtract)
print("==========================================================================")
print("==========================================================================")
# triton exp
triton_exp_int, triton_exp_scale = triton_exp(subtract_int, subtract_scale)
triton_exp_quant = triton_exp_int * triton_exp_scale

print("torch exp quant: ", torch_exp_quant)
print("torch exp float: ", torch_exp_float)
print("triton exp quant: ", triton_exp_quant)