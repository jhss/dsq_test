import torch

from modules import *

base_path = "/home/dsq_test"
bit=8

# 1. load input
input_int = torch.load(f"{base_path}/data/qk_int.pt", weights_only=True)
input_scale = torch.load(f"{base_path}/data/qk_scale.pt", weights_only=True)
input_scale = torch.tensor(input_scale)

# 2. calculate float
input_float = input_int * input_scale
input_float_max = torch.amax(input_float, dim=-1, keepdim=True)
input_float_subtract = input_float - input_float_max
#input_subtract_scale = get_scale(input_float_subtract, 16)

# 3. calculate difference
print("[DEBUG] input_float: ", input_float)
print("[DEBUG] input_int: ", input_int)
print("[DEBUG] input_scale: ", input_scale)
print("[DEBUG] input_int_max: ", torch.amax(input_int, dim=-1, keepdim=True)[0,0,0,0])
print("[DEBUG] input_float_max: ", torch.amax(input_float, dim=-1, keepdim=True)[0,0,0,0])
input_int_max = torch.amax(input_int, dim=-1, keepdim=True)
input_int_subtract = input_int.to(torch.int16) - input_int_max.to(torch.int16)
print("[DEBUG] input_int_subtract: ", input_int_subtract.min(), input_int_subtract.max())
input_quant_subtract = input_int_subtract * input_scale
print("[DEBUG] input_int_subtract: ", input_int_subtract)
print("[DEBUG] input_quant_subtract: ", input_quant_subtract)
print("[DEBUG] input_float_subtract: ", input_float_subtract)
input_diff = (input_quant_subtract - input_float_subtract).abs()
print("[DEBUG] input_diff: ", input_diff)