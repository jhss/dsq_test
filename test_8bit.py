import torch

from modules import *

base_path = "/home/dsq_test"
bit=8

# 1. load input
input_int = torch.load(f"{base_path}/data/qk_int.pt", weights_only=True)
input_scale = torch.load(f"{base_path}/data/qk_scale.pt", weights_only=True)
input_scale = torch.tensor(input_scale)

# 2. Calculate input_max_subtract scale
input_float = input_int * input_scale
input_float_max = torch.amax(input_float, dim=-1, keepdim=True)
input_float_subtract = input_float - input_float_max

# 2. Calculate integer-only exp
input_int_max = torch.amax(input_int, dim=-1, keepdim=True)
input_int_subtract = input_int.to(torch.int64) - input_int_max.to(torch.int64)
print("[DEBUG] input_int_subtract: ", input_int_subtract.min(), input_int_subtract.max())
input_quant_subtract = input_int_subtract * input_scale

# 3. requantize subtract
subtract_scale = get_scale(input_quant_subtract, bit)
subtract_int_req = quantize(input_quant_subtract, subtract_scale, bit)
subtract_quant_req = subtract_int_req * subtract_scale
print("[DEBUG] subtract_int_req: ", subtract_int_req)
input_diff = (input_float_subtract - subtract_quant_req).abs()
print("[DEBUG] input float: ", input_float_subtract)
print("[DEBUG] input quant: ", subtract_quant_req)
print("[DEBUG] input_diff max: ", input_diff.max().item(), ", ", input_diff.mean().item())
print("=========================================================================")
exp_int, exp_scale = int_exp_shift(subtract_int_req, subtract_scale)

# # 3. Calculate torch float exp
exp_float = torch.exp(input_float_subtract)
exp_quant = exp_int * exp_scale
print("=========================================================================")
print("[DEBUG] exp_quant: ", exp_quant)
print("[DEBUG] exp_float: ", exp_float)

# # 4. Calculate diff
exp_diff = (exp_float - exp_quant).abs()
print("[DEBUG] exp_diff max: ", exp_diff.max().item(), " mean: ", exp_diff.mean().item())

# # 5. Calculate max diff idx
max_idx = (exp_diff == exp_diff.max()).nonzero()
print("max_idx: ", max_idx)

# # 6. Compare exp quant and float at maximum idx
max_idx = max_idx[0]
print("[DEBUG] exp quant max idx: ", exp_quant[max_idx[0], max_idx[1], max_idx[2], 45:48])
print("[DEBUG] exp float max idx: ", exp_float[max_idx[0], max_idx[1], max_idx[2], 45:48])
print("=========================================================================")

print("exp quant shape: ", exp_quant.shape)
softmax_from_quant = exp_quant / exp_quant.sum(dim=-1)
softmax_from_float = exp_float / exp_float.sum(dim=-1)
print("softmax_from_quant: ", softmax_from_quant)
print("softmax_from_float: ", softmax_from_float)

softmax_diff = (softmax_from_quant - softmax_from_float).abs()
print("softmax_diff max: ", softmax_diff.max().item(), ", ", softmax_diff.mean().item())