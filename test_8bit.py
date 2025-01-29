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
input_subtract_scale = get_scale(input_float_subtract, bit)
combined_scale = torch.round((input_subtract_scale / input_scale) * (2**bit)).to(torch.int16)

# 2. Calculate integer-only exp
input_int_max = torch.amax(input_int, dim=-1, keepdim=True)
input_int_subtract = input_int.to(torch.int64) - input_int_max.to(torch.int64)
input_int_subtract_shift = (combined_scale * input_int_subtract) >> bit
exp_int, exp_scale = int_exp_shift(input_int_subtract_shift, input_subtract_scale)

# 2.1. Check input subtract shift
input_int_subtract_quant = input_int_subtract_shift * input_subtract_scale
print("=======================================================================================")
print("[DEBUG] input_int_subract_quant: ", input_int_subtract_quant)
print("[DEBUG] input_float_subtract: ", input_float_subtract)
exit(0)

# 3. Calculate torch float exp
exp_float = torch.exp(input_float_subtract)
exp_quant = exp_int * exp_scale

print("[DEBUG] exp_quant: ", exp_quant)
print("[DEBUG] exp_float: ", exp_float)

# 4. Calculate diff
exp_diff = (exp_float - exp_quant).abs()
print("[DEBUG] exp_diff max: ", exp_diff.max().item(), " mean: ", exp_diff.mean().item())

# 5. Calculate max diff idx
max_idx = (exp_diff == exp_diff.max()).nonzero()
print("max_idx: ", max_idx)

# 6. Compare exp quant and float at maximum idx
max_idx = max_idx[0]
print("[DEBUG] exp quant max idx: ", exp_quant[max_idx[0], max_idx[1], max_idx[2], max_idx[3]])
print("[DEBUG] exp float max idx: ", exp_float[max_idx[0], max_idx[1], max_idx[2], max_idx[3]])