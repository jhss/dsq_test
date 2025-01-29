import torch

from modules import int_exp_shift, quantize, get_scale, get_min_max, get_max_idx

base_path = "/home/dsq_test"

# 1. load input
input_int = torch.load(f"{base_path}/data/qk_int.pt", weights_only=True)
input_scale = torch.load(f"{base_path}/data/qk_scale.pt", weights_only=True)
input_scale = torch.tensor(input_scale)

# 1.1. convert input to 16-bit
input_quant = (input_int * input_scale)
input_scale = get_scale(input_quant, bit=16)
input_int = quantize(input_quant, input_scale, bit=16)

# 1.2. calculate input quantization accuracy
input_float_16 = input_quant
input_quant_16 = input_int * input_scale
input_quant_diff = (input_float_16 - input_quant_16).abs()
print("[DEBUG] input_quant diff max: ", input_quant_diff.max().item(), " mean: ", input_quant_diff.mean().item())

# 2. Calculate integer-only exp
input_int_max = torch.amax(input_int, dim=-1, keepdim=True)
input_int_subtract = input_int.to(torch.int16) - input_int_max.to(torch.int16)
exp_int, exp_scale = int_exp_shift(input_int_subtract, input_scale)

print("[DEBUG] input_int_subtract min, max: ", get_min_max(input_int_subtract))
max_idx = get_max_idx(input_int_subtract)

print("[DEBUG] input_int: ", input_int[max_idx[0], max_idx[1], max_idx[2], max_idx[3]])
print("[DEBUG] max: ", input_int_max[0,0,max_idx[2],0], " max shape: ", input_int_max.shape)
print("[DEBUG] input_int_subtract: ", input_int_subtract[max_idx[0], max_idx[1], max_idx[2], max_idx[3]])

print("============================================================================")

# 3. Calculate torch float exp
input_quant = input_int_subtract * input_scale
exp_float = torch.exp(input_quant)
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