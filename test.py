import torch

base_path = "/home/dsq_test"

# 1. load input
input_int = torch.load(f"{base_path}/data/qk_int.pt")
input_scale = torch.load(f"{base_path}/data/qk_scale.pt")
input_scale = torch.tensor(input_scale)

# 2. Calculate integer-only exp
input_int_max = torch.amax(input_int, dim=-1, keepdim=True)
print("[DEBUG] input_int shape: ", input_int.shape)
input_int_subtract = input_int - input_int_max
print("[DEBUG] input_int min, max: ", input_int.min().item(), ", ", input_int.max().item())
print("[DEBUG] input_int max: ", input_int_max.shape)
print("[DEBUG] min, max: ", input_int_subtract.min().item(), ", ", input_int_subtract.max().item())
# exp_int, exp_scale = int_exp_shift(input_int_subtract, input_scale)

# # 3. Calculate torch float exp
# input_quant = input_int * input_scale
# exp_float = torch.exp(input_quant)