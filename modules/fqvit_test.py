import torch
import torch.nn as nn

from fqvit.layers import QIntSoftmax

base_path = "/home/dsq_test"
layer = QIntSoftmax()

# 1. load input
input_int = torch.load(f"{base_path}/data/qk_int.pt", weights_only=True)
input_scale = torch.load(f"{base_path}/data/qk_scale.pt", weights_only=True)
input_scale = torch.tensor(input_scale)

# 2. Calculate input_max_subtract scale
input_float = input_int * input_scale

output_dequant = layer(input_float)
output_float = torch.exp(input_float)

print(output_dequant)
print(output_float)