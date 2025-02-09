import torch
import torch.nn as nn

from .base import BaseQuantizer

class SplitQuantizer(BaseQuantizer):
    def __init__(self, bit_type, observer, module_type):
        super(SplitQuantizer, self).__init__(bit_type, observer, module_type)

    def update_quantization_params(self, *args, **kwargs):
        pass

    def quant(self, inputs, scale=None, zero_point=None, threshold=None):
        scale_A = threshold / (2**self.bit_type.bits-1)
        scale_B = (1.0 - threshold) / (2**self.bit_type.bits-1)

        inputs_A = torch.where(inputs <= threshold, inputs, 0)
        inputs_B = torch.where(inputs > threshold, inputs, 0)

        outputs_A = (inputs_A / self.scale_A).round().clamp(self.bit_type.lower_bound,
                                                            self.bit_type.upper_bound)
        outputs_B = (inputs_B / self.scale_B).round().clamp(self.bit_type.lower_bound,
                                                            self.bit_type.upper_bound)
        return outputs_A, outputs_B

    def dequntize(self, inputs, scale=None, zero_point=None, threshold=None):
        scale_A = threshold / (2**self.bit_type.bits-1)
        scale_B = (1.0 - threshold) / (2**self.bit_type.bits-1)
        
        outputs = inputs[0] * scale_A + inputs[1] * scale_B

        return outputs
