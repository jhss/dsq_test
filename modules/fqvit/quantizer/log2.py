# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch

from .base import BaseQuantizer


class Log2Quantizer(BaseQuantizer):

    def __init__(self, bit_type, observer, module_type):
        super(Log2Quantizer, self).__init__(
            bit_type,
            observer,
            module_type,
        )
        self.softmax_mask = None

    def quant(self, inputs, threshold=None):
        rounds = torch.round(-1 * inputs.log2())
        self.softmax_mask = rounds >= 2**self.bit_type.bits
        outputs = torch.clamp(rounds, 0, 2**self.bit_type.bits - 1)
        return outputs

    def dequantize(self, inputs, threshold=None):
        outputs = 2**(-1 * inputs)
        outputs[self.softmax_mask] = 0
        return outputs
