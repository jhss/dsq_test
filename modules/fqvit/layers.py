# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F

from .bit_type import BIT_TYPE_DICT
from .observer import build_observer
from .quantizer import build_quantizer
from .ibert import QuantAct
from .utils import floor_ste, round_ste

class QConv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'conv_weight'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            #print("[DEBUG-CONV-CALIBRATE]")
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            #print("[DEBUG-CONV-FLOAT]")
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        #print("[DEBUG-CONV-QUANT]")
        weight = self.quantizer(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class QLinear(nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QLinear, self).__init__(in_features, out_features, bias)

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'linear_weight'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            #print("[DEBUG-LINEAR] float")
            return F.linear(x, self.weight, self.bias)
        #print("[DEBUG-LINEAR] quant")
        weight = self.quantizer(self.weight)
        return F.linear(x, weight, self.bias)


class QAct(nn.Module):

    def __init__(self,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QAct, self).__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(x)
            if self.last_calibrate:
                # import ipdb;ipdb.set_trace()
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            #print("[DEBUG-QACT] float")
            return x
        #print("[DEBUG-QACT] quant")
        x = self.quantizer(x)
        return x


class QIntLayerNorm(nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(QIntLayerNorm, self).__init__(normalized_shape, eps,
                                            elementwise_affine)
        assert isinstance(normalized_shape, int)
        self.mode = 'ln'

    def get_MN(self, x):
        bit = 8
        N = torch.clamp(bit - 1 - torch.floor(torch.log2(x)), 0, 31)
        M = torch.clamp(torch.floor(x * torch.pow(2, N)), 0, 2 ** bit - 1)
        return M, N

    def forward(self,
                x,
                in_quantizer=None,
                out_quantizer=None,
                in_scale_expand=1):
        if self.mode == 'ln':
            #print("[DEBUG-LAYERNORM] float")
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
        elif self.mode == 'int':
            #print("[DEBUG-LAYERNORM] int")
            in_scale = in_quantizer.scale
            if in_scale_expand != 1:
                in_scale = in_scale.unsqueeze(-1).expand(
                    -1, in_scale_expand).T.reshape(-1)
            out_scale = out_quantizer.scale
            assert in_scale is not None and out_scale is not None
            channel_nums = x.shape[-1]
            in_scale = in_scale.reshape(1, 1, -1)
            out_scale = out_scale.reshape(1, 1, -1)
            #x_q = (x / in_scale).round()
            x_q = round_ste.apply(x/in_scale)
            in_scale1 = in_scale.min()
            #in_scale_mask = (in_scale / in_scale1).round()
            in_scale_mask = round_ste.apply(in_scale / in_scale1)

            x_q = x_q * in_scale_mask

            mean_x_q = x_q.mean(dim=-1) * in_scale1
            std_x_q = (in_scale1 / channel_nums) * torch.sqrt(
                channel_nums * (x_q**2).sum(dim=-1) - x_q.sum(dim=-1)**2)

            A = (in_scale1 / std_x_q).unsqueeze(-1) * \
                self.weight.reshape(1, 1, -1) / out_scale
            A_sign = A.sign()
            M, N = self.get_MN(A.abs())
            B = floor_ste.apply((self.bias.reshape(1, 1, -1) -
                  (mean_x_q / std_x_q).unsqueeze(-1) *
                  self.weight.reshape(1, 1, -1)) / out_scale *
                 torch.pow(2, N))

            x_q = floor_ste.apply((A_sign * M * x_q + B) / torch.pow(2, N))
            x = x_q * out_scale
        else:
            raise NotImplementedError
        return x


class QIntSoftmax(nn.Module):

    def __init__(self,
                 log_i_softmax=False,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QIntSoftmax, self).__init__()

        self.log_i_softmax = log_i_softmax
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    @staticmethod
    def log_round(x):
        #x_log_floor = x.log2().floor()
        x_log_floor = floor_ste.apply(x.log2())
        big = x_log_floor
        extra_mask = (x - 2**big) >= 2**(big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big

    @staticmethod
    def int_softmax(x, scaling_factor):

        def int_polynomial(x_int, scaling_factor):
            coef = [0.35815147, 0.96963238, 1.]  # ax**2 + bx + c
            coef[1] /= coef[0]
            coef[2] /= coef[0]
            #b_int = torch.floor(coef[1] / scaling_factor)
            #c_int = torch.floor(coef[2] / scaling_factor**2)
            b_int = floor_ste.apply(coef[1] / scaling_factor)
            c_int = floor_ste.apply(coef[2] / scaling_factor**2)
            #print("[DEBUG] b_int: ", b_int)
            #print("[DEBUG] c_int: ", c_int)
            z = x_int + b_int
            #print("[DEBUG] +b_int: ", z)
            z = x_int * z
            #print("[DEBUG] r*z: ", z)
            z = z + c_int
            #print("[DEBUG] +c_int: ", z)
            scaling_factor = coef[0] * scaling_factor**2
            return z, scaling_factor

        def int_exp(x_int, scaling_factor):
            x0 = -0.69314718  # -ln2
            n = 15  # sufficiently large integer
            #x0_int = torch.floor(x0 / scaling_factor)
            x0_int = floor_ste.apply(x0 / scaling_factor)
            x_int = torch.max(x_int, n * x0_int)
            #print("[DEBUG] x0_int: ", x0_int)
            #print("[DEBUG] x_int_clipped: ", x_int)
            #q = torch.floor(x_int / x0_int)
            q = floor_ste.apply(x_int / x0_int)
            #print("[DEBUG] q: ", q)
            r = x_int - x0_int * q
            #print("[DEBUG] r: ", r)
            exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
            #exp_int = torch.clamp(torch.floor(exp_int * 2**(n - q)), min=0)
            #print("[DEBUG] before clamp exp_int: ", exp_int)
            #print("[DEBUG] n-q: ", n-q)
            exp_int = torch.clamp(floor_ste.apply(exp_int * 2**(n - q)), min=0)
            #print("[DEBUG] after clamp exp_int: ", exp_int)
            scaling_factor = exp_scaling_factor / 2**n
            return exp_int, scaling_factor


        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        exp_int, exp_scaling_factor = int_exp(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        return exp_int, exp_int_sum, exp_scaling_factor

    def forward(self, x, scale=None):
        self.quantizer.observer.update(x)
        self.quantizer.update_quantization_params(x)
        scale, zeropoint = self.quantizer.observer.get_quantization_params()

        exp_int, exp_int_sum, exp_scaling_factor = self.int_softmax(x, scale)
        return exp_int * exp_scaling_factor
        # softmax_out = torch.round(exp_int_sum / exp_int)
        # rounds = self.log_round(softmax_out)
        # mask = rounds >= 2**self.bit_type.bits
        # qlog = torch.clamp(rounds, 0, 2**self.bit_type.bits - 1)
        # deq_softmax = 2**(-qlog)
        # deq_softmax[mask] = 0
        # return deq_softmax

        # if self.quant and self.log_i_softmax and scale is not None:
        #     exp_int, exp_int_sum, _ = self.int_softmax(x, scale)
        #     softmax_out = torch.round(exp_int_sum / exp_int)
        #     rounds = self.log_round(softmax_out)
        #     mask = rounds >= 2**self.bit_type.bits
        #     qlog = torch.clamp(rounds, 0, 2**self.bit_type.bits - 1)
        #     deq_softmax = 2**(-qlog)
        #     deq_softmax[mask] = 0
        #     return deq_softmax
        # else:
        #     x = x.softmax(dim=-1)
        #     if self.calibrate:
        #         self.quantizer.observer.update(x)
        #         if self.last_calibrate:
        #             self.quantizer.update_quantization_params(x)
        #     if not self.quant:
        #         return x
        #     x = self.quantizer(x)
        #     return x

class QSplitIntSoftmax(QIntSoftmax):
    def __init__(self,
                 split_softmax=False,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QSplitIntSoftmax, self).__init__(False,
                                               quant, 
                                               calibrate,
                                               last_calibrate,
                                               bit_type,
                                               calibration_mode,
                                               observer_str,
                                               quantizer_str)
        self.split_softmax=split_softmax
        self.threshold = nn.Parameter(torch.Tensor([0.1]))
        self.threshold_bits = 8
        self.output_bit = bit_type.bits
        self.max_bit = 31
        #self.qact = QuantAct(16, quant_mode='symmetric')

    def forward(self, x, scale):
        if not self.quant:
            x = x.softmax(dim=-1)
            return x
        elif self.split_softmax and scale is not None:
            #print("x: ", x)
            exp_int, exp_int_sum, exp_scaling_factor = self.int_softmax(x, scale)
            #exp, exp_scaling_factor = self.qact(exp_int, exp_scaling_factor)
            #exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
            tmp_int_softmax = exp_int / exp_int_sum
            #print("[DEBUG] tmp softmax: ", tmp_int_softmax[0,0,0,:])
            factor = floor_ste.apply((2**32)/exp_int_sum)
            out_scale_A = (self.threshold - 0.0) / (2**(self.output_bit)-1)
            #out_scale_B = (1.0 - self.threshold) / (2**(self.output_bit)-1)
            out_scale_B = 1.0 / (2**(self.output_bit)-1)
            #print("[DEBUG] output bit: ", self.output_bit)
            #print("[DEBUG] out_scale_A: ", out_scale_A) 
            #print("[DEBUG] out_scale_B: ", out_scale_B)
            approx_threshold = floor_ste.apply(self.threshold * (2**self.threshold_bits)) * exp_int_sum / (2**self.threshold_bits)
            print("approx threshold first batch first head: ", approx_threshold[0,0,:,:])
            print("approx threshold first batch second head: ", approx_threshold[0,1,:,:])
            print("approx threshold second batch first head: ", approx_threshold[1,0,:,:])
            print("approx threshold second batch second head: ", approx_threshold[1,1,:,:])
            exp_int_A = torch.where(exp_int <= approx_threshold, exp_int, torch.tensor(0.0, dtype=exp_int.dtype).to(exp_int.device))
            exp_int_B = torch.where(exp_int > approx_threshold, exp_int, torch.tensor(0.0, dtype=exp_int.dtype).to(exp_int.device))
            softmax_int_A = floor_ste.apply(exp_int_A * factor / (2 ** 32*out_scale_A))
            softmax_int_B = floor_ste.apply(exp_int_B * factor / (2 ** 32*out_scale_B))
            softmax_int_B = torch.clamp(softmax_int_B, max=255)
            dequant_output = softmax_int_A * out_scale_A + softmax_int_B * out_scale_B
            torch_softmax = x.softmax(dim=-1)
            #print("[DEBUG] dequant softmax output: ", dequant_output[0,0,0,:])
            #print("[DEBUG] float softmax output: ",torch_softmax[0,0,0,:] )
            #exit(0)
            return dequant_output
        else:
            x = x.softmax(dim=-1)
            if self.calibrate:
                #print("[DEBUG-SPLIT] before quantize: ", x)
                x = self.quantizer(x)
                #print("[DEBUG-SPLIT] after quantize: ", x)
                #exit(0)
                # self.quantizer.observer.update(x)
                # if self.last_calibrate:
                #     self.quantizer.update_quantization_params(x)
            if not self.quant:
                return x
            #print("[DEBUG-SPLIT] before quantize: ", x)
            x = self.quantizer(x)
            #print("[DEBUG-SPLIT] after quantize: ", x)
            #exit(0)
            return x

class Shiftmax(nn.Module):
    def __init__(self, output_bit=8):
        super(Shiftmax, self).__init__()
        self.output_bit = output_bit

        self.n = 20

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        print("[DEBUG-SHIFT] x_int max: ", x_int_max)
        x_int = x_int - x_int_max
        print("[DEBUG-SHIFT] x_int after max: ", x_int)

        x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2 ** 4)
        print("[DEBUG-SHIFT] x_int after shift: ", x_int)
        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        print("[DEBUG-SHIFT] x_int minimum: ", torch.min(x_int))
        print("[DEBUG-SHIFT] n * x0_int: ", self.n * x0_int)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r/2 - x0_int
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2 ** self.n
        return exp_int, scaling_factor

    def forward(self, x, scaling_factor):
        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int, exp_scale = self.int_exp_shift(x_int, scaling_factor)
        # exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        # exp_int_sum.clamp_max_(2**31-1)
        # factor = floor_ste.apply((2**31-1) / exp_int_sum)
        # exp_int = floor_ste.apply(exp_int * factor / 2 ** (31-self.output_bit+1))
        # scaling_factor = torch.Tensor([1 / 2 ** (self.output_bit-1)])

        # self.act_scaling_factor = scaling_factor
        return exp_int * scaling_factor, scaling_factor


class QSplitShiftmax(Shiftmax):
    def __init__(self,
                 split_softmax=False,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QSplitShiftmax, self).__init__(8)
        self.split_softmax=split_softmax
        self.threshold = nn.Parameter(torch.Tensor([0.03]))
        self.threshold_bits = 8
        self.output_bit = bit_type.bits
        self.max_bit = 31
        #print("[DEBUG] output bit: ", self.output_bit)
        #self.qact = QuantAct(16, quant_mode='symmetric')

    def forward(self, x, scale):
        #print("[DEBUG-SPLIT-SOFTMAX] split forward (quant)")
        #torch.save(x.cpu(), "/home/workspace/FQ-ViT/tensorrt_evaluation/data/input/x.pt")
        #torch.save(scale.cpu(), "/home/workspace/FQ-ViT/tensorrt_evaluation/data/input/scale.pt")
        exp_int, exp_scaling_factor = self.int_exp_shift(x, scale)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        exp_int_sum.clamp_max_(2**32)

        factor = floor_ste.apply((2**32)/exp_int_sum)
        out_scale_A = (self.threshold - 0.0) / (2**(self.output_bit)-1)
        out_scale_B = (1.0 - self.threshold) / (2**(self.output_bit)-1)
        approx_threshold = floor_ste.apply(self.threshold * (2**self.threshold_bits)) * exp_int_sum / (2**self.threshold_bits)
        exp_int_A = torch.where(exp_int <= approx_threshold, exp_int, torch.tensor(0.0, dtype=exp_int.dtype).to(exp_int.device))
        exp_int_B = torch.where(exp_int > approx_threshold, exp_int, torch.tensor(0.0, dtype=exp_int.dtype).to(exp_int.device))
        softmax_int_A = floor_ste.apply(exp_int_A * factor / (2 ** 32*out_scale_A))
        softmax_int_B = floor_ste.apply(exp_int_B * factor / (2 ** 32*out_scale_B))
        softmax_int_B = torch.clamp(softmax_int_B, max=255)
        dequant_output = softmax_int_A * out_scale_A + softmax_int_B * out_scale_B
        #torch.save(dequant_output.cpu(), "/home/workspace/FQ-ViT/tensorrt_evaluation/data/torch_results/dequant_softmax.pt")
        return dequant_output
        