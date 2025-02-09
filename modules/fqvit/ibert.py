import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
from torch.nn import functional as F
import decimal
from decimal import Decimal

class floor_ste(Function):
    """
    Straight-through Estimator(STE) for torch.floor()
    """
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class QuantAct(nn.Module):
    """
    Class to quantize given activations

    Parameters:
    ----------
    activation_bit : int
        Bitwidth for quantized activations.
    act_range_momentum : float, default 0.95
        Momentum for updating the activation quantization range.
    running_stat : bool, default True
        Whether to use running statistics for activation quantization range.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    channel_len : int, default None
        Specify the channel length when using the per_channel mode.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """
    def __init__(self,
                 activation_bit,
                 act_range_momentum=0.95,
                 running_stat=True,
                 per_channel=False,
                 channel_len=None,
                 quant_mode="none"):
        super(QuantAct, self).__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.percentile = False

        if not per_channel:
            self.register_buffer('x_min', torch.zeros(1))
            self.register_buffer('x_max', torch.zeros(1))
            self.register_buffer('act_scaling_factor', torch.zeros(1))
        else:
            assert channel_len is not None
            self.register_buffer('x_min', torch.zeros(channel_len))
            self.register_buffer('x_max', torch.zeros(channel_len))
            self.register_buffer('act_scaling_factor', torch.zeros(channel_len))

        self.quant_mode = quant_mode
        self.per_channel = per_channel

        if self.quant_mode == "none":
            self.act_function = None
        elif self.quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def __repr__(self):
        return "{0}(activation_bit={1}, " \
               "quant_mode: {2}, Act_min: {3:.2f}, " \
               "Act_max: {4:.2f})".format(self.__class__.__name__, self.activation_bit,
                                          self.quant_mode, self.x_min.item(), self.x_max.item())
    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False
        
    def unfix(self):
        """
        unfix the activation range by setting running stat
        """
        self.running_stat = True

    def forward(self, x, 
                pre_act_scaling_factor=None, 
                identity=None, 
                identity_scaling_factor=None,
                specified_min=None,
                specified_max=None):
        # collect runnng stats
        x_act = x if identity is None else identity + x
        if self.running_stat:
            if not self.percentile:
                if not self.per_channel:
                    x_min = x_act.data.min()
                    x_max = x_act.data.max()
                else:
                    x_min = x_act.data.min(axis=0).values.min(axis=0).values
                    x_max = x_act.data.max(axis=0).values.max(axis=0).values
            else:
                raise NotImplementedError("percentile mode is not currently supported.")

            # Initialization
            if torch.eq(self.x_min, self.x_max).all():
                print("CHECK1")
                self.x_min = self.x_min + x_min
                self.x_max = self.x_max + x_max

            # exponential moving average (EMA)
            # use momentum to prevent the quantized values change greatly every iteration
            elif self.act_range_momentum == -1:
                print("CHECK2")
                self.x_min = torch.min(self.x_min, x_min)
                self.x_max = torch.max(self.x_max, x_max)
            else:
                print("[DEBUG] before x_min: ", self.x_min)
                self.x_min = self.x_min * self.act_range_momentum +\
                        x_min * (1 - self.act_range_momentum)
                print("[DEBUG] next x_min: ", self.x_min)
                print("[DEBUG] before x_max: ", self.x_max)
                self.x_max = self.x_max * self.act_range_momentum +\
                        x_max * (1 - self.act_range_momentum)
                print("[DEBUG] next x_max: ", self.x_max)

        print("[DEBUG] act_range_momentum: ", self.act_range_momentum)
        print("[DEBUG] x_max: ", self.x_max)
        print("[DEBUG] dtype: ", self.x_max.dtype)
        if self.quant_mode == 'none':
            return x_act, None

        assert self.quant_mode == 'symmetric', \
                "unsupported quant mode: {}".format(quant_mode)
        
        x_min = self.x_min if specified_min is None else specified_min
        x_max = self.x_max if specified_max is None else specified_max

        self.act_scaling_factor = symmetric_linear_quantization_params(
            self.activation_bit, x_min, x_max, 
            per_channel=self.per_channel)
        print("[DEBUG] pre_act_scaling_factor: ", self.act_scaling_factor)
        print("[DEBUG] x_min, x_max: ", x_min, x_max)
        if pre_act_scaling_factor is None:
            # this is for the input quantization 
            quant_act_int = self.act_function(x, self.activation_bit, \
                    self.percentile, self.act_scaling_factor)
        else:
            quant_act_int = fixedpoint_mul.apply(
                    x, pre_act_scaling_factor, 
                    self.activation_bit, self.quant_mode, 
                    self.act_scaling_factor, 
                    identity, identity_scaling_factor)

        correct_output_scale = self.act_scaling_factor.view(-1)
        return quant_act_int * correct_output_scale, self.act_scaling_factor

class SymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """
    @staticmethod
    def forward(ctx, x, k, percentile_mode=False, specified_scale=None):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of SymmetricQuantFunction requires pre-calculated scaling factor.
        specified_scale: pre-calculated scaling factor for the tensor x
        """
        
        if specified_scale is not None:
            scale = specified_scale

        zero_point = torch.tensor(0.).cuda()

        n = 2 ** (k - 1) - 1
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        new_quant_x = torch.clamp(new_quant_x, -n, n-1)

        ctx.scale = scale 
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):

        scale = ctx.scale
        if len(grad_output.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
        # reshape scale and zeropoint for linear weights
        elif len(grad_output.shape) == 2:
            scale = scale.view(-1, 1)
        else:
            scale = scale.view(-1)

        return grad_output.clone() / scale, None, None, None, None

def symmetric_linear_quantization_params(num_bits,
                                        saturation_min,
                                        saturation_max,
                                        per_channel=False):
    """
    Compute the scaling factor with the given quantization range for symmetric quantization.

    Parameters:
    ----------
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    """
    # in this part, we do not need any gradient computation,
    # in order to enfore this, we put torch.no_grad()
    with torch.no_grad():
        n = 2 ** (num_bits - 1) - 1

        if per_channel:
            scale, _ = torch.max(torch.stack([saturation_min.abs(), saturation_max.abs()], dim=1), dim=1)
            scale = torch.clamp(scale, min=1e-8) / n 

        else:
            scale = max(saturation_min.abs(), saturation_max.abs())
            scale = torch.clamp(scale, min=1e-8) / n 

    return scale 

class fixedpoint_mul(Function):
    """
    Function to perform fixed-point arthmetic that can match integer arthmetic on hardware.

    Parameters:
    ----------
    pre_act: input tensor
    pre_act_scaling_factor: ithe scaling factor of the input tensor
    bit_num: quantization bitwidth
    quant_mode: The mode for quantization, 'symmetric' or 'asymmetric'
    z_scaling_factor: the scaling factor of the output tensor
    identity: identity tensor
    identity_scaling_factor: the scaling factor of the identity tensor
    """
    @ staticmethod
    def forward (ctx, pre_act, pre_act_scaling_factor, 
                 bit_num, quant_mode, z_scaling_factor, 
                 identity=None, identity_scaling_factor=None):

        #TODO(Sehoon): May require other type of reshape
        if len(pre_act_scaling_factor.shape) == 3:
            reshape = lambda x : x
        else:
            reshape = lambda x : x.view(1, 1, -1)
        ctx.identity = identity

        if quant_mode == 'symmetric':
            n = 2 ** (bit_num - 1) - 1
        else:
            n = 2 ** bit_num - 1

        with torch.no_grad():
            pre_act_scaling_factor = reshape(pre_act_scaling_factor)
            if identity is not None:
                identity_scaling_factor = reshape(identity_scaling_factor)

            ctx.z_scaling_factor = z_scaling_factor
            
            z_int = torch.round(pre_act / pre_act_scaling_factor) 
            _A = pre_act_scaling_factor.type(torch.double)
            _B = (z_scaling_factor.type(torch.float)).type(torch.double)
            print("[DEBUG] A: ", _A)
            print("[DEBUG] B: ", _B)
            new_scale = _A / _B
            new_scale = reshape(new_scale)
            print("[DEBUG]  new_scaalee: ", new_scale)
            m, e = batch_frexp(new_scale)

            output = z_int.type(torch.double) * m.type(torch.double)
            output = torch.round( output / (2.0**e) )

            if identity is not None:
                # needs addition of identity activation
                wx_int = torch.round(identity / identity_scaling_factor)

                _A = identity_scaling_factor.type(torch.double)
                _B = (z_scaling_factor.type(torch.float)).type(torch.double)
                new_scale = _A / _B
                new_scale = reshape(new_scale)

                m1, e1 = batch_frexp(new_scale)
                output1 = wx_int.type(torch.double) * m1.type(torch.double)
                output1 = torch.round(output1 / (2.0**e1))

                output = output1 + output

            if bit_num in [4, 8, 16]:
                if quant_mode == 'symmetric':
                    return torch.clamp( output.type(torch.float), -n - 1, n)
                else:
                    return torch.clamp( output.type(torch.float), 0, n)
            else:
                return output.type(torch.float)

    @ staticmethod
    def backward(ctx, grad_output):
        identity_grad = None
        if ctx.identity is not None:
            identity_grad = grad_output.clone() / ctx.z_scaling_factor
        return grad_output.clone() / ctx.z_scaling_factor, None, None, None, None,\
                identity_grad, None

def batch_frexp(inputs, max_bit=31):
    """
    Decompose the scaling factor into mantissa and twos exponent.

    Parameters:
    ----------
    inputs: scaling factor
    return: (mantissa, exponent)
    """

    shape_of_input = inputs.size()

    # trans the input to be a 1-d tensor
    inputs = inputs.view(-1)
    
    output_m, output_e = np.frexp(inputs.cpu().numpy())
    tmp_m = []
    for m in output_m:
        int_m_shifted = int(Decimal(m * (2**max_bit)).quantize(Decimal('1'), 
            rounding=decimal.ROUND_HALF_UP))
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)

    output_e = float(max_bit) - output_e

    return torch.from_numpy( output_m ).cuda().view(shape_of_input), \
           torch.from_numpy( output_e ).cuda().view(shape_of_input)

def freeze_model(model):
    """
    freeze the activation range. Resursively invokes layer.fix()
    """
    if type(model) in [QuantAct]:
        model.fix()
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            freeze_model(m)
    elif type(model) == nn.ModuleList:
        for n in model:
            freeze_model(n)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                freeze_model(mod)


def unfreeze_model(model):
    """
    unfreeze the activation range. Resursively invokes layer.unfix()
    """
    if type(model) in [QuantAct]:
        model.unfix()
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            unfreeze_model(m)
    elif type(model) == nn.ModuleList:
        for n in model:
            unfreeze_model(n)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                unfreeze_model(mod)