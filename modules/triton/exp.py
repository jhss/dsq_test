import os
os.environ["TRITON_INTERPRET"] = "1"

import torch

import triton
import triton.language as tl

from ..utils import *

def int_exp_poly(x_int, scaling_factor):
    x0 = -0.6931 # -ln2
    n = 15 # sufficiently large integer
    coef = [0.35815147, 0.96963238, 1.] # ax**2 + bx + c
    coef[1] /= coef[0]
    coef[2] /= coef[0]

    def int_polynomial(x_int, scaling_factor):
        with torch.no_grad():
            b_int = torch.floor(coef[1] / scaling_factor)
            c_int = torch.floor(coef[2] / scaling_factor ** 2)
        print("b_int: ", b_int)
        z = x_int + b_int
        print("after b_int: ", z)
        z = x_int * z
        print("after z: ", z)
        z = z + c_int
        scaling_factor = coef[0] * scaling_factor ** 2
        return z, scaling_factor

    def int_exp(x_int, scaling_factor):
        with torch.no_grad():
            x0_int = torch.floor(x0 / scaling_factor)
        x_int = torch.max(x_int, n * x0_int)
        print("x_int: ", x_int)
        print("x0_int: ", x0_int)
        q = floor_ste.apply(x_int / x0_int)
        print("q: ", q)
        r = x_int - x0_int * q
        print("r: ", r)
        exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
        print("after int_poly: ", exp_int)
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (n - q)), min=0)
        print("after shift: ", exp_int)
        scaling_factor = exp_scaling_factor / 2 ** n
        return exp_int, scaling_factor

    return int_exp(x_int, scaling_factor)

@triton.jit
def _int_exp_poly(x_int_ptr, x0_int, b_int, c_int, out_ptr,
                  row_stride: tl.constexpr, out_stride: tl.constexpr):
    row_id = tl.program_id(0)
    input_ptr = x_int_ptr + row_id * row_stride
    offset = tl.arange(0, 8)
    input_ptrs = input_ptr + offset
    mask = offset < 8
    x_int = tl.load(input_ptrs, mask=mask, other=0.0)

    x_int = x_int.to(tl.int64)
    x0_int = x0_int.to(tl.int64)
    x_int = tl.maximum(x_int, x0_int*15)
    print("x_int: ", x_int)
    print("x0_int: ", x0_int)
    q = x_int // x0_int
    r = x_int - x0_int * q
    print("q: ", q)
    print("r: ", r)
    # int poly
    z = x_int + b_int
    print("after b_int: ", z)
    z = x_int * z
    print("after z: ", z)
    exp_int = z + c_int

    # int exp
    print("after int poly exp_int: ", exp_int)
    exp_int = exp_int << (15-q)
    print("after shift: ", exp_int)
    #exp_int = tl.clamp(exp_int, min=0)
    #exp_int = tl.minimum(exp_int, 0)

    out_ptrs = out_ptr + row_id * out_stride + offset
    tl.store(out_ptrs, exp_int)

class _exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_int, x_scale, bit=16):
        # input_float = x_int * x_scale
        # input_float_max = torch.amax(input_float, dim=-1, keepdim=True)
        # input_float_subtract = input_float - input_float_max

        # subtract_scale = get_scale(input_float_subtract, bit)
        # subtract_int = quantize(input_float_subtract, subtract_scale, bit)
        # subtract_int = subtract_int.to(torch.int16)

        out = torch.empty_like(x_int).to(dtype=torch.int32)

        x0 = -0.6931
        coef = [0.35815147, 0.96963238, 1.] # ax**2 + bx + c
        coef[1] /= coef[0]
        coef[2] /= coef[0]
        n=15
        
        with torch.no_grad():
            x0_int = torch.floor(torch.floor(x0/x_scale)).to(torch.int32).item()
            b_int = torch.floor(coef[1] / x_scale).to(torch.int32).item()
            c_int = torch.floor(coef[2] / x_scale ** 2).to(torch.int32).item()
        print("b_int: ", b_int)
        grid = lambda args: (4,)
        _int_exp_poly[grid](
            x_int, x0_int, b_int, c_int, out,
            x_int.stride(0), out.stride(0)
        )

        exp_scale = coef[0] * x_scale ** 2 / 2 ** n

        return out, exp_scale


