import torch

import triton
import triton.language as tl

def int_exp_poly(x_int, scaling_factor, subtract_scaling_factor):
    x0 = -0.6931 # -ln2
    n = 30 # sufficiently large integer
    coef = [0.35815147, 0.96963238, 1.] # ax**2 + bx + c
    coef[1] /= coef[0]
    coef[2] /= coef[0]

    def int_polynomial(x_int, scaling_factor):
        with torch.no_grad():
            b_int = torch.floor(coef[1] / scaling_factor)
            c_int = torch.floor(coef[2] / scaling_factor ** 2)
        z = x_int + b_int
        z = x_int * z
        z = z + c_int
        scaling_factor = coef[0] * scaling_factor ** 2
        return z, scaling_factor

    def int_exp(x_int, scaling_factor):
        with torch.no_grad():
            x0_int = torch.floor(x0 / scaling_factor)
        x_int = torch.max(x_int, n * x0_int)
        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (n - q)), min=0)
        scaling_factor = exp_scaling_factor / 2 ** n
        return exp_int, scaling_factor

    return int_exp(x_int, scaling_factor_exp)

@triton.jit
def _int_exp_poly(x_int_ptr, x0_int, b_int, c_int, out_ptr, scale_factor,
                  row_stride: tl.constexpr, out_stride: tl.constexpr):
    row_id = tl.program_id(0)
    input_ptr = x_int_ptr + row_id * row_stride
    offset = tl.arange(0, 8)
    input_ptrs = input_ptr + offset
    mask = offset < 8
    x_int = tl.load(input_ptrs, mask=mask, other=0.0)

    x_int = x_int.to(tl.int32)
    x0_int = x0_int.to(tl.int32)
    x_int = tl.maximum(x_int, x0_int*30)

    q = x_int // x0_int
    r = x_int - x0_int * q

    # int poly
    z = x_int + b_int
    z = x_int * z
    z = z + c_int
    poly_scaling_factor = 0.35815147 * scale_factor * scale_factor

    # int exp
    exp_int = tl.floor(exp_int << (30-q))
    exp_int = tl.clamp(exp_int, min=0)

    out_ptrs = out_ptr + row_id * out_stride
    tl.store(out_ptrs, exp_int)

class _exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_int, x_scale, bit=8):
        input_float = x_int * x_scale
        input_float_max = torch.amax(input_float, dim=-1, keepdim=True)
        input_float_subtract = input_float - input_float_max

        subtract_scale = get_scale(input_float_subtract, bit)
        subtract_int = quantize(input_float_subtract, subtract_scale, bit)
        subtract_int = subtract_int.to(torch.int16)
        out = torch.empty_like(subtract_int).to(dtype=torch.int32)

        x0 = -0.6931
        coef = [0.35815147, 0.96963238, 1.] # ax**2 + bx + c
        coef[1] /= coef[0]
        coef[2] /= coef[0]
        
        with torch.no_grad():
            x0_int = torch.floor(torch.floor(x0/x_scale)).to(torch.int32)
            b_int = torch.floor(coef[1] / x_scale).to(torch.int32)
            c_int = torch.floor(coef[2] / x_scale ** 2).to(torch.int32)

        grid = lambda args: (4,)
        _int_exp_poly[grid](
            x_int, x0_int, b_int, c_int, out, x_scale,
            x_int.stride(0), out.stride(0)
        )

        return out

triton_exp = _exp.apply
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor = torch.randn((4,8), dtype=torch.float32, device=device)
input_bit 
