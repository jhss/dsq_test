import torch
from .utils import floor_ste

def ibert_exp(x_int, scaling_factor_exp):
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
        print("x0_int: ", x0_int)
        print("x_int: ", x_int)
        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (n - q)), min=0)
        scaling_factor = exp_scaling_factor / 2 ** n
        return exp_int, scaling_factor

    return int_exp(x_int, scaling_factor_exp)