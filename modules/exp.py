import torch

from .utils import floor_ste

def int_exp_shift(x_int, scaling_factor, n=60):
    print("torch int exp x_int first: ", x_int[0,0,1,15:18])
    x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2 ** 4)
    with torch.no_grad():
        x0_int = torch.floor(-1.0 / scaling_factor)
    print("torch shift x_int: ", x_int[0,0,1,15:18])
    print("x0_int: ", x0_int)
    print("max value: ", n * x0_int)
    x_int = torch.max(x_int, n * x0_int)
    #print("torch x0_int*15: ", x0_int*15)
    print("torch int after max: ", x_int[0,0,1,15:18])
    print("torch x0_int: ", x0_int)
    q = floor_ste.apply(x_int / x0_int)
    print("torch q: ", q[0,0,1,15:18])
    print("q min, max: ", q.min(), ", ", q.max())
    r = x_int - x0_int * q
    print("torch r: ", r[0,0,1,15:18])
    print("r min, max: ", r.min(), ", ", r.max())
    exp_int = torch.floor(r/2) - x0_int
    print("after exp_int: ", exp_int[0,0,1,15:18])
    exp_before_clamp = exp_int * 2 ** (n - q)
    print("before clamp exp shift: ", exp_before_clamp[0,0,1,15:18])
    exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (n - q)), min=0)
    print("after clamp exp shift: ", exp_int[0,0,1,15:18])
    exp_int_max = torch.max(exp_int)
    print("torch exp max aftter shift: ", exp_int_max)
    scaling_factor = scaling_factor / 2 ** n
    #print("torch fianl exp_float: ", exp_int * scaling_factor)
    return exp_int, scaling_factor

def int_exp_poly(x_int, scaling_factor):

    def int_polynomial(x_int, scaling_factor):
        coef = [0.35815147, 0.96963238, 1.]  # ax**2 + bx + c
        coef[1] /= coef[0]
        coef[2] /= coef[0]
        b_int = torch.floor(coef[1] / scaling_factor)
        c_int = torch.floor(coef[2] / scaling_factor**2)
        z = x_int + b_int
        z = x_int * z
        z = z + c_int
        scaling_factor = coef[0] * scaling_factor**2
        return z, scaling_factor

    def int_exp(x_int, scaling_factor):
        x0 = -0.6931  # -ln2
        n = 30  # sufficiently large integer
        x0_int = torch.floor(x0 / scaling_factor)
        print("before clamp x_int: ", x_int[  0,   0,   4,  45:48])
        print("x0_int: ", x0_int)
        x_int = torch.max(x_int, n * x0_int)
        print("after clamp x_int: ", x_int[  0,   0,   4,  45:48])
        q = torch.floor(x_int / x0_int)
        print("q: ", q[  0,   0,   4,  45:48])
        r = x_int - x0_int * q
        print("r: ", q[  0,   0,   4,  45:48])
        exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
        exp_int = torch.clamp(torch.floor(exp_int * 2**(n - q)), min=0)
        scaling_factor = exp_scaling_factor / 2**n
        return exp_int, scaling_factor

    exp_int, exp_scaling_factor = int_exp(x_int, scaling_factor)
    exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
    return exp_int, exp_scaling_factor