import torch

from .utils import floor_ste

def int_exp_shift(x_int, scaling_factor, n=15):
    print("torch int exp x_int first: ", x_int[0,0,1,15:18])
    x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2 ** 4)
    with torch.no_grad():
        x0_int = torch.floor(-1.0 / scaling_factor)
    print("torch shift x_int: ", x_int[0,0,1,15:18])
    print("max value: ", n * x0_int)
    #x_int = torch.max(x_int, n * x0_int)
    #print("torch x0_int*15: ", x0_int*15)
    print("torch int after max: ", x_int[0,0,1,15:18])
    print("torch x0_int: ", x0_int)
    q = floor_ste.apply(x_int / x0_int)
    print("torch q: ", q[0,0,1,15:18])
    r = x_int - x0_int * q
    print("torch r: ", r[0,0,1,15:18])
    exp_int = r/2 - x0_int
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