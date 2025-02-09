import math
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

def improved_int_exp(I, S, Q=16, N=16):

    def fixed_point_mul(X, Y, Q):
        print("X: ", X)
        print("Y: ", Y)
        return (X * Y) >> Q

    # Precompute fixed-point multiplier for conversion to base-2.
    # M_inv = 2^Q / ln2, rounded to nearest integer.
    M_inv = int(round((1 << Q) / math.log(2)))
    
    # Compute y = I * S * M_inv in fixed-point.
    # Here we assume I and S are given such that S * I is representable.
    I_int = I.to(torch.int64)
    S_int = (torch.round(S * (1 << Q))).to(torch.int64)

    y = fixed_point_mul(I, S_int, Q)  # e.g. treat S as fixed-point if needed.
    y = fixed_point_mul(y, M_inv, Q)  # now y is in Q-format.
    
    # Extract integer and fractional parts:
    k = y >> Q           # integer part
    r = y - (k << Q)     # fractional part (0 <= r < 2^Q)
    
    # Now, approximate 2^(r/2^Q) using a polynomial.
    # Let x = r / 2^Q be the fractional value.
    # For example, choose a third-order polynomial:
    #   2^x ≈ c0 + c1*x + c2*x^2 + c3*x^3.
    # In fixed point, precompute coefficients c0, c1, c2, c3 (scaled by 2^Q).
    # The coefficients might be, for instance, (in Q-format):
    c0 = int(round(1 * (1 << Q)))        # ideally 1.0
    c1 = int(round(math.log(2) * (1 << Q)))  # ln2
    c2 = int(round((math.log(2)**2) * (1 << Q) / 2))
    c3 = int(round((math.log(2)**3) * (1 << Q) / 6))
    
    # Let x = r, which is already in fixed-point with Q fractional bits.
    # Compute the polynomial approximation. Use fixed-point arithmetic:
    # poly = c0 + (c1 * x >> Q) + ((c2 * x * x) >> (2*Q)) + ((c3 * x * x * x) >> (3*Q))
    poly = c0
    poly += (c1 * r) >> Q
    poly += (c2 * r * r) >> (2 * Q)
    poly += (c3 * r * r * r) >> (3 * Q)
    
    # poly now approximates 2^(r/2^Q) in Q-format.
    
    # Multiply by 2^k. In integer arithmetic, this is a bit-shift.
    # Compute 2^k and 2^(-k) elementwise. Using torch.pow:
    factor_pos = torch.pow(2, k)         # For k >= 0
    factor_neg = torch.pow(2, -k.clamp(max=0))  # For k < 0

    # Use torch.where to choose the appropriate scaling.
    # Make sure poly, factor_pos, factor_neg are integer tensors.
    I_exp = torch.where(k >= 0, poly * factor_pos, poly // factor_neg)
    
    # Adjust the scaling factor for the result.
    # For example, if you wish S_exp such that S_exp * I_exp approximates e^(S*I):
    S_exp = S / (1 << N)  # where N is chosen appropriately.
    
    return I_exp, S_exp