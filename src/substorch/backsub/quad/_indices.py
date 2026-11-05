__all__ = [
    "cal_kwargs_data_of_quad_relaxation_gemm",
    "cal_kwargs_data_of_quad_relaxation_conv2d",
    "cal_prod_weight_bias",
    "cal_square_bias",
    "cal_mul_weight_2bias",
    "cal_prod_weight_weight",
]


import torch
from torch import Tensor


def cal_kwargs_data_of_quad_relaxation_gemm(l: Tensor, u: Tensor) -> dict[str, Tensor]:
    l = l.flatten()
    u = u.flatten()

    h = (l + u) * 0.5
    sqr_l, sqr_u, sqr_h = l**2, u**2, h**2
    k_lu = torch.where(u != l, (sqr_u - sqr_l) / (u - l), 2 * l)
    k_h = 2 * h

    lx, ux = l.unsqueeze(0), u.unsqueeze(0)
    ly, uy = l.unsqueeze(1), u.unsqueeze(1)
    mul_lx_ly, mul_ux_uy = lx * ly, ux * uy
    mul_lx_uy, mul_ux_ly = lx * uy, ux * ly
    add_lx_ux, add_ly_uy = lx + ux, ly + uy

    cached_data = {
        "h": h,
        "sqr_l": sqr_l,
        "sqr_u": sqr_u,
        "sqr_h": sqr_h,
        "k_lu": k_lu,
        "k_h": k_h,
        "lx": lx,
        "ux": ux,
        "ly": ly,
        "uy": uy,
        "mul_lx_ly": mul_lx_ly,
        "mul_ux_uy": mul_ux_uy,
        "mul_lx_uy": mul_lx_uy,
        "mul_ux_ly": mul_ux_ly,
        "add_lx_ux": add_lx_ux,
        "add_ly_uy": add_ly_uy,
    }

    return cached_data


def cal_kwargs_data_of_quad_relaxation_conv2d(
    l: Tensor,
    u: Tensor,
    *,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
) -> dict[str, Tensor]:
    cached_data = cal_kwargs_data_of_quad_relaxation_gemm(l, u)

    c_in, h_in, w_in = input_size
    c_out, h_out, w_out = output_size
    chw_in = c_in * h_in * w_in
    kh, kw = kernel_size
    ph, pw = padding
    dh, dw = dilation
    sh, sw = stride
    kk = kh * kw
    ckk = c_in * kk
    kwargs = {"device": l.device}

    h_out_i, w_out_i, c_in_i, kh_i, kw_i, c_in_j, kh_j, kw_j = torch.meshgrid(
        torch.arange(h_out, **kwargs),
        torch.arange(w_out, **kwargs),
        torch.arange(c_in, **kwargs),
        torch.arange(kh, **kwargs),
        torch.arange(kw, **kwargs),
        torch.arange(c_in, **kwargs),
        torch.arange(kh, **kwargs),
        torch.arange(kw, **kwargs),
        indexing="ij",
    )

    h_in_i = h_out_i * sh + kh_i * dh - ph
    h_in_j = h_out_i * sh + kh_j * dh - ph
    w_in_i = w_out_i * sw + kw_i * dw - pw
    w_in_j = w_out_i * sw + kw_j * dw - pw

    valid_i = torch.nonzero(
        (0 <= h_in_i)
        & (h_in_i < h_in)
        & (0 <= h_in_j)
        & (h_in_j < h_in)
        & (0 <= w_in_i)
        & (w_in_i < w_in)
        & (0 <= w_in_j)
        & (w_in_j < w_in),
        as_tuple=True,
    )

    h_out_i, w_out_i = h_out_i[valid_i], w_out_i[valid_i]
    c_in_i, h_in_i, w_in_i = c_in_i[valid_i], h_in_i[valid_i], w_in_i[valid_i]
    c_in_j, h_in_j, w_in_j = c_in_j[valid_i], h_in_j[valid_i], w_in_j[valid_i]
    kh_i, kw_i = kh_i[valid_i], kw_i[valid_i]
    kh_j, kw_j = kh_j[valid_i], kw_j[valid_i]

    idx_i = c_in_i * h_in * w_in + h_in_i * w_in + w_in_i
    idx_j = c_in_j * h_in * w_in + h_in_j * w_in + w_in_j
    idx_in = idx_i * chw_in + idx_j

    idx_out = (
        h_out_i * w_out * ckk * ckk
        + w_out_i * ckk * ckk
        + c_in_i * kk * ckk
        + kh_i * kw * ckk
        + kw_i * ckk
        + c_in_j * kk
        + kh_j * kw
        + kw_j
    )
    idx_out2 = None

    cached_data.update(
        {
            "idx_in": idx_in,
            "idx_out": idx_out,
            "idx_out2": idx_out2,
        }
    )

    return cached_data


def cal_prod_weight_bias(weight: Tensor, bias: Tensor) -> Tensor:
    weight_bias = torch.cat([weight, bias.unsqueeze(1)], dim=1)
    weight_bias_prod = torch.einsum("mi, mj -> mij", weight_bias, weight_bias)
    return weight_bias_prod


def cal_square_bias(bias: Tensor) -> Tensor:
    return torch.square(bias)


def cal_mul_weight_2bias(weight: Tensor, bias: Tensor) -> Tensor:
    weight_flat = weight.flatten(1)
    weight_mul_2bias = (weight_flat * (2 * bias).unsqueeze(1)).reshape(*weight.shape)
    return weight_mul_2bias


def cal_prod_weight_weight(weight: Tensor) -> Tensor:
    weight_flat = weight.flatten(1)
    weight_prod = weight_flat.unsqueeze(2) * weight_flat.unsqueeze(1)
    return weight_prod.flatten(1)
