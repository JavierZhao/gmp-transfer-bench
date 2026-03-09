''' Particle Transformer (ParT)

Paper: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772
'''
import math
import random
import warnings
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from networks.logger import _logger


@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part


def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps)**(-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = ((pti <= ptj) * pti + (pti > ptj) * ptj) if for_onnx else torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    # the following features are not symmetric for (i, j)
    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    assert (len(outputs) == num_outputs)
    return torch.cat(outputs, dim=1)


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len)
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len)
    i = torch.cat((
        torch.arange(0, batch_size, device=uu.device).repeat_interleave(num_fts * num_pairs).unsqueeze(0),
        torch.arange(0, num_fts, device=uu.device).repeat_interleave(num_pairs).repeat(batch_size).unsqueeze(0),
        idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0),
        idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0),
    ), dim=0)
    return torch.sparse_coo_tensor(
        i, uu.flatten(),
        size=(batch_size, num_fts, seq_len + 1, seq_len + 1),
        device=uu.device).to_dense()[:, :, :seq_len, :seq_len]


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # From https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/weight_init.py#L8
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class SequenceTrimmer(nn.Module):

    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def forward(self, x, v=None, mask=None, uu=None, points=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
        # points: (N, C_points, P), generic point coordinates
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled:
            if self._counter < 5:
                self._counter += 1
            else:
                if self.training:
                    q = min(1, random.uniform(*self.target))
                    maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
                    rand = torch.rand_like(mask.type_as(x))
                    rand.masked_fill_(~mask, -1)
                    perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P)
                    mask = torch.gather(mask, -1, perm)
                    x = torch.gather(x, -1, perm.expand_as(x))
                    if v is not None:
                        v = torch.gather(v, -1, perm.expand_as(v))
                    if points is not None:
                        points = torch.gather(points, -1, perm.expand_as(points))
                    if uu is not None:
                        uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                        uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                else:
                    maxlen = mask.sum(dim=-1).max()
                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = v[:, :, :maxlen]
                    if points is not None:
                        points = points[:, :, :maxlen]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]

        return x, v, mask, uu, points


class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation='gelu'):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend([
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()
        # x: (seq_len, batch, embed_dim)
        return self.embed(x)


def quantize_coordinates(c1: torch.Tensor, c2: torch.Tensor, grid_size: float, pad: torch.Tensor | None = None):
    if pad is None:
        valid = torch.ones_like(c1, dtype=torch.bool)
    else:
        valid = ~pad

    c1_for_min = c1.masked_fill(~valid, float("inf"))
    c2_for_min = c2.masked_fill(~valid, float("inf"))
    c1_min = c1_for_min.min(dim=1, keepdim=True).values
    c2_min = c2_for_min.min(dim=1, keepdim=True).values
    c1_min = torch.where(torch.isfinite(c1_min), c1_min, torch.zeros_like(c1_min))
    c2_min = torch.where(torch.isfinite(c2_min), c2_min, torch.zeros_like(c2_min))

    c1_shift = (c1 - c1_min).masked_fill(~valid, 0.0)
    c2_shift = (c2 - c2_min).masked_fill(~valid, 0.0)
    coord_eta = c1_shift / grid_size
    coord_phi = c2_shift / grid_size

    grid_eta = coord_eta.floor().to(torch.long)
    grid_phi = coord_phi.floor().to(torch.long)
    grid_eta = grid_eta.masked_fill(~valid, 0)
    grid_phi = grid_phi.masked_fill(~valid, 0)

    if valid.any().item():
        H = int(grid_eta.max().item()) + 1
        W = int(grid_phi.max().item()) + 1
    else:
        H, W = 1, 1

    H = max(H, 1)
    W = max(W, 1)
    grid_eta = grid_eta.clamp(0, H - 1)
    grid_phi = grid_phi.clamp(0, W - 1)

    B, P = c1.shape
    batch_idx = torch.arange(B, device=c1.device).view(B, 1).expand(B, P)
    flat_local = grid_eta * W + grid_phi
    return {
        "valid": valid,
        "grid_eta": grid_eta,
        "grid_phi": grid_phi,
        "coord_eta": coord_eta,
        "coord_phi": coord_phi,
        "batch_idx": batch_idx,
        "flat_local": flat_local,
        "height": H,
        "width": W,
    }


def scatter_to_grid(
        x: torch.Tensor,
        quantized: dict,
        scatter_reduce: str = "sum",
        eps: float = 1e-6):
    B, _, C = x.shape
    H = quantized["height"]
    W = quantized["width"]
    HW = H * W
    flat_idx = (quantized["batch_idx"] * HW + quantized["flat_local"]).reshape(-1)
    valid = quantized["valid"].reshape(-1).to(x.dtype)

    grid_flat = x.new_zeros((B * HW, C))
    values = (x * quantized["valid"].unsqueeze(-1).to(x.dtype)).reshape(-1, C)
    grid_flat.scatter_add_(0, flat_idx[:, None].expand(-1, C), values)

    counts = x.new_zeros((B * HW,))
    counts.scatter_add_(0, flat_idx, valid)
    if scatter_reduce == "mean":
        grid_flat = grid_flat / (counts[:, None] + eps)

    grid = grid_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    return grid, counts.view(B, HW)


def gather_from_grid(grid: torch.Tensor, quantized: dict) -> torch.Tensor:
    grid_bhwc = grid.permute(0, 2, 3, 1).contiguous()
    gathered = grid_bhwc[quantized["batch_idx"], quantized["grid_eta"], quantized["grid_phi"]]
    return gathered * quantized["valid"].unsqueeze(-1).to(gathered.dtype)


def grid_sample_coordinates(quantized: dict) -> torch.Tensor:
    H = quantized["height"]
    W = quantized["width"]
    coord_eta = quantized["coord_eta"]
    coord_phi = quantized["coord_phi"]

    if W > 1:
        norm_x = 2.0 * coord_phi / (W - 1) - 1.0
    else:
        norm_x = torch.zeros_like(coord_phi)
    if H > 1:
        norm_y = 2.0 * coord_eta / (H - 1) - 1.0
    else:
        norm_y = torch.zeros_like(coord_eta)

    coords = torch.stack((norm_x, norm_y), dim=-1)
    return coords.masked_fill(~quantized["valid"].unsqueeze(-1), 0.0)


def select_active_grid_tokens(grid: torch.Tensor, counts: torch.Tensor):
    B, C, H, W = grid.shape
    HW = H * W
    occupied = counts > 0
    num_tokens = occupied.sum(dim=1)
    max_tokens = int(num_tokens.max().item()) if num_tokens.numel() > 0 else 0
    if max_tokens == 0:
        return None, None

    dense_tokens = grid.permute(0, 2, 3, 1).reshape(B, HW, C)
    flat_positions = torch.arange(HW, device=grid.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    sort_keys = torch.where(occupied, flat_positions, flat_positions + HW)
    top_idx = sort_keys.argsort(dim=1)[:, :max_tokens]
    token_mask = torch.arange(max_tokens, device=grid.device).unsqueeze(0) < num_tokens.unsqueeze(1)
    tokens = dense_tokens.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, C))
    tokens = tokens.masked_fill(~token_mask.unsqueeze(-1), 0.0)
    return tokens, token_mask


class PairEmbed(nn.Module):
    def __init__(
            self, pairwise_lv_dim, pairwise_input_dim, dims,
            remove_self_pair=False, use_pre_activation_pair=True, mode='sum',
            normalize_input=True, activation='gelu', eps=1e-8,
            for_onnx=False):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.for_onnx = for_onnx
        self.pairwise_lv_fts = partial(pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps, for_onnx=for_onnx)
        self.out_dim = dims[-1]

        if self.mode == 'concat':
            input_dim = pairwise_lv_dim + pairwise_input_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend([
                    nn.Conv1d(input_dim, dim, 1),
                    nn.BatchNorm1d(dim),
                    nn.GELU() if activation == 'gelu' else nn.ReLU(),
                ])
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)
        elif self.mode == 'sum':
            if pairwise_lv_dim > 0:
                input_dim = pairwise_lv_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.embed = nn.Sequential(*module_list)

            if pairwise_input_dim > 0:
                input_dim = pairwise_input_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.fts_embed = nn.Sequential(*module_list)
        else:
            raise RuntimeError('`mode` can only be `sum` or `concat`')

    def forward(self, x, uu=None):
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        assert (x is not None or uu is not None)
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else:
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric and not self.for_onnx:
                i, j = torch.tril_indices(seq_len, seq_len, offset=-1 if self.remove_self_pair else 0,
                                          device=(x if x is not None else uu).device)
                if x is not None:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                    xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
                    xj = x[:, :, j, i]
                    x = self.pairwise_lv_fts(xi, xj)
                if uu is not None:
                    # (batch, dim, seq_len*(seq_len+1)/2)
                    uu = uu[:, :, i, j]
            else:
                if x is not None:
                    x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                    if self.remove_self_pair:
                        i = torch.arange(0, seq_len, device=x.device)
                        x[:, :, i, i] = 0
                    x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.view(-1, self.pairwise_input_dim, seq_len * seq_len)
            if self.mode == 'concat':
                if x is None:
                    pair_fts = uu
                elif uu is None:
                    pair_fts = x
                else:
                    pair_fts = torch.cat((x, uu), dim=1)

        if self.mode == 'concat':
            elements = self.embed(pair_fts)  # (batch, embed_dim, num_elements)
        elif self.mode == 'sum':
            if x is None:
                elements = self.fts_embed(uu)
            elif uu is None:
                elements = self.embed(x)
            else:
                elements = self.embed(x) + self.fts_embed(uu)

        if self.is_symmetric and not self.for_onnx:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=elements.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)
        return y


class Block(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='gelu',
                 scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(1, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        if x_cls is not None:
            with torch.no_grad():
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)  # (seq_len+1, batch, embed_dim)
            u = self.pre_attn_norm(u)
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[0]  # (1, batch, embed_dim)
        else:
            residual = x
            x = self.pre_attn_norm(x)
            x = self.attn(x, x, x, key_padding_mask=padding_mask,
                          attn_mask=attn_mask)[0]  # (seq_len, batch, embed_dim)

        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum('tbhd,h->tbdh', x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=2,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 activation='gelu'):
        super().__init__()

        self.embed_dim = embed_dim
        self.ffn_dim = embed_dim * ffn_ratio
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout)
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim)
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

    def forward(self, query, context, query_padding_mask=None, context_padding_mask=None):
        residual = query
        q = self.query_norm(query)
        kv = self.context_norm(context)
        query = self.attn(q, kv, kv, key_padding_mask=context_padding_mask)[0]
        if query_padding_mask is not None:
            query = query.masked_fill(query_padding_mask.transpose(0, 1).unsqueeze(-1), 0.0)
        query = residual + self.dropout(query)

        residual = query
        query = self.pre_fc_norm(query)
        query = self.act(self.fc1(query))
        query = self.act_dropout(query)
        query = self.post_fc_norm(query)
        query = self.fc2(query)
        query = residual + self.dropout(query)
        if query_padding_mask is not None:
            query = query.masked_fill(query_padding_mask.transpose(0, 1).unsqueeze(-1), 0.0)
        return query


class GeometricMessagePassing(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        grid_size: float = 0.05,
        scatter_reduce: str = "sum",   # "sum" or "mean"
        eps: float = 1e-6,
    ):
        super().__init__()
        assert scatter_reduce in ("sum", "mean")
        self.channels = channels
        self.kernel_size = kernel_size
        self.grid_size = float(grid_size)
        self.scatter_reduce = scatter_reduce
        self.eps = eps

        self.conv2d = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=True,
        )
        self.pointwise = nn.Linear(channels, channels, bias=True)
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def encode_grid(self, x: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, pad: torch.Tensor | None = None):
        quantized = quantize_coordinates(c1, c2, self.grid_size, pad=pad)
        grid, counts = scatter_to_grid(
            x,
            quantized,
            scatter_reduce=self.scatter_reduce,
            eps=self.eps,
        )
        grid = self.conv2d(grid)
        gathered = gather_from_grid(grid, quantized)
        return {
            "grid": grid,
            "counts": counts,
            "quantized": quantized,
            "gathered": gathered,
            "sample_coords": grid_sample_coordinates(quantized),
        }

    def forward(self,
                x: torch.Tensor,
                c1: torch.Tensor,
                c2: torch.Tensor,
                pad: torch.Tensor | None = None,
                return_context: bool = False):
        B, _, C = x.shape
        assert C == self.channels
        residual = x

        context = self.encode_grid(x, c1, c2, pad=pad)
        out = self.pointwise(context["gathered"])
        out = self.norm(out)
        out = residual + out
        if return_context:
            return out, context
        return out


class GridParticleCrossAttention(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 num_heads=8,
                 num_layers=1,
                 dropout=0.1,
                 attn_dropout=0.1,
                 activation_dropout=0.1,
                 activation='gelu'):
        super().__init__()
        self.grid_blocks = nn.ModuleList([
            Block(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=2,
                dropout=dropout,
                attn_dropout=attn_dropout,
                activation_dropout=activation_dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ])
        self.cross_block = CrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=2,
            dropout=dropout,
            attn_dropout=attn_dropout,
            activation_dropout=activation_dropout,
            activation=activation,
        )

    def forward(self, x: torch.Tensor, grid_context: dict, pad: torch.Tensor | None = None) -> torch.Tensor:
        tokens, token_mask = select_active_grid_tokens(grid_context["grid"], grid_context["counts"])
        if tokens is None:
            return x

        grid_tokens = tokens.permute(1, 0, 2).contiguous()
        grid_padding_mask = ~token_mask
        for block in self.grid_blocks:
            grid_tokens = block(grid_tokens, padding_mask=grid_padding_mask)

        particle_tokens = x.permute(1, 0, 2).contiguous()
        particle_tokens = self.cross_block(
            particle_tokens,
            grid_tokens,
            query_padding_mask=pad,
            context_padding_mask=grid_padding_mask,
        )
        x = particle_tokens.permute(1, 0, 2).contiguous()
        if pad is not None:
            x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        return x


class SubstructureAwareDeformableAttention(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 num_heads=8,
                 num_samples=8,
                 offset_scale=2.0,
                 dropout=0.1,
                 activation_dropout=0.1,
                 activation='gelu'):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}.")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_samples = num_samples
        self.offset_scale = float(offset_scale)

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.offset_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Linear(embed_dim, num_samples * 2),
        )
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.distance_penalty = nn.Parameter(torch.ones(1), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim * 2)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(embed_dim * 2)
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x: torch.Tensor, grid_context: dict, pad: torch.Tensor | None = None) -> torch.Tensor:
        grid = grid_context["grid"]
        base_coords = grid_context["sample_coords"]
        B, P, _ = x.shape

        x_norm = self.pre_attn_norm(x)
        offsets = self.offset_mlp(x_norm).view(B, P, self.num_samples, 2)
        step_x = 2.0 / max(grid_context["quantized"]["width"] - 1, 1)
        step_y = 2.0 / max(grid_context["quantized"]["height"] - 1, 1)
        step = x.new_tensor([self.offset_scale * step_x, self.offset_scale * step_y]).view(1, 1, 1, 2)
        sample_coords = (base_coords.unsqueeze(2) + torch.tanh(offsets) * step).clamp(-1.0, 1.0)

        sampled = F.grid_sample(
            grid,
            sample_coords,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        sampled = sampled.permute(0, 2, 3, 1).contiguous()

        q = self.q_proj(x_norm).view(B, P, self.num_heads, self.head_dim)
        k = self.k_proj(sampled).view(B, P, self.num_samples, self.num_heads, self.head_dim)
        v = self.v_proj(sampled).view(B, P, self.num_samples, self.num_heads, self.head_dim)
        attn_logits = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(self.head_dim)
        distance_bias = (sample_coords - base_coords.unsqueeze(2)).square().sum(dim=-1, keepdim=True)
        attn_logits = attn_logits - self.distance_penalty.abs() * distance_bias

        if pad is not None:
            invalid_queries = pad.unsqueeze(-1).unsqueeze(-1)
            attn_logits = torch.where(invalid_queries, torch.zeros_like(attn_logits), attn_logits)

        attn = torch.softmax(attn_logits, dim=2)
        if pad is not None:
            attn = attn.masked_fill(invalid_queries, 0.0)

        out = (attn.unsqueeze(-1) * v).sum(dim=2).reshape(B, P, self.embed_dim)
        out = self.out_proj(out)
        if pad is not None:
            out = out.masked_fill(pad.unsqueeze(-1), 0.0)
        x = x + self.dropout(out)

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = residual + self.dropout(x)
        if pad is not None:
            x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        return x

def compute_eta_phi_from_p4(v: torch.Tensor, eps: float = 1e-8):
    """
    v: [N, 4, P] with [px, py, pz, E]
    returns eta, phi: [N, P], [N, P]
    """
    px, py, pz = v[:, 0, :], v[:, 1, :], v[:, 2, :]
    pt = torch.sqrt(px * px + py * py + eps)
    phi = torch.atan2(py, px)
    eta = torch.asinh(pz / (pt + eps))
    return eta, phi

def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi

def unwrap_phi_per_jet(phi: torch.Tensor, pad: torch.Tensor | None = None) -> torch.Tensor:
    """
    phi: (B, P) in radians
    pad: (B, P) bool, True where padded
    returns: dphi in (-pi, pi], centered per jet so seam is not a problem
    """
    if pad is None:
        sin_mean = torch.sin(phi).mean(dim=1, keepdim=True)
        cos_mean = torch.cos(phi).mean(dim=1, keepdim=True)
    else:
        w = (~pad).to(phi.dtype)  # 1 for real, 0 for padded
        denom = w.sum(dim=1, keepdim=True).clamp(min=1.0)
        sin_mean = (torch.sin(phi) * w).sum(dim=1, keepdim=True) / denom
        cos_mean = (torch.cos(phi) * w).sum(dim=1, keepdim=True) / denom

    phi0 = torch.atan2(sin_mean, cos_mean)          # (B,1) circular mean direction
    dphi = wrap_to_pi(phi - phi0)                   # (B,P) now seam-safe
    return dphi

class ParticleTransformer(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],
                 activation='gelu',
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 gmp_coords = "raw",
                 use_gmp = False,
                 use_gpca = False,
                 gpca_layers = 1,
                 use_sada = False,
                 sada_num_samples = 8,
                 sada_offset_scale = 2.0,
                 gmp_kernel = 3,
                 gmp_grid = 0.2,
                 gmp_reduce = "sum",
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.use_amp = use_amp

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim

        self.gmp_coords = gmp_coords
        self.use_gmp = use_gmp
        self.use_gpca = use_gpca
        self.use_sada = use_sada
        self.gmp = None
        self.gpca = None
        self.sada = None

        if (self.use_gpca or self.use_sada) and not self.use_gmp:
            raise ValueError("GPCA and SADA currently require use_gmp=True so they can reuse the GMP grid context.")

        if self.use_gmp:
            self.gmp = GeometricMessagePassing(
                channels=embed_dim,
                kernel_size=gmp_kernel,
                grid_size=gmp_grid,
                scatter_reduce=gmp_reduce,
            )

        _logger.info(
            f"GMP ENABLED: use_gmp={use_gmp}, use_gpca={use_gpca}, use_sada={use_sada}, "
            f"grid={gmp_grid}, coords={gmp_coords}, kernel={gmp_kernel}"
        )

        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True)

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.info('cfg_block: %s' % str(cfg_block))

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)
        _logger.info('cfg_cls_block: %s' % str(cfg_cls_block))

        if self.use_gpca:
            self.gpca = GridParticleCrossAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=gpca_layers,
                dropout=cfg_block['dropout'],
                attn_dropout=cfg_block['attn_dropout'],
                activation_dropout=cfg_block['activation_dropout'],
                activation=activation,
            )
        if self.use_sada:
            self.sada = SubstructureAwareDeformableAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_samples=sada_num_samples,
                offset_scale=sada_offset_scale,
                dropout=cfg_block['dropout'],
                activation_dropout=cfg_block['activation_dropout'],
                activation=activation,
            )

        self.pair_extra_dim = pair_extra_dim
        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList([Block(**cfg_cls_block) for _ in range(num_cls_layers)])
        self.norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None, points=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None
        # points: (N, C_points, P), generic point coordinates

        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu, points = self.trimmer(x, v, mask, uu, points)
            padding_mask = ~mask.squeeze(1)  # (N, P)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # input embedding
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)

            if self.gmp is not None or self.gpca is not None or self.sada is not None:
                x_bpc = x.permute(1, 0, 2).contiguous()  # (N,P,C)
                pad = ~mask.squeeze(1)                   # (N,P) True where padded
                if self.gmp_coords in ("raw", "pt"):
                    if v is None:
                        raise ValueError(f"gmp_coords={self.gmp_coords} requires `v` with shape (N,4,P).")
                    eta, phi = compute_eta_phi_from_p4(v)    # (N,P), (N,P)
                    phi_centered = unwrap_phi_per_jet(phi, pad=pad)

                    if self.gmp_coords == "pt":
                        px, py = v[:, 0, :], v[:, 1, :]
                        pt = torch.sqrt(px * px + py * py + 1e-8)
                        c1, c2 = pt * eta, pt * phi_centered
                    else:
                        c1, c2 = eta, phi_centered
                elif self.gmp_coords in ("points", "xy"):
                    if points is None:
                        raise ValueError(f"gmp_coords={self.gmp_coords} requires `points` with shape (N,C_points,P).")
                    if points.dim() != 3 or points.size(1) < 2:
                        raise ValueError("`points` must have shape (N, C_points, P) with C_points >= 2.")
                    c1 = points[:, 0, :]
                    c2 = points[:, 1, :]
                else:
                    raise ValueError(f"Unsupported gmp_coords={self.gmp_coords}. Use one of: raw, pt, points, xy.")

                grid_context = None
                if self.gmp is not None:
                    if self.gpca is not None or self.sada is not None:
                        x_bpc, grid_context = self.gmp(
                            x_bpc,
                            c1,
                            c2,
                            pad=pad,
                            return_context=True,
                        )
                    else:
                        x_bpc = self.gmp(x_bpc, c1, c2, pad=pad)
                if self.gpca is not None:
                    x_bpc = self.gpca(x_bpc, grid_context, pad=pad)
                if self.sada is not None:
                    x_bpc = self.sada(x_bpc, grid_context, pad=pad)
                x = x_bpc.permute(1, 0, 2).contiguous()  # back to (P,N,C)
            
            if v is not None and self.pair_embed is not None:
                v = v.masked_fill(~mask.expand_as(v), 0.0)
            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))  # (N*num_heads, P, P)

            # transform
            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

            # extract class token
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            x_cls = self.norm(cls_tokens).squeeze(0)

            # fc
            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            # print('output:\n', output)
            return output


class ParticleTransformerTagger(nn.Module):

    def __init__(self,
                 pf_input_dim,
                 sv_input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],
                 activation='gelu',
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 use_gmp = False,
                 use_gpca = False,
                 gpca_layers = 1,
                 use_sada = False,
                 sada_num_samples = 8,
                 sada_offset_scale = 2.0,
                 gmp_coords = "raw",
                 gmp_kernel = 3,
                 gmp_grid = 0.05,
                 gmp_reduce = "sum",
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.use_amp = use_amp

        self.pf_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.sv_trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        self.pf_embed = Embed(pf_input_dim, embed_dims, activation=activation)
        self.sv_embed = Embed(sv_input_dim, embed_dims, activation=activation)

        self.part = ParticleTransformer(input_dim=embed_dims[-1],
                                        num_classes=num_classes,
                                        # network configurations
                                        pair_input_dim=pair_input_dim,
                                        pair_extra_dim=pair_extra_dim,
                                        remove_self_pair=remove_self_pair,
                                        use_pre_activation_pair=use_pre_activation_pair,
                                        embed_dims=[],
                                        pair_embed_dims=pair_embed_dims,
                                        num_heads=num_heads,
                                        num_layers=num_layers,
                                        num_cls_layers=num_cls_layers,
                                        block_params=block_params,
                                        cls_block_params=cls_block_params,
                                        fc_params=fc_params,
                                        activation=activation,
                                        # misc
                                        trim=False,
                                        for_inference=for_inference,
                                        use_amp=use_amp,
                                        use_gmp=use_gmp,
                                        use_gpca=use_gpca,
                                        gpca_layers=gpca_layers,
                                        use_sada=use_sada,
                                        sada_num_samples=sada_num_samples,
                                        sada_offset_scale=sada_offset_scale,
                                        gmp_coords=gmp_coords,
                                        gmp_kernel=gmp_kernel,
                                        gmp_grid=gmp_grid,
                                        gmp_reduce=gmp_reduce,)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'part.cls_token', }

    def forward(self, pf_x, pf_v=None, pf_mask=None, sv_x=None, sv_v=None, sv_mask=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0

        with torch.no_grad():
            pf_x, pf_v, pf_mask, _, _ = self.pf_trimmer(pf_x, pf_v, pf_mask)
            sv_x, sv_v, sv_mask, _, _ = self.sv_trimmer(sv_x, sv_v, sv_mask)
            v = torch.cat([pf_v, sv_v], dim=2)
            mask = torch.cat([pf_mask, sv_mask], dim=2)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            pf_x = self.pf_embed(pf_x)  # after embed: (seq_len, batch, embed_dim)
            sv_x = self.sv_embed(sv_x)
            x = torch.cat([pf_x, sv_x], dim=0)

            return self.part(x, v, mask)


class ParticleTransformerTaggerWithExtraPairFeatures(nn.Module):

    def __init__(self,
                 pf_input_dim,
                 sv_input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],
                 activation='gelu',
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 use_gmp = False,
                 use_gpca = False,
                 gpca_layers = 1,
                 use_sada = False,
                 sada_num_samples = 8,
                 sada_offset_scale = 2.0,
                 gmp_coords = "raw",
                 gmp_kernel = 3,
                 gmp_grid = 0.05,
                 gmp_reduce = "sum",
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.use_amp = use_amp
        self.for_inference = for_inference

        self.pf_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.sv_trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        self.pf_embed = Embed(pf_input_dim, embed_dims, activation=activation)
        self.sv_embed = Embed(sv_input_dim, embed_dims, activation=activation)

        self.part = ParticleTransformer(input_dim=embed_dims[-1],
                                        num_classes=num_classes,
                                        # network configurations
                                        pair_input_dim=pair_input_dim,
                                        pair_extra_dim=pair_extra_dim,
                                        remove_self_pair=remove_self_pair,
                                        use_pre_activation_pair=use_pre_activation_pair,
                                        embed_dims=[],
                                        pair_embed_dims=pair_embed_dims,
                                        num_heads=num_heads,
                                        num_layers=num_layers,
                                        num_cls_layers=num_cls_layers,
                                        block_params=block_params,
                                        cls_block_params=cls_block_params,
                                        fc_params=fc_params,
                                        activation=activation,
                                        # misc
                                        trim=False,
                                        for_inference=for_inference,
                                        use_amp=use_amp,
                                        use_gmp=use_gmp,
                                        use_gpca=use_gpca,
                                        gpca_layers=gpca_layers,
                                        use_sada=use_sada,
                                        sada_num_samples=sada_num_samples,
                                        sada_offset_scale=sada_offset_scale,
                                        gmp_coords=gmp_coords,
                                        gmp_kernel=gmp_kernel,
                                        gmp_grid=gmp_grid,
                                        gmp_reduce=gmp_reduce)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'part.cls_token', }

    def forward(self, pf_x, pf_v=None, pf_mask=None, sv_x=None, sv_v=None, sv_mask=None, pf_uu=None, pf_uu_idx=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0

        with torch.no_grad():
            if not self.for_inference:
                if pf_uu_idx is not None:
                    pf_uu = build_sparse_tensor(pf_uu, pf_uu_idx, pf_x.size(-1))

            pf_x, pf_v, pf_mask, pf_uu, _ = self.pf_trimmer(pf_x, pf_v, pf_mask, pf_uu)
            sv_x, sv_v, sv_mask, _, _ = self.sv_trimmer(sv_x, sv_v, sv_mask)
            v = torch.cat([pf_v, sv_v], dim=2)
            mask = torch.cat([pf_mask, sv_mask], dim=2)
            uu = torch.zeros(v.size(0), pf_uu.size(1), v.size(2), v.size(2), dtype=v.dtype, device=v.device)
            uu[:, :, :pf_x.size(2), :pf_x.size(2)] = pf_uu

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            pf_x = self.pf_embed(pf_x)  # after embed: (seq_len, batch, embed_dim)
            sv_x = self.sv_embed(sv_x)
            x = torch.cat([pf_x, sv_x], dim=0)

            return self.part(x, v, mask, uu)
