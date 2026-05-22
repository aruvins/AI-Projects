import torch
from encoding import positional_encoding


def _raw_to_rgb_and_weights(outputs, t_vals, device):
    """Shared helper: convert raw model output → rgb_map + weights."""
    rgb   = torch.sigmoid(outputs[..., :3])
    sigma = torch.relu(outputs[...,  3])

    delta = t_vals[..., 1:] - t_vals[..., :-1]          # (B, S-1)
    delta = torch.cat(
        [delta, torch.full((*delta.shape[:-1], 1), 1e10, device=device)],
        dim=-1
    )                                                     # (B, S)

    alpha   = 1.0 - torch.exp(-sigma * delta)            # (B, S)
    T       = torch.cumprod(
        torch.cat([torch.ones((*alpha.shape[:-1], 1), device=device),
                   1.0 - alpha + 1e-10], dim=-1),
        dim=-1
    )[..., :-1]                                           # (B, S)
    weights = alpha * T                                   # (B, S)
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2) # (B, 3)
    return rgb_map, weights


def _sample_fine(weights_coarse, t_coarse, n_fine, det=False):
    """
    Importance-sample `n_fine` new t values guided by the coarse weights.
    Returns combined & sorted t values of shape (B, n_coarse + n_fine).
    """
    # Build a PDF from coarse weights (skip first/last bin edge artefacts)
    weights = weights_coarse[..., 1:-1] + 1e-5           # (B, S-2)  avoid zeros
    pdf     = weights / weights.sum(dim=-1, keepdim=True)
    cdf     = torch.cumsum(pdf, dim=-1)
    cdf     = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (B, S-1)

    # Sample from CDF
    if det:
        u = torch.linspace(0, 1, n_fine, device=weights.device)
        u = u.expand(*cdf.shape[:-1], n_fine)
    else:
        u = torch.rand(*cdf.shape[:-1], n_fine, device=weights.device)
    u = u.contiguous()

    # Invert CDF
    inds    = torch.searchsorted(cdf, u, right=True)
    below   = (inds - 1).clamp(min=0)
    above   = inds.clamp(max=cdf.shape[-1] - 1)
    inds_g  = torch.stack([below, above], dim=-1)        # (B, n_fine, 2)

    cdf_g   = torch.gather(cdf,      -1, inds_g.view(*inds_g.shape[:-2], -1)
                           ).view(*inds_g.shape)
    bins_g  = torch.gather(t_coarse, -1, inds_g.view(*inds_g.shape[:-2], -1)
                           ).view(*inds_g.shape)

    denom   = (cdf_g[..., 1] - cdf_g[..., 0]).clamp(min=1e-5)
    t_fine  = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (
                  bins_g[..., 1] - bins_g[..., 0])

    # Merge coarse + fine and sort
    t_all, _ = torch.sort(torch.cat([t_coarse, t_fine], dim=-1), dim=-1)
    return t_all


def volume_render(model, rays_o, rays_d,
                  near=2.0, far=6.0,
                  n_coarse=64, n_fine=128):
    """
    Two-pass hierarchical volume rendering.

    Pass 1 (coarse): stratified uniform samples with per-ray jitter.
    Pass 2 (fine):   importance-sampled points guided by coarse weights,
                     merged with coarse points and re-evaluated.

    Args:
        model    : NeRF network
        rays_o   : (B, 3) ray origins
        rays_d   : (B, 3) ray directions
        near/far : depth bounds
        n_coarse : uniform samples per ray  (was hardcoded 64)
        n_fine   : extra importance samples (new)

    Returns:
        rgb_map  : (B, 3) fine-pass colour
    """
    device = rays_o.device
    B      = rays_o.shape[0]

    # ── Coarse pass ──────────────────────────────────────────────────────────
    # Stratified sampling: divide [near, far] into n_coarse bins, pick one
    # random point inside each bin → eliminates aliasing from uniform grids.
    t_edges  = torch.linspace(near, far, n_coarse + 1, device=device)   # (S+1,)
    t_coarse = (t_edges[:-1] + t_edges[1:]) / 2.0                       # (S,) bin centres
    # jitter each ray independently
    jitter   = torch.rand(B, n_coarse, device=device) * (far - near) / n_coarse
    t_coarse = t_coarse.unsqueeze(0) + jitter                            # (B, S)
    t_coarse = t_coarse.clamp(near, far)

    pts_c    = rays_o[:, None, :] + rays_d[:, None, :] * t_coarse[..., None]  # (B,S,3)
    enc_c    = positional_encoding(pts_c.reshape(-1, 3))
    out_c    = model(enc_c).reshape(B, n_coarse, 4)
    _, weights_c = _raw_to_rgb_and_weights(out_c, t_coarse, device)

    # ── Fine pass ─────────────────────────────────────────────────────────────
    t_fine = _sample_fine(weights_c, t_coarse, n_fine)                  # (B, S+n_fine)
    n_all  = t_fine.shape[-1]

    pts_f  = rays_o[:, None, :] + rays_d[:, None, :] * t_fine[..., None]  # (B, n_all, 3)
    enc_f  = positional_encoding(pts_f.reshape(-1, 3))
    out_f  = model(enc_f).reshape(B, n_all, 4)
    rgb_map, _ = _raw_to_rgb_and_weights(out_f, t_fine, device)

    return rgb_map