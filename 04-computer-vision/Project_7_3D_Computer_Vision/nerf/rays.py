import torch


def generate_rays(H, W, focal, c2w, device):
    i, j = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing='xy'
    )


    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i)
    ], dim=-1)

    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)

    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d