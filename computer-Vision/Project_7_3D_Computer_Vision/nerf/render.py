import torch
from encoding import positional_encoding


def volume_render(model, rays_o, rays_d, near=2.0, far=6.0, samples=64):
    device = rays_o.device

    t_vals = torch.linspace(near, far, samples).to(device)

    pts = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., :, None]

    encoded = positional_encoding(pts.reshape(-1, 3))

    outputs = model(encoded)

    outputs = outputs.reshape(*pts.shape[:-1], 4)

    rgb = torch.sigmoid(outputs[..., :3])
    sigma = torch.relu(outputs[..., 3])

    delta = t_vals[1:] - t_vals[:-1]
    delta = torch.cat([delta, torch.tensor([1e10]).to(device)])

    alpha = 1.0 - torch.exp(-sigma * delta)

    T = torch.cumprod(
        torch.cat([
            torch.ones((alpha.shape[0], 1)).to(device),
            1.0 - alpha + 1e-10
        ], dim=-1),
        dim=-1
    )[:, :-1]

    weights = alpha * T

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

    return rgb_map