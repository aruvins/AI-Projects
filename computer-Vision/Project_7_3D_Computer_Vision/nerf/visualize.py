import torch
import numpy as np
import imageio
from tqdm import tqdm

from model import NeRF
from rays import generate_rays
from render import volume_render


def render_image(model, H, W, focal, c2w, device):
    rays_o, rays_d = generate_rays(H, W, focal, c2w, device)

    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    rgb = []

    chunk = 1024
    for i in range(0, rays_o.shape[0], chunk):
        ro = rays_o[i:i+chunk]
        rd = rays_d[i:i+chunk]
        rgb_chunk = volume_render(model, ro, rd)
        rgb.append(rgb_chunk)

    rgb = torch.cat(rgb, dim=0)
    rgb = rgb.reshape(H, W, 3)

    return rgb.detach().cpu().numpy()

def create_orbit_poses(radius=2.0, n_frames=60):
    poses = []

    for t in range(n_frames):
        angle = 2 * np.pi * t / n_frames

        c2w = np.eye(4)
        c2w[0, 3] = radius * np.cos(angle)
        c2w[2, 3] = radius * np.sin(angle)

        poses.append(torch.tensor(c2w, dtype=torch.float32))

    return poses

def render_video(model, H, W, focal, device, output="nerf.mp4"):
    poses = create_orbit_poses()

    frames = []

    for pose in tqdm(poses, desc="Rendering video"):
        frame = render_image(model, H, W, focal, pose.to(device), device)
        frames.append((frame * 255).astype(np.uint8))

    imageio.mimsave(output, frames, fps=30)
    print(f"Saved video → {output}")