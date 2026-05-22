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

def create_orbit_poses(radius=2.0, elevation=0.3, n_frames=60):
    poses = []

    for t in range(n_frames):
        angle = 2 * np.pi * t / n_frames

        # Calculate camera position on a sphere (adding slight elevation often looks better)
        cam_x = radius * np.cos(angle)
        cam_z = radius * np.sin(angle)
        cam_y = elevation  # Keeps the camera slightly raised looking down

        camera_position = np.array([cam_x, cam_y, cam_z])
        target_position = np.array([0.0, 0.0, 0.0]) # Looking at origin
        up_vector = np.array([0.0, 1.0, 0.0]) # Standard OpenGL up-vector

        # Compute Look-At Rotation Matrix
        # Note: Depending on your generate_rays convention (OpenGL vs. COLMAP), 
        # you might need to flip signs for forward/right vectors.
        forward = target_position - camera_position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(up_vector, forward)
        right = right / np.linalg.norm(right)

        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        # Construct the 4x4 camera-to-world (c2w) matrix
        c2w = np.eye(4)
        c2w[0:3, 0] = right
        c2w[0:3, 1] = up
        c2w[0:3, 2] = -forward  # NeRF/OpenGL convention: camera looks down -Z
        c2w[0:3, 3] = camera_position

        poses.append(torch.tensor(c2w, dtype=torch.float32))

    return poses

def render_video(model, H, W, focal, device, output="outputs/nerf/nerf.mp4"):
    poses = create_orbit_poses()

    frames = []

    for pose in tqdm(poses, desc="Rendering video"):
        frame = render_image(model, H, W, focal, pose.to(device), device)
        frames.append((frame * 255).astype(np.uint8))

    imageio.mimsave(output, frames, fps=30)
    print(f"Saved video → {output}")