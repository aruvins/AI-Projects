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

def create_orbit_poses(radius=3.5, elevation=0.3, roll_angle_deg=0.0, n_frames=60):
    poses = []
    roll_rad = np.radians(roll_angle_deg)

    for t in range(n_frames):
        angle = 2 * np.pi * t / n_frames

        # CHANGED: Swap axes so the orbit circles horizontally (left to right)
        cam_x = radius * np.cos(angle)
        cam_y = radius * np.sin(angle)
        cam_z = elevation  # Constant height, looking slightly down if positive

        camera_position = np.array([cam_x, cam_y, cam_z])
        target_position = np.array([0.0, 0.0, 0.0]) # Looking at origin

        # Compute standard forward vector pointing at the origin
        forward = target_position - camera_position
        forward = forward / np.linalg.norm(forward)

        # Base world up-vector (If Z is up, your base up vector should be [0, 0, 1])
        base_up = np.array([0.0, 0.0, 1.0])

        # Find a temporary right vector to help us compute a rolled up-vector
        temp_right = np.cross(base_up, forward)
        temp_right = temp_right / np.linalg.norm(temp_right)
        
        # Re-compute true world up for this position
        true_up = np.cross(forward, temp_right)

        # Tilt the up_vector on its side by blending the true_up and temp_right
        up_vector = true_up * np.cos(roll_rad) + temp_right * np.sin(roll_rad)

        # Recompute final camera coordinates with the new tilted up_vector
        right = np.cross(up_vector, forward)
        right = right / np.linalg.norm(right)

        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        # Construct the 4x4 camera-to-world (c2w) matrix
        c2w = np.eye(4)
        c2w[0:3, 0] = right
        c2w[0:3, 1] = up
        c2w[0:3, 2] = -forward  
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