import torch
import numpy as np
import imageio
import os
import glob

from model import NeRF
from render import volume_render
from rays import generate_rays
from visualize import render_image, render_video

# ======================
# Device selection
# ======================
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# ======================
# Paths
# ======================
CHECKPOINT_DIR = "nerf/checkpoints"
TOTAL_STEPS    = 300_000
CKPT_EVERY     = 50_000   # save a checkpoint every N steps

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("outputs/nerf", exist_ok=True)

# ======================
# Load TinyNeRF dataset
# ======================
data = np.load("./data/nerf/tiny_nerf_data.npz")

images = data["images"]   # (N, H, W, 3)
poses  = data["poses"]    # (N, 4, 4)
focal  = float(data["focal"])

H, W = images.shape[1], images.shape[2]

images = torch.from_numpy(images).to(device)
poses  = torch.from_numpy(poses).to(device)

# ======================
# Model & optimiser
# ======================
model     = NeRF().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# ======================
# Resume from latest checkpoint
# ======================
start_step = 0

ckpt_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt_step_*.pth")))
if ckpt_files:
    latest = ckpt_files[-1]
    print(f"Resuming from checkpoint: {latest}")
    ckpt       = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"] + 1
    print(f"  → starting at step {start_step}")
else:
    print("No checkpoint found — starting from scratch.")

# ======================
# Training loop
# ======================
for step in range(start_step, TOTAL_STEPS):

    idx = torch.randint(0, images.shape[0], (1,)).item()

    target_img = images[idx]
    c2w        = poses[idx]

    rays_o, rays_d = generate_rays(H, W, focal, c2w, device)

    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    target = target_img.reshape(-1, 3)

    # sample batch
    batch_idx = torch.randint(0, rays_o.shape[0], (1024,))

    pred = volume_render(model, rays_o[batch_idx], rays_d[batch_idx])
    gt   = target[batch_idx]

    loss = ((pred - gt) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step:>6d} | Loss: {loss.item():.4f}")

    if step % CKPT_EVERY == 0 and step > 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_step_{step:07d}.pth")
        torch.save({"step": step, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, ckpt_path)
        print(f"  ✓ Checkpoint saved → {ckpt_path}")

# ======================
# Final outputs
# ======================
print(model.eval())

img = render_image(model, H, W, focal, poses[83].to(device), device)
imageio.imwrite("outputs/nerf/render.png", (img * 255).astype(np.uint8))

render_video(model, H, W, focal, device)
