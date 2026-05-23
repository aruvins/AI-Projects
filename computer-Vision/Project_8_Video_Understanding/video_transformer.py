"""
VideoTransformer — Built from Scratch
======================================
A temporal Vision Transformer for video action recognition.

Architecture overview
─────────────────────
1.  Tubelet Embedding     – divides the video into (t, h, w) non-overlapping
                            "tubes" and linearly projects each into a token.
2.  Spatial-Temporal PE   – learnable positional embeddings added to tokens.
3.  [CLS] Token           – prepended; used for classification.
4.  Transformer Encoder   – N layers of multi-head self-attention + FFN.
5.  Classification Head   – linear layer on the [CLS] representation.

The Transformer block itself is also hand-rolled so every component is
visible and educational — no nn.TransformerEncoder is used.

Usage
─────
    python video_transformer.py                       # quick smoke-test
    python video_transformer.py --train               # train on UCF-101
    python video_transformer.py --train --data ./data # custom data dir
"""

import math
import json
import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ══════════════════════════════════════════════════════════════════════════════
#  1.  TUBELET EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════
class TubeletEmbedding(nn.Module):
    """
    Splits a video tensor into (t_patch × h_patch × w_patch) non-overlapping
    3-D patches ('tubelets') and projects each to `embed_dim` with a single
    Conv3d whose kernel and stride equal the patch sizes.

    Input  : (B, C, T, H, W)
    Output : (B, N, embed_dim)   where N = n_t * n_h * n_w
    """

    def __init__(
        self,
        img_size: int = 112,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 192,
        num_frames: int = 16,
    ):
        super().__init__()
        self.img_size           = img_size
        self.patch_size         = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim          = embed_dim

        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        assert num_frames % temporal_patch_size == 0, (
            "num_frames must be divisible by temporal_patch_size"
        )

        self.n_h = img_size  // patch_size
        self.n_w = img_size  // patch_size
        self.n_t = num_frames // temporal_patch_size
        self.num_tokens = self.n_t * self.n_h * self.n_w

        # One Conv3d does the patch extraction + linear projection in one step.
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.proj(x)                   # (B, D, n_t, n_h, n_w)
        B, D, nt, nh, nw = x.shape
        x = x.flatten(2)                   # (B, D, N)
        x = x.transpose(1, 2)             # (B, N, D)
        return x


# ══════════════════════════════════════════════════════════════════════════════
#  2.  MULTI-HEAD SELF-ATTENTION (hand-rolled, no nn.MultiheadAttention)
# ══════════════════════════════════════════════════════════════════════════════
class MultiHeadSelfAttention(nn.Module):
    """
    Vanilla scaled dot-product attention with multiple heads.

    For each head h:
        Q_h = X W_Q^h,  K_h = X W_K^h,  V_h = X W_V^h
        Attn_h = softmax( Q_h K_h^T / sqrt(d_k) ) V_h
    Output = concat(Attn_1 ... Attn_H) W_O

    We implement this efficiently with a single (D → 3*D) projection
    that is then split into Q, K, V.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5

        # Fused Q, K, V projection
        self.qkv   = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj  = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim

        # Project to Q, K, V  →  (B, N, 3*D)
        qkv = self.qkv(x).reshape(B, N, 3, H, d)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, d)
        q, k, v = qkv.unbind(0)            # each (B, H, N, d)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values
        out = (attn @ v)                   # (B, H, N, d)
        out = out.transpose(1, 2).reshape(B, N, D)   # (B, N, D)
        out = self.proj(out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  3.  FEED-FORWARD NETWORK (MLP sub-layer)
# ══════════════════════════════════════════════════════════════════════════════
class FeedForward(nn.Module):
    """
    Two-layer MLP with GELU activation:
        FFN(x) = GELU( x W_1 + b_1 ) W_2 + b_2
    The hidden dimension is typically 4 × embed_dim.
    """

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
#  4.  TRANSFORMER ENCODER BLOCK
# ══════════════════════════════════════════════════════════════════════════════
class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer block (more stable training than post-LN):

        x = x + MHSA( LayerNorm(x) )
        x = x + FFN(  LayerNorm(x) )
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = FeedForward(embed_dim, mlp_ratio, dropout=dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ══════════════════════════════════════════════════════════════════════════════
#  5.  POSITIONAL EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════
class SpatioTemporalPE(nn.Module):
    """
    Learnable positional embedding: one vector per token (including [CLS]).
    Shape: (1, 1 + N, embed_dim)  — broadcast over batch.

    Initialised with a truncated normal (~std 0.02) to break symmetry gently.
    """

    def __init__(self, num_tokens: int, embed_dim: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


# ══════════════════════════════════════════════════════════════════════════════
#  6.  FULL VIDEO TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════
class VideoTransformer(nn.Module):
    """
    End-to-end Video Transformer for action recognition.

    Pipeline:
        (B, C, T, H, W)
            ↓  TubeletEmbedding
        (B, N, D)
            ↓  prepend [CLS], add SpatioTemporalPE
        (B, 1+N, D)
            ↓  N × TransformerBlock
        (B, 1+N, D)
            ↓  extract [CLS] token, LayerNorm
        (B, D)
            ↓  Linear head
        (B, num_classes)

    Hyperparameters (defaults = "Tiny" config, trainable on CPU/small GPU):
        img_size=112, patch_size=16, temporal_patch_size=2, num_frames=16
        embed_dim=192, depth=4, num_heads=3
    """

    def __init__(
        self,
        num_classes: int   = 101,
        img_size: int      = 112,
        patch_size: int    = 16,
        temporal_patch_size: int = 2,
        num_frames: int    = 16,
        in_channels: int   = 3,
        embed_dim: int     = 192,
        depth: int         = 4,
        num_heads: int     = 3,
        mlp_ratio: float   = 4.0,
        dropout: float     = 0.1,
        attn_dropout: float = 0.0,
    ):
        super().__init__()

        # ── Patch / tubelet embedding ──────────────────────────────────────
        self.tubelet_embed = TubeletEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_frames=num_frames,
        )
        N = self.tubelet_embed.num_tokens

        # ── [CLS] token ────────────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Positional embedding (1 + N tokens) ───────────────────────────
        self.pos_embed = SpatioTemporalPE(1 + N, embed_dim)

        self.pos_drop  = nn.Dropout(dropout)

        # ── Transformer encoder ────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(depth)
        ])

        # ── Classification head ────────────────────────────────────────────
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Weight init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T, H, W)   float32, pixel values in [0, 1] or normalised
        returns : (B, num_classes) logits
        """
        B = x.size(0)

        # 1. Tubelet embedding
        tokens = self.tubelet_embed(x)                     # (B, N, D)

        # 2. Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)             # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)           # (B, 1+N, D)

        # 3. Add positional embedding + dropout
        tokens = self.pos_embed(tokens)
        tokens = self.pos_drop(tokens)

        # 4. Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # 5. Extract [CLS] and normalise
        cls_out = self.norm(tokens[:, 0])                  # (B, D)

        # 6. Classify
        logits = self.head(cls_out)                        # (B, num_classes)
        return logits

    # ── Convenience methods ──────────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_token_shape(self):
        """Returns (n_t, n_h, n_w) — useful for visualising attention maps."""
        e = self.tubelet_embed
        return e.n_t, e.n_h, e.n_w


# ══════════════════════════════════════════════════════════════════════════════
#  7.  DATASET
# ══════════════════════════════════════════════════════════════════════════════
class VideoFrameDataset(Dataset):
    """
    Reads pre-extracted frame folders produced by download_data.py.

    Each entry in the JSON manifest:
        {
            "video_id":   "v_Archery_g01_c01",
            "class_name": "Archery",
            "class_idx":  0,
            "frames_dir": "/data/frames/Archery/v_Archery_g01_c01",
            "num_frames": 16
        }

    The dataset:
        1. Lists frame files in `frames_dir`  (sorted by name).
        2. Uniformly samples `num_frames` of them.
        3. Loads each as a float32 tensor in [0, 1].
        4. Returns (video_tensor, label) where video_tensor is (C, T, H, W).
    """

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def __init__(self, manifest_path: str, num_frames: int = 16, augment: bool = False):
        with open(manifest_path) as f:
            self.records = json.load(f)
        self.num_frames = num_frames
        self.augment    = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        frames_dir = Path(rec["frames_dir"])

        # Collect frame files (support .jpg and .ppm from synthetic data)
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        if not frame_files:
            frame_files = sorted(frames_dir.glob("frame_*.ppm"))
        if not frame_files:
            # Return zeros + label (graceful degradation)
            return torch.zeros(3, self.num_frames, 112, 112), rec["class_idx"]

        # Uniform temporal sampling
        total = len(frame_files)
        indices = [int(i * total / self.num_frames) for i in range(self.num_frames)]
        selected = [frame_files[min(i, total - 1)] for i in indices]

        frames = []
        for fp in selected:
            frame = self._load_frame(fp)
            frames.append(frame)

        video = torch.stack(frames, dim=1)   # (C, T, H, W)

        # ImageNet normalisation
        video = (video - self.MEAN) / self.STD

        if self.augment:
            video = self._augment(video)

        return video, rec["class_idx"]

    @staticmethod
    def _load_frame(path: Path) -> torch.Tensor:
        """Loads a single frame image as a (3, H, W) float tensor in [0,1]."""
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            arr = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
            arr = arr.view(img.height, img.width, 3).float() / 255.0
            return arr.permute(2, 0, 1)   # (3, H, W)
        except ImportError:
            pass
        try:
            import cv2, numpy as np
            img = cv2.imread(str(path))
            if img is None:
                return torch.zeros(3, 112, 112)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(img).float() / 255.0
            return t.permute(2, 0, 1)
        except ImportError:
            pass
        # PPM fallback (synthetic data)
        return VideoFrameDataset._load_ppm(path)

    @staticmethod
    def _load_ppm(path: Path) -> torch.Tensor:
        import numpy as np
        with open(path, "rb") as f:
            magic = f.readline().strip()
            assert magic == b"P6"
            dims  = f.readline().split()
            w, h  = int(dims[0]), int(dims[1])
            f.readline()  # max val
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w, 3)
        t = torch.from_numpy(data.copy()).float() / 255.0
        return t.permute(2, 0, 1)

    @staticmethod
    def _augment(video: torch.Tensor) -> torch.Tensor:
        """
        Simple spatio-temporal augmentation:
          • Random horizontal flip (same flip applied to all frames).
          • Colour jitter (brightness / contrast — applied uniformly).
        """
        if random.random() > 0.5:
            video = video.flip(-1)  # flip width axis
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            video = (video * factor).clamp(0, 1)
        return video


# ══════════════════════════════════════════════════════════════════════════════
#  8.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        B    = targets.size(0)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0).item() * 100 / B for k in topk]


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    loss_m, top1_m = AverageMeter(), AverageMeter()

    for step, (videos, labels) in enumerate(loader):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(videos)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        top1 = accuracy(logits, labels, topk=(1,))[0]
        loss_m.update(loss.item(), videos.size(0))
        top1_m.update(top1,        videos.size(0))

        if step % 20 == 0:
            print(
                f"  Epoch {epoch:02d} | step {step:04d}/{len(loader):04d} "
                f"| loss {loss_m.avg:.4f} | top-1 {top1_m.avg:.2f}%"
            )

    return loss_m.avg, top1_m.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_m, top1_m = AverageMeter(), AverageMeter()

    for videos, labels in loader:
        videos = videos.to(device)
        labels = labels.to(device)
        logits = model(videos)
        loss   = criterion(logits, labels)
        top1   = accuracy(logits, labels, topk=(1,))[0]
        loss_m.update(loss.item(), videos.size(0))
        top1_m.update(top1,        videos.size(0))

    return loss_m.avg, top1_m.avg


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    data_dir = Path(args.data)
    train_manifest = data_dir / "train_manifest.json"
    test_manifest  = data_dir / "test_manifest.json"
    class_map_file = data_dir / "class_to_idx.json"

    if not train_manifest.exists():
        raise FileNotFoundError(
            f"Train manifest not found at {train_manifest}.\n"
            "Run `python download_data.py --synthetic` first."
        )

    with open(class_map_file) as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)
    print(f"Classes: {num_classes}")

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = VideoFrameDataset(train_manifest, num_frames=args.num_frames, augment=True)
    test_ds  = VideoFrameDataset(test_manifest,  num_frames=args.num_frames, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = VideoTransformer(
        num_classes=num_classes,
        img_size=args.img_size,
        patch_size=args.patch_size,
        temporal_patch_size=args.temporal_patch_size,
        num_frames=args.num_frames,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    print(f"Parameters: {model.count_parameters():,}")
    print(f"Token grid : {model.get_token_shape()} (t × h × w)")

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Training loop ─────────────────────────────────────────────────────
    best_acc = 0.0
    ckpt_path = Path(args.checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\n── Epoch {epoch}/{args.epochs}  lr={lr:.6f} ──")

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(
            f"  Train: loss={tr_loss:.4f} top-1={tr_acc:.2f}%  |  "
            f"Val: loss={val_loss:.4f} top-1={val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_acc": val_acc},
                ckpt_path / "best.pt",
            )
            print(f"  ✓ New best: {best_acc:.2f}% — checkpoint saved")

    print(f"\n✓ Training complete. Best val top-1: {best_acc:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  9.  SMOKE TEST (no data needed)
# ══════════════════════════════════════════════════════════════════════════════
def smoke_test():
    """
    Verifies shapes are correct end-to-end with random tensors.
    No dataset required.
    """
    print("\n── Smoke Test ──────────────────────────────────────────────────")
    device = torch.device("cpu")

    # Config: tiny model for fast CPU test
    cfg = dict(
        num_classes=101, img_size=112, patch_size=16,
        temporal_patch_size=2, num_frames=16,
        embed_dim=192, depth=4, num_heads=3,
    )
    model = VideoTransformer(**cfg).to(device)
    print(f"  Parameters : {model.count_parameters():,}")
    print(f"  Token grid : {model.get_token_shape()} (t × h × w)")
    print(f"  Tokens N   : {model.tubelet_embed.num_tokens}")

    B = 2
    x = torch.randn(B, 3, 16, 112, 112)   # (B, C, T, H, W)
    logits = model(x)
    print(f"  Input  : {tuple(x.shape)}")
    print(f"  Output : {tuple(logits.shape)}   (should be [{B}, 101])")
    assert logits.shape == (B, 101), "Shape mismatch!"

    # Also test gradient flow
    loss = logits.sum()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"  Grad norms (sample): {[f'{g:.4f}' for g in grad_norms[:5]]}")
    print("  ✓ Smoke test passed!\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="VideoTransformer training")
    parser.add_argument("--train",               action="store_true")
    parser.add_argument("--data",                default="./data")
    parser.add_argument("--checkpoint_dir",      default="./checkpoints")
    parser.add_argument("--epochs",              type=int,   default=30)
    parser.add_argument("--batch_size",          type=int,   default=8)
    parser.add_argument("--lr",                  type=float, default=1e-4)
    parser.add_argument("--weight_decay",        type=float, default=0.05)
    parser.add_argument("--dropout",             type=float, default=0.1)
    parser.add_argument("--num_frames",          type=int,   default=16)
    parser.add_argument("--img_size",            type=int,   default=112)
    parser.add_argument("--patch_size",          type=int,   default=16)
    parser.add_argument("--temporal_patch_size", type=int,   default=2)
    parser.add_argument("--embed_dim",           type=int,   default=192)
    parser.add_argument("--depth",               type=int,   default=4)
    parser.add_argument("--num_heads",           type=int,   default=3)
    parser.add_argument("--workers",             type=int,   default=2)
    args = parser.parse_args()

    smoke_test()

    if args.train:
        train(args)
    else:
        print("Tip: run with --train to start training on real/synthetic data.")
        print("     First run: python download_data.py --synthetic")


if __name__ == "__main__":
    main()