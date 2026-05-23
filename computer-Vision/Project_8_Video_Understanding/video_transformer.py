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
    python video_transformer.py                       # smoke-test + architecture plot
    python video_transformer.py --train               # train on UCF-101
    python video_transformer.py --train --data ./data # custom data dir
    python video_transformer.py --visualize           # attention maps on random input
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
        self.img_size            = img_size
        self.patch_size          = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim           = embed_dim

        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        assert num_frames % temporal_patch_size == 0, (
            "num_frames must be divisible by temporal_patch_size"
        )

        self.n_h = img_size   // patch_size
        self.n_w = img_size   // patch_size
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
        x = self.proj(x)          # (B, D, n_t, n_h, n_w)
        B, D, nt, nh, nw = x.shape
        x = x.flatten(2)          # (B, D, N)
        x = x.transpose(1, 2)    # (B, N, D)
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

    When return_attn=True the raw attention weights are also returned,
    which lets us build attention-map visualisations later.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5

        self.qkv       = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj      = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, d)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, H, N, d)
        q, k, v = qkv.unbind(0)             # each (B, H, N, d)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        attn_weights = attn                              # save before dropout
        attn = self.attn_drop(attn)

        out = (attn @ v)                                 # (B, H, N, d)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)

        if return_attn:
            return out, attn_weights
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

    When return_attn=True the attention weights from MHSA are passed back
    so callers can extract them for visualisation.
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

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if return_attn:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attn=True)
            x = x + self.drop(attn_out)
            x = x + self.drop(self.ffn(self.norm2(x)))
            return x, attn_weights
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
    """

    def __init__(
        self,
        num_classes: int        = 101,
        img_size: int           = 112,
        patch_size: int         = 16,
        temporal_patch_size: int = 2,
        num_frames: int         = 16,
        in_channels: int        = 3,
        embed_dim: int          = 192,
        depth: int              = 4,
        num_heads: int          = 3,
        mlp_ratio: float        = 4.0,
        dropout: float          = 0.1,
        attn_dropout: float     = 0.0,
    ):
        super().__init__()

        self.tubelet_embed = TubeletEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_frames=num_frames,
        )
        N = self.tubelet_embed.num_tokens

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_embed = SpatioTemporalPE(1 + N, embed_dim)
        self.pos_drop  = nn.Dropout(dropout)

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

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
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
        B = x.size(0)
        tokens = self.tubelet_embed(x)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_embed(tokens)
        tokens = self.pos_drop(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        cls_out = self.norm(tokens[:, 0])
        return self.head(cls_out)

    def forward_with_attn(self, x: torch.Tensor):
        """
        Same as forward() but also returns the attention weights from every
        block. Used by the visualisation functions.

        Returns:
            logits      : (B, num_classes)
            attn_list   : list of (B, H, N+1, N+1) tensors, one per block
        """
        B = x.size(0)
        tokens = self.tubelet_embed(x)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_embed(tokens)
        tokens = self.pos_drop(tokens)

        attn_list = []
        for block in self.blocks:
            tokens, attn_w = block(tokens, return_attn=True)
            attn_list.append(attn_w)   # (B, H, N+1, N+1)

        cls_out = self.norm(tokens[:, 0])
        logits  = self.head(cls_out)
        return logits, attn_list

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_token_shape(self):
        e = self.tubelet_embed
        return e.n_t, e.n_h, e.n_w


# ══════════════════════════════════════════════════════════════════════════════
#  7.  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class VideoFrameDataset(Dataset):
    """
    Reads pre-extracted frame folders produced by download_data.py.
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
        rec        = self.records[idx]
        frames_dir = Path(rec["frames_dir"])

        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        if not frame_files:
            frame_files = sorted(frames_dir.glob("frame_*.ppm"))
        if not frame_files:
            return torch.zeros(3, self.num_frames, 112, 112), rec["class_idx"]

        total   = len(frame_files)
        indices = [int(i * total / self.num_frames) for i in range(self.num_frames)]
        selected = [frame_files[min(i, total - 1)] for i in indices]

        frames = [self._load_frame(fp) for fp in selected]
        video  = torch.stack(frames, dim=1)          # (C, T, H, W)
        video  = (video - self.MEAN) / self.STD

        if self.augment:
            video = self._augment(video)

        return video, rec["class_idx"]

    @staticmethod
    def _load_frame(path: Path) -> torch.Tensor:
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            arr = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
            arr = arr.view(img.height, img.width, 3).float() / 255.0
            return arr.permute(2, 0, 1)
        except ImportError:
            pass
        try:
            import cv2
            import numpy as np
            img = cv2.imread(str(path))
            if img is None:
                return torch.zeros(3, 112, 112)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t   = torch.from_numpy(img).float() / 255.0
            return t.permute(2, 0, 1)
        except ImportError:
            pass
        return VideoFrameDataset._load_ppm(path)

    @staticmethod
    def _load_ppm(path: Path) -> torch.Tensor:
        import numpy as np
        with open(path, "rb") as f:
            assert f.readline().strip() == b"P6"
            dims = f.readline().split()
            w, h = int(dims[0]), int(dims[1])
            f.readline()
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w, 3)
        t = torch.from_numpy(data.copy()).float() / 255.0
        return t.permute(2, 0, 1)

    @staticmethod
    def _augment(video: torch.Tensor) -> torch.Tensor:
        if random.random() > 0.5:
            video = video.flip(-1)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            video  = (video * factor).clamp(0, 1)
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
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        B    = targets.size(0)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred    = pred.t()
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

    data_dir       = Path(args.data)
    train_manifest = data_dir / "train_manifest.json"
    test_manifest  = data_dir / "test_manifest.json"
    class_map_file = data_dir / "class_to_idx.json"

    if not train_manifest.exists():
        raise FileNotFoundError(
            f"Train manifest not found at {train_manifest}.\n"
            "Run `python download_data.py --output_dir ./data` first."
        )

    with open(class_map_file) as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)
    print(f"Classes: {num_classes}")

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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc  = 0.0
    ckpt_path = Path(args.checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # ── History for training-curve plot ──────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\n── Epoch {epoch}/{args.epochs}  lr={lr:.6f} ──")

        tr_loss, tr_acc   = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

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

    # Save training curves automatically after training finishes
    out_dir = Path(args.vis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, out_dir / "training_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
#  9.  VISUALISATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — works everywhere
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
        return plt, gridspec, np
    except ImportError:
        raise ImportError(
            "matplotlib and numpy are required for visualisations.\n"
            "Install with:  pip install matplotlib numpy"
        )


# ── 9a. Architecture diagram ─────────────────────────────────────────────────
def plot_architecture(model: VideoTransformer, save_path: Path):
    """
    Draws a vertical block diagram of the model's layers with parameter
    counts annotated on each block.
    """
    plt, gridspec, np = _require_matplotlib()

    n_t, n_h, n_w = model.get_token_shape()
    N = model.tubelet_embed.num_tokens
    depth = len(model.blocks)

    layers = [
        ("Input video\n(B, 3, T, H, W)", "#4A90D9", 0),
        (f"Tubelet Embedding\nConv3D → {N} tokens", "#5BA85A", 0),
        ("[CLS] prepend\n+ Positional Embedding", "#9B59B6", 0),
    ]
    for i in range(depth):
        blk = model.blocks[i]
        p   = sum(p.numel() for p in blk.parameters())
        layers.append((
            f"Transformer Block {i+1}\nMHSA + FFN  ({p/1e3:.0f}k params)",
            "#E67E22",
            p,
        ))
    layers += [
        ("LayerNorm\n+ Extract [CLS]", "#9B59B6", 0),
        (f"Linear Head\n→ {model.head.out_features} classes", "#E74C3C", 0),
    ]

    fig, ax = plt.subplots(figsize=(5, len(layers) * 0.9 + 1))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, len(layers) - 0.3)
    ax.axis("off")
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")

    bh, bw, bx = 0.72, 7.0, 1.5
    arrow_props = dict(arrowstyle="-|>", color="#AAAAAA", lw=1.2)

    for i, (label, color, _) in enumerate(layers):
        y = len(layers) - 1 - i
        # Draw box
        rect = plt.Rectangle((bx, y - bh / 2), bw, bh,
                              facecolor=color, edgecolor="white",
                              linewidth=0.8, alpha=0.88, zorder=3)
        ax.add_patch(rect)
        ax.text(bx + bw / 2, y, label,
                ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold", zorder=4,
                multialignment="center")
        # Arrow to next box
        if i < len(layers) - 1:
            y_next = len(layers) - 2 - i
            ax.annotate("", xy=(bx + bw / 2, y_next + bh / 2 + 0.02),
                        xytext=(bx + bw / 2, y - bh / 2 - 0.02),
                        arrowprops=arrow_props, zorder=2)

    total = model.count_parameters()
    fig.suptitle(
        f"VideoTransformer Architecture\n{total/1e6:.2f}M parameters  |  "
        f"token grid {n_t}×{n_h}×{n_w}",
        color="white", fontsize=10, fontweight="bold", y=0.98,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved architecture diagram → {save_path}")


# ── 9b. Attention maps ───────────────────────────────────────────────────────
def plot_attention_maps(model: VideoTransformer, x: torch.Tensor, save_path: Path):
    """
    For a single video clip `x` (1, C, T, H, W), runs a forward pass
    and visualises the [CLS]→patch attention for every block and head.

    Layout:  rows = blocks,  columns = attention heads.
    Each cell shows the attention weights from [CLS] to all spatial tokens,
    averaged over temporal positions and reshaped back onto the patch grid.
    """
    plt, gridspec, np = _require_matplotlib()

    model.eval()
    with torch.no_grad():
        _, attn_list = model.forward_with_attn(x)

    n_t, n_h, n_w  = model.get_token_shape()
    depth          = len(attn_list)
    num_heads      = attn_list[0].shape[1]

    fig, axes = plt.subplots(
        depth, num_heads,
        figsize=(num_heads * 2.2, depth * 2.2),
        squeeze=False,
    )
    fig.patch.set_facecolor("#1A1A2E")
    fig.suptitle(
        "Attention Maps — [CLS] token attending to video patches\n"
        "(each cell = one head in one Transformer block)",
        color="white", fontsize=10, fontweight="bold",
    )

    for b_idx, attn in enumerate(attn_list):
        # attn: (1, H, N+1, N+1)  — strip batch dim
        attn = attn[0]   # (H, N+1, N+1)

        for h_idx in range(num_heads):
            ax = axes[b_idx][h_idx]
            ax.set_facecolor("#0D0D1A")

            # Row 0 of attention matrix = [CLS] attending to everything
            cls_attn = attn[h_idx, 0, 1:]   # (N,) — drop CLS→CLS weight

            # Reshape: (n_t, n_h, n_w) then average over temporal axis
            spatial = cls_attn.reshape(n_t, n_h, n_w).mean(dim=0)   # (n_h, n_w)
            spatial = spatial.numpy()
            spatial = (spatial - spatial.min()) / (spatial.max() - spatial.min() + 1e-8)

            im = ax.imshow(spatial, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            if h_idx == 0:
                ax.set_ylabel(f"Block {b_idx+1}", color="white", fontsize=8)
            if b_idx == 0:
                ax.set_title(f"Head {h_idx+1}", color="white", fontsize=8)

            # Thin white border
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")
                spine.set_linewidth(0.5)

    # Shared colourbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.outline.set_edgecolor("white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)
    cbar.set_label("Attention weight", color="white", fontsize=8)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved attention maps → {save_path}")


# ── 9c. Token grid ───────────────────────────────────────────────────────────
def plot_token_grid(model: VideoTransformer, x: torch.Tensor, save_path: Path):
    """
    Shows the first frame of the input video overlaid with the patch grid,
    so it's easy to see what spatial region each token covers.
    """
    plt, gridspec, np = _require_matplotlib()

    n_t, n_h, n_w = model.get_token_shape()
    patch_size     = model.tubelet_embed.patch_size
    img_size       = model.tubelet_embed.img_size

    # Denormalise the first frame for display
    MEAN = np.array([0.485, 0.456, 0.406])
    STD  = np.array([0.229, 0.224, 0.225])
    frame = x[0, :, 0].permute(1, 2, 0).numpy()   # (H, W, 3)
    frame = (frame * STD + MEAN).clip(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("#1A1A2E")
    fig.suptitle(
        f"Tubelet Token Grid  —  patch size {patch_size}×{patch_size}px  "
        f"→  {n_h}×{n_w} = {n_h*n_w} spatial tokens per temporal step  "
        f"({n_t} temporal steps  →  {n_t*n_h*n_w} total)",
        color="white", fontsize=9, fontweight="bold",
    )

    for ax in axes:
        ax.set_facecolor("#0D0D1A")

    # Left: raw frame
    axes[0].imshow(frame)
    axes[0].set_title("Input frame (first of 16)", color="white", fontsize=9)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Right: frame + patch grid overlay
    axes[1].imshow(frame)
    for row in range(n_h + 1):
        axes[1].axhline(row * patch_size - 0.5, color="#00FFAA", lw=0.6, alpha=0.8)
    for col in range(n_w + 1):
        axes[1].axvline(col * patch_size - 0.5, color="#00FFAA", lw=0.6, alpha=0.8)

    # Number each patch
    for r in range(n_h):
        for c in range(n_w):
            axes[1].text(
                c * patch_size + patch_size / 2,
                r * patch_size + patch_size / 2,
                str(r * n_w + c),
                ha="center", va="center",
                color="white", fontsize=5.5,
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.4, pad=0.5, edgecolor="none"),
            )

    axes[1].set_title("Patch grid overlay (token indices)", color="white", fontsize=9)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved token grid → {save_path}")


# ── 9d. Prediction confidence bars ──────────────────────────────────────────
def plot_prediction(
    model: VideoTransformer,
    x: torch.Tensor,
    class_names: list,
    save_path: Path,
    true_label: str = None,
    topk: int = 10,
):
    """
    Runs inference on `x` and shows a horizontal bar chart of the top-k
    predicted classes with their softmax confidence scores.
    """
    plt, gridspec, np = _require_matplotlib()

    model.eval()
    with torch.no_grad():
        logits = model(x)                          # (1, num_classes)
        probs  = F.softmax(logits[0], dim=0)       # (num_classes,)

    topk_vals, topk_idxs = probs.topk(min(topk, len(class_names)))
    topk_vals  = topk_vals.numpy()
    topk_names = [class_names[i] for i in topk_idxs.tolist()]

    fig, ax = plt.subplots(figsize=(7, max(3, topk * 0.42)))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#0D0D1A")

    colors = ["#E74C3C" if (true_label and n == true_label) else "#4A90D9"
              for n in topk_names]

    bars = ax.barh(range(len(topk_names)), topk_vals, color=colors,
                   edgecolor="#333355", linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, topk_vals):
        ax.text(
            min(val + 0.005, 0.97), bar.get_y() + bar.get_height() / 2,
            f"{val*100:.1f}%",
            va="center", ha="left", color="white", fontsize=8,
        )

    ax.set_yticks(range(len(topk_names)))
    ax.set_yticklabels(topk_names, color="white", fontsize=8)
    ax.set_xlabel("Softmax confidence", color="white", fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.tick_params(colors="white")
    ax.xaxis.set_tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    pred_name = topk_names[0]
    title = f"Top-{topk} Predictions  —  predicted: {pred_name}"
    if true_label:
        title += f"  |  ground truth: {true_label}"
        title += "  ✓" if pred_name == true_label else "  ✗"
    ax.set_title(title, color="white", fontsize=9, fontweight="bold")

    if true_label:
        from matplotlib.patches import Patch
        legend = [Patch(color="#E74C3C", label="Ground truth class"),
                  Patch(color="#4A90D9", label="Other predictions")]
        ax.legend(handles=legend, loc="lower right",
                  facecolor="#1A1A2E", edgecolor="#555577",
                  labelcolor="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved prediction chart → {save_path}")


# ── 9e. Training curves ──────────────────────────────────────────────────────
def plot_training_curves(history: dict, save_path: Path):
    """
    Plots loss and top-1 accuracy curves for train and validation over epochs.
    `history` is a dict with keys: train_loss, val_loss, train_acc, val_acc.
    """
    plt, gridspec, np = _require_matplotlib()

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#1A1A2E")

    for ax in (ax_loss, ax_acc):
        ax.set_facecolor("#0D0D1A")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        ax.grid(color="#222244", linestyle="--", linewidth=0.5)

    # Loss
    ax_loss.plot(epochs, history["train_loss"], color="#4A90D9", lw=2, label="Train")
    ax_loss.plot(epochs, history["val_loss"],   color="#E74C3C", lw=2, label="Val",
                 linestyle="--")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training & Validation Loss")
    ax_loss.legend(facecolor="#1A1A2E", edgecolor="#555577", labelcolor="white")

    # Accuracy
    ax_acc.plot(epochs, history["train_acc"], color="#4A90D9", lw=2, label="Train")
    ax_acc.plot(epochs, history["val_acc"],   color="#E74C3C", lw=2, label="Val",
                linestyle="--")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Top-1 Accuracy (%)")
    ax_acc.set_title("Training & Validation Accuracy")
    ax_acc.legend(facecolor="#1A1A2E", edgecolor="#555577", labelcolor="white")

    best_val = max(history["val_acc"])
    fig.suptitle(
        f"Training Curves  —  best val top-1: {best_val:.2f}%",
        color="white", fontsize=11, fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved training curves → {save_path}")


# ── 9f. Run all visualisations at once ──────────────────────────────────────
def run_visualizations(args):
    """
    Generates all four static visualisations using a random input tensor
    (or real data if a manifest is available). Saves PNGs to --vis_dir.
    """
    out_dir = Path(args.vis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Visualisations ──────────────────────────────────────────────")

    # Build model
    data_dir       = Path(args.data)
    class_map_file = data_dir / "class_to_idx.json"

    if class_map_file.exists():
        with open(class_map_file) as f:
            class_to_idx = json.load(f)
        class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
        num_classes  = len(class_names)
    else:
        num_classes  = 101
        class_names  = [f"Class_{i:03d}" for i in range(num_classes)]

    model = VideoTransformer(
        num_classes=num_classes,
        img_size=args.img_size,
        patch_size=args.patch_size,
        temporal_patch_size=args.temporal_patch_size,
        num_frames=args.num_frames,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
    )

    # Load checkpoint if available
    ckpt = Path(args.checkpoint_dir) / "best.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
        print(f"  Loaded checkpoint: val_acc={state['val_acc']:.2f}%")
    else:
        print("  No checkpoint found — using random weights (untrained model).")
        print("  Visualisations will show structure, not meaningful predictions.")

    model.eval()

    # Use a real sample if data is available, else random tensor
    test_manifest = data_dir / "test_manifest.json"
    true_label    = None
    if test_manifest.exists():
        ds  = VideoFrameDataset(test_manifest, num_frames=args.num_frames)
        vid, lbl = ds[0]
        x        = vid.unsqueeze(0)        # (1, C, T, H, W)
        true_label = class_names[lbl]
        print(f"  Using real sample: {true_label}")
    else:
        x = torch.randn(1, 3, args.num_frames, args.img_size, args.img_size)
        print("  Using random input tensor (no dataset found).")

    # 1. Architecture diagram
    plot_architecture(model, out_dir / "architecture.png")

    # 2. Token grid (what the patches look like on the frame)
    plot_token_grid(model, x, out_dir / "token_grid.png")

    # 3. Attention maps from every block and head
    plot_attention_maps(model, x, out_dir / "attention_maps.png")

    # 4. Prediction confidence bars
    plot_prediction(
        model, x, class_names,
        out_dir / "predictions.png",
        true_label=true_label,
    )

    print(f"\n✓ All visualisations saved to {out_dir}/")
    print("  Files:")
    for f in sorted(out_dir.glob("*.png")):
        print(f"    {f.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  10.  SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════
def smoke_test():
    print("\n── Smoke Test ──────────────────────────────────────────────────")
    device = torch.device("cpu")

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
    x      = torch.randn(B, 3, 16, 112, 112)
    logits = model(x)
    print(f"  Input  : {tuple(x.shape)}")
    print(f"  Output : {tuple(logits.shape)}   (should be [{B}, 101])")
    assert logits.shape == (B, 101), "Shape mismatch!"

    # Test forward_with_attn
    logits2, attn_list = model.forward_with_attn(x[:1])
    assert len(attn_list) == 4, "Wrong number of attention tensors"
    assert attn_list[0].shape == (1, 3, 393, 393), f"Unexpected attn shape: {attn_list[0].shape}"

    loss = logits.sum()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"  Grad norms (sample): {[f'{g:.4f}' for g in grad_norms[:5]]}")
    print("  ✓ Smoke test passed!\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="VideoTransformer training & visualisation")
    parser.add_argument("--train",               action="store_true",
                        help="Run the training loop")
    parser.add_argument("--visualize",           action="store_true",
                        help="Generate visualisation PNGs and exit")
    parser.add_argument("--data",                default="./data")
    parser.add_argument("--vis_dir",             default="./visualizations",
                        help="Folder to save PNG outputs")
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

    if args.visualize:
        run_visualizations(args)
    elif args.train:
        train(args)
    else:
        print("Tips:")
        print("  python video_transformer.py --visualize          # generate all plots")
        print("  python video_transformer.py --train --data ./data")
        print("  (run 'python download_synthetic_data.py --output_dir ./data' first if no data yet)")


if __name__ == "__main__":
    main()