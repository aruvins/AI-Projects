"""
World Model Simulator
=====================
A predictive environment model that simulates and predicts future states.

Core AI Concepts:
- World Models: Neural networks that learn to predict environment dynamics
- Latent Space: Compressed representation of environment states
- Transition Model: Predicts next state given current state + action
- Reward Model: Predicts reward signals
- Policy: Strategy for selecting actions (random, greedy, learned)
- Rollout: Simulated trajectory through the world model
"""

import numpy as np
import random
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap


# ── Colour palette ────────────────────────────
BLUE   = "#378ADD"
AMBER  = "#EF9F27"
TEAL   = "#1D9E75"
CORAL  = "#D85A30"
PURPLE = "#7F77DD"
GRAY   = "#888780"
BG     = "#F8F7F4"
DARK   = "#2C2C2A"


# ─────────────────────────────────────────────
# 1. ENVIRONMENT
# ─────────────────────────────────────────────

class CartPoleEnv:
    GRAVITY       = 9.8
    CART_MASS     = 1.0
    POLE_MASS     = 0.1
    POLE_HALF_LEN = 0.5
    FORCE_MAG     = 10.0
    DT            = 0.02
    FAIL_ANGLE    = 12 * math.pi / 180
    FAIL_POS      = 2.4

    def __init__(self):
        self.state = None
        self.steps = 0
        self.reset()

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        x, xd, theta, thetad = self.state
        force      = self.FORCE_MAG if action == 1 else -self.FORCE_MAG
        total_mass = self.CART_MASS + self.POLE_MASS
        pole_mc    = self.POLE_MASS * self.POLE_HALF_LEN
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        tmp        = (force + pole_mc * thetad**2 * sin_t) / total_mass
        theta_acc  = (self.GRAVITY * sin_t - cos_t * tmp) / \
                     (self.POLE_HALF_LEN * (4/3 - self.POLE_MASS * cos_t**2 / total_mass))
        x_acc      = tmp - pole_mc * theta_acc * cos_t / total_mass
        x      += self.DT * xd
        xd     += self.DT * x_acc
        theta  += self.DT * thetad
        thetad += self.DT * theta_acc
        self.state = np.array([x, xd, theta, thetad])
        self.steps += 1
        done   = (abs(x) > self.FAIL_POS or abs(theta) > self.FAIL_ANGLE or self.steps >= 500)
        reward = 1.0 if not done else 0.0
        return self.state.copy(), reward, done


# ─────────────────────────────────────────────
# 2. REPLAY BUFFER
# ─────────────────────────────────────────────

@dataclass
class Transition:
    state:      np.ndarray
    action:     int
    next_state: np.ndarray
    reward:     float
    done:       bool

class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# 3. WORLD MODEL
# ─────────────────────────────────────────────

class NeuralLayer:
    def __init__(self, in_dim, out_dim, activation="relu"):
        scale   = math.sqrt(2.0 / in_dim)
        self.W  = np.random.randn(in_dim, out_dim) * scale
        self.b  = np.zeros(out_dim)
        self.act = activation

    def forward(self, x):
        z = x @ self.W + self.b
        if self.act == "relu":  return np.maximum(0, z)
        if self.act == "tanh":  return np.tanh(z)
        return z

    def backward(self, x, grad_out, lr):
        z = x @ self.W + self.b
        if self.act == "relu":  dz = grad_out * (z > 0).astype(float)
        elif self.act == "tanh": dz = grad_out * (1 - np.tanh(z)**2)
        else:                    dz = grad_out
        self.W -= lr * (x.T @ dz)
        self.b -= lr * dz.sum(axis=0)
        return dz @ self.W.T


class WorldModel:
    STATE_DIM  = 4
    ACTION_DIM = 2
    HIDDEN     = 64
    OUT_DIM    = 6

    def __init__(self, lr=1e-3):
        in_dim = self.STATE_DIM + self.ACTION_DIM
        self.layers = [
            NeuralLayer(in_dim,      self.HIDDEN,  "relu"),
            NeuralLayer(self.HIDDEN, self.HIDDEN,  "relu"),
            NeuralLayer(self.HIDDEN, self.OUT_DIM, "linear"),
        ]
        self.lr = lr
        self.loss_history = []

    def _encode(self, state, action):
        oh = np.zeros(self.ACTION_DIM); oh[action] = 1.0
        return np.concatenate([state, oh])

    def _fwd(self, x):
        acts = [x]
        for L in self.layers:
            x = L.forward(x)
            acts.append(x)
        return acts

    def predict(self, state, action):
        x   = self._encode(state, action)[None, :]
        out = self._fwd(x)[-1][0]
        logit     = float(np.clip(out[5], -30, 30))
        pred_done = float(1 / (1 + math.exp(-logit)))
        return state + out[:4], float(np.clip(out[4], 0, 1)), pred_done

    def train_batch(self, transitions):
        if not transitions: return 0.0
        total = 0.0
        for t in transitions:
            x    = self._encode(t.state, t.action)[None, :]
            acts = self._fwd(x)
            out  = acts[-1]
            tgt  = np.array([[*(t.next_state - t.state), t.reward,
                               10.0 if t.done else -10.0]])
            grad = 2 * (out - tgt) / len(transitions)
            total += float(np.mean((out - tgt)**2))
            for i in reversed(range(len(self.layers))):
                grad = self.layers[i].backward(acts[i], grad, self.lr)
        avg = total / len(transitions)
        self.loss_history.append(avg)
        return avg


# ─────────────────────────────────────────────
# 4. POLICY
# ─────────────────────────────────────────────

class Policy:
    @staticmethod
    def random(_):
        return random.randint(0, 1)

    @staticmethod
    def balance(state):
        return 1 if state[2] > 0 else 0

    @staticmethod
    def epsilon_greedy(state, world_model, epsilon=0.1, horizon=5):
        if random.random() < epsilon:
            return random.randint(0, 1)
        best_a, best_v = 0, -float("inf")
        for action in range(2):
            v, s = 0.0, state.copy()
            a = action
            for h in range(horizon):
                s, r, dp = world_model.predict(s, a)
                v += r * (0.99 ** h)
                if dp > 0.5: break
                a = 1 if s[2] > 0 else 0
            if v > best_v:
                best_v, best_a = v, action
        return best_a


# ─────────────────────────────────────────────
# 5. IMAGINATION ENGINE
# ─────────────────────────────────────────────

@dataclass
class ImaginaryRollout:
    states:     list
    actions:    list
    rewards:    list
    done_probs: list
    total_reward: float = field(init=False)
    def __post_init__(self): self.total_reward = sum(self.rewards)

class ImaginationEngine:
    def __init__(self, world_model):
        self.model = world_model

    def rollout(self, start_state, policy_fn, horizon=20):
        states, actions, rewards, dps = [], [], [], []
        s = start_state.copy()
        for _ in range(horizon):
            a = policy_fn(s)
            ns, r, dp = self.model.predict(s, a)
            states.append(s.copy()); actions.append(a)
            rewards.append(r);       dps.append(dp)
            s = ns
            if dp > 0.9: break
        return ImaginaryRollout(states, actions, rewards, dps)


# ─────────────────────────────────────────────
# 6. TRAINER
# ─────────────────────────────────────────────

class WorldModelTrainer:
    def __init__(self, env, world_model, buffer, batch_size=64):
        self.env    = env
        self.model  = world_model
        self.buf    = buffer
        self.bs     = batch_size
        self.engine = ImaginationEngine(world_model)
        self.real_rewards  = []
        self.dream_rewards = []
        self.model_losses  = []
        self.train_steps   = 0

        # per-episode prediction errors for viz
        self.mae_history = []

    def collect_episode(self, epsilon=0.3):
        state, total, done = self.env.reset(), 0.0, False
        while not done:
            action = Policy.epsilon_greedy(state, self.model, epsilon=epsilon) \
                     if len(self.buf) > 200 else Policy.random(state)
            ns, r, done = self.env.step(action)
            self.buf.push(state, action, ns, r, done)
            total += r; state = ns
        return total

    def train_model(self, n_steps=10):
        if len(self.buf) < self.bs: return 0.0
        total = 0.0
        for _ in range(n_steps):
            total += self.model.train_batch(self.buf.sample(self.bs))
            self.train_steps += 1
        avg = total / n_steps
        self.model_losses.append(avg)
        return avg

    def dream_episode(self, horizon=30):
        start = np.random.uniform(-0.05, 0.05, size=4)
        roll  = self.engine.rollout(
            start,
            lambda s: Policy.epsilon_greedy(s, self.model, epsilon=0.05),
            horizon=horizon)
        self.dream_rewards.append(roll.total_reward)
        return roll

    def _snapshot_mae(self, n_steps=50):
        """Quick in-loop MAE snapshot across 4 state dims."""
        state = self.env.reset()
        errs  = []
        for _ in range(n_steps):
            a = random.randint(0, 1)
            rn, _, rd = self.env.step(a)
            pn, _,  _ = self.model.predict(state, a)
            errs.append(np.abs(rn - pn))
            state = self.env.reset() if rd else rn
        self.mae_history.append(np.mean(errs, axis=0))

    def train(self, n_episodes=30, model_steps_per_ep=20, verbose=True):
        print("=" * 60)
        print(" WORLD MODEL SIMULATOR — Training")
        print("=" * 60)
        print(f"{'Ep':>4} | {'Real':>6} | {'Loss':>8} | {'Dream':>6} | {'Buf':>5}")
        print("-" * 40)

        for ep in range(1, n_episodes + 1):
            eps    = max(0.05, 0.4 - ep * 0.01)
            real_r = self.collect_episode(epsilon=eps)
            loss   = self.train_model(n_steps=model_steps_per_ep)
            self.real_rewards.append(real_r)
            dream_r = 0.0
            if len(self.buf) > 200:
                dream_r = self.dream_episode().total_reward
            self._snapshot_mae()
            if verbose:
                print(f"{ep:>4} | {real_r:>6.1f} | {loss:>8.4f} | {dream_r:>6.1f} | {len(self.buf):>5}")

        print("=" * 60)
        return {"real_rewards": self.real_rewards,
                "dream_rewards": self.dream_rewards,
                "model_losses": self.model_losses}


# ─────────────────────────────────────────────
# 7. VISUALISATION
# ─────────────────────────────────────────────

def smooth(data, w=3):
    """Simple moving average."""
    if len(data) < w: return data
    return [np.mean(data[max(0, i-w):i+1]) for i in range(len(data))]

def draw_cartpole(ax, cart_x, pole_angle_deg, title="", real=True):
    """Draw a single CartPole snapshot."""
    ax.set_xlim(-3, 3); ax.set_ylim(-0.4, 1.6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=9, color=DARK, pad=4)

    # Track
    ax.plot([-2.4, 2.4], [0, 0], color=GRAY, lw=1.5, zorder=1)
    ax.plot([-2.4, -2.4], [-0.05, 0.05], color=GRAY, lw=1)
    ax.plot([ 2.4,  2.4], [-0.05, 0.05], color=GRAY, lw=1)

    # Cart
    cart_w, cart_h = 0.5, 0.2
    cart = mpatches.FancyBboxPatch(
        (cart_x - cart_w/2, -cart_h/2), cart_w, cart_h,
        boxstyle="round,pad=0.02",
        facecolor=BLUE if real else AMBER, edgecolor=DARK, linewidth=1.2, zorder=3)
    ax.add_patch(cart)

    # Wheels
    for dx in [-0.15, 0.15]:
        wheel = plt.Circle((cart_x + dx, -cart_h/2 - 0.07), 0.07,
                            color=DARK, zorder=4)
        ax.add_patch(wheel)

    # Pole
    pole_rad = math.radians(pole_angle_deg)
    pole_len = 1.0
    px = cart_x + pole_len * math.sin(pole_rad)
    py = cart_h/2 + pole_len * math.cos(pole_rad)
    ax.plot([cart_x, px], [cart_h/2, py],
            color=CORAL if real else PURPLE, lw=4, solid_capstyle="round", zorder=3)
    ax.plot(px, py, "o", color=DARK, ms=8, zorder=5)

    # Labels
    color = BLUE if real else AMBER
    label = "Real env" if real else "World model"
    ax.text(0, -0.35, label, ha="center", fontsize=8,
            color=color, fontweight="bold")


def make_visualisation(trainer: WorldModelTrainer, env: CartPoleEnv,
                       save_path="world_model_dashboard.png"):
    """
    Produce a 3×3 dashboard showing:
      Row 0: CartPole snapshots (real vs imagined)
      Row 1: Reward curves + model loss
      Row 2: Per-dim MAE over training + prediction error scatter
    """
    fig = plt.figure(figsize=(16, 13), facecolor=BG)
    fig.suptitle("World Model Simulator — Training Dashboard",
                 fontsize=16, color=DARK, fontweight="bold", y=0.97)

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.55, wspace=0.38,
                           left=0.07, right=0.97,
                           top=0.92, bottom=0.06)

    # ── helpers ──────────────────────────────

    def styled_ax(ax, title, xlabel="", ylabel=""):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color("#CCCBC4")
            sp.set_linewidth(0.6)
        ax.tick_params(colors=GRAY, labelsize=8)
        ax.set_title(title, fontsize=10, color=DARK, pad=6, fontweight="bold")
        if xlabel: ax.set_xlabel(xlabel, fontsize=8, color=GRAY)
        if ylabel: ax.set_ylabel(ylabel, fontsize=8, color=GRAY)
        ax.grid(True, color="#E0DED8", linewidth=0.5, linestyle="--")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ROW 0 — CartPole snapshots
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Grab a real trajectory
    real_states = []
    s = env.reset()
    env_snap = CartPoleEnv()
    env_snap.state = np.array([0.0, 0.0, 0.08, 0.0])
    env_snap.steps = 0
    for _ in range(20):
        a = Policy.balance(s)
        ns, _, done = env.step(a)
        real_states.append(ns.copy())
        s = env.reset() if done else ns

    # Grab an imagined trajectory from the trained model
    engine = ImaginationEngine(trainer.model)
    start  = np.array([0.0, 0.0, 0.08, 0.0])
    rollout = engine.rollout(start, Policy.balance, horizon=20)

    # Pick 3 evenly-spaced frames
    frames = [0, len(real_states)//2, len(real_states)-1]
    for col, fi in enumerate(frames):
        ax_r = fig.add_subplot(gs[0, col])
        rs   = real_states[fi]
        # mirrored imagined state
        ii   = min(fi, len(rollout.states)-1)
        im_s = rollout.states[ii]

        draw_cartpole(ax_r,
                      cart_x=float(np.clip(rs[0], -2.2, 2.2)),
                      pole_angle_deg=math.degrees(rs[2]),
                      title=f"Step {fi+1}  |  Real vs Imagined",
                      real=True)

        # Overlay imagined pole in amber/purple
        ax_r.set_xlim(-3, 3); ax_r.set_ylim(-0.4, 1.6)
        cx = float(np.clip(im_s[0], -2.2, 2.2))
        pr = math.radians(math.degrees(im_s[2]))
        pl = 1.0
        px = cx + pl * math.sin(pr)
        py = 0.1 + pl * math.cos(pr)
        ax_r.plot([cx, px], [0.1, py],
                  color=PURPLE, lw=2.5, linestyle="--",
                  solid_capstyle="round", zorder=5, alpha=0.8)
        ax_r.plot(px, py, "D", color=PURPLE, ms=5, zorder=6)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ROW 1 — Reward curves + loss
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 1a: Real vs dream rewards
    ax1 = fig.add_subplot(gs[1, :2])
    styled_ax(ax1, "Episode Rewards: Real Environment vs World Model Imagination",
              xlabel="Episode", ylabel="Total reward")
    eps = list(range(1, len(trainer.real_rewards)+1))
    ax1.fill_between(eps, trainer.real_rewards, alpha=0.12, color=BLUE)
    ax1.plot(eps, trainer.real_rewards, color=BLUE, lw=1.5, alpha=0.5, label="Real (raw)")
    ax1.plot(eps, smooth(trainer.real_rewards, 5), color=BLUE, lw=2.5, label="Real (smoothed)")
    if trainer.dream_rewards:
        dr = trainer.dream_rewards
        ax1.fill_between(range(1, len(dr)+1), dr, alpha=0.1, color=AMBER)
        ax1.plot(range(1, len(dr)+1), dr, color=AMBER, lw=1.5, alpha=0.5, label="Dream (raw)")
        ax1.plot(range(1, len(dr)+1), smooth(dr, 5), color=AMBER, lw=2.5, label="Dream (smoothed)")
    ax1.legend(fontsize=8, framealpha=0.7, loc="upper left")

    # 1b: Model loss
    ax2 = fig.add_subplot(gs[1, 2])
    styled_ax(ax2, "World Model Loss", xlabel="Training episode", ylabel="MSE loss")
    if trainer.model_losses:
        ml = trainer.model_losses
        ax2.fill_between(range(1, len(ml)+1), ml, alpha=0.15, color=CORAL)
        ax2.plot(range(1, len(ml)+1), ml, color=CORAL, lw=2)
        ax2.plot(range(1, len(ml)+1), smooth(ml, 5), color=CORAL, lw=2.5,
                 linestyle="--", label="Smoothed")
        ax2.legend(fontsize=8)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ROW 2 — Per-dim MAE over training + error scatter
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    dim_labels  = ["Cart pos", "Cart vel", "Pole angle", "Pole vel"]
    dim_colors  = [BLUE, TEAL, CORAL, PURPLE]

    # 2a: MAE per dimension over training
    ax3 = fig.add_subplot(gs[2, :2])
    styled_ax(ax3, "Prediction Error (MAE) per State Dimension — as Model Learns",
              xlabel="Episode", ylabel="MAE")
    if trainer.mae_history:
        mae_arr = np.array(trainer.mae_history)  # (n_ep, 4)
        for d, (lab, col) in enumerate(zip(dim_labels, dim_colors)):
            series = mae_arr[:, d]
            ax3.fill_between(range(1, len(series)+1), series, alpha=0.08, color=col)
            ax3.plot(range(1, len(series)+1), series, color=col, lw=2, label=lab)
        ax3.legend(fontsize=8, ncol=2, framealpha=0.7)

    # 2b: Real vs predicted state scatter (pole angle — most informative)
    ax4 = fig.add_subplot(gs[2, 2])
    styled_ax(ax4, "Real vs Predicted Pole Angle", xlabel="Real angle (rad)", ylabel="Predicted")

    reals, preds = [], []
    s = env.reset()
    for _ in range(300):
        a = random.randint(0, 1)
        rn, _, rd = env.step(a)
        pn, _, _  = trainer.model.predict(s, a)
        reals.append(rn[2]); preds.append(pn[2])
        s = env.reset() if rd else rn

    reals = np.array(reals); preds = np.array(preds)
    err_mag = np.abs(reals - preds)
    sc = ax4.scatter(reals, preds, c=err_mag, cmap="YlOrRd",
                     s=14, alpha=0.7, edgecolors="none")
    lo, hi = min(reals.min(), preds.min()), max(reals.max(), preds.max())
    ax4.plot([lo, hi], [lo, hi], color=TEAL, lw=1.5, linestyle="--",
             label="Perfect prediction")
    ax4.legend(fontsize=8)
    cbar = fig.colorbar(sc, ax=ax4, pad=0.02)
    cbar.set_label("Abs. error", fontsize=7, color=GRAY)
    cbar.ax.tick_params(labelsize=7)

    # ── Legend strip ─────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=BLUE,   label="Real environment"),
        mpatches.Patch(facecolor=AMBER,  label="World model (imagination)"),
        mpatches.Patch(facecolor=CORAL,  label="Model loss"),
        mpatches.Patch(facecolor=PURPLE, label="Imagined trajectory (dashed)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=8.5, framealpha=0.8, bbox_to_anchor=(0.5, 0.01))

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\n  Saved dashboard → {save_path}")


# ─────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    env    = CartPoleEnv()
    model  = WorldModel(lr=5e-4)
    buffer = ReplayBuffer(capacity=20_000)

    print("Pre-filling replay buffer…")
    s = env.reset()
    for _ in range(500):
        a = Policy.random(s)
        ns, r, done = env.step(a)
        buffer.push(s, a, ns, r, done)
        s = env.reset() if done else ns
    print(f"Buffer: {len(buffer)} transitions\n")

    trainer = WorldModelTrainer(env, model, buffer, batch_size=64)
    stats   = trainer.train(n_episodes=30, model_steps_per_ep=20)

    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "world_model_dashboard.png")
    print("\nGenerating dashboard…")
    make_visualisation(trainer, env, save_path=out_path)
    print(f"  Dashboard saved → {out_path}")

    recent_real  = np.mean(stats["real_rewards"][-10:])
    recent_dream = np.mean([r for r in stats["dream_rewards"][-10:] if r > 0] or [0])
    print(f"\n  Real reward  (last 10 ep avg): {recent_real:.1f}")
    print(f"  Dream reward (last 10 ep avg): {recent_dream:.1f}")
    print(f"  Training steps: {trainer.train_steps}")