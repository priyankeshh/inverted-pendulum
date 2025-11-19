"""Unified CartPole DQN script

Aim: Design and simulate a Reinforcement Learning (DQN) agent to stabilize an
inverted pendulum (CartPole-v1). This file merges the assignment-friendly
`cartpole_dqn.py` and the more feature-rich `rl_project.py` into one configurable,
clean script suitable for both experimentation and reporting.

Features merged:
- CLI arguments for all key hyperparameters (episodes, batch size, buffer size, etc.)
- Deterministic seeding
- Replay buffer with optional prefill (for more stable early learning)
- Epsilon-greedy exploration with decay
- Target network updated either by steps or by episode (configurable)
- Early stopping when solved (average over last N episodes threshold)
- Model checkpointing when average reward improves
- Reward plotting with smoothing and artifact saving
- Evaluation mode (no exploration)
- GIF recording (lightweight) and MP4 recording (higher quality) of trained agent
- Artifacts directory management

Usage examples (PowerShell):
  python cartpole_final.py --episodes 600 --prefill 5000
  python cartpole_final.py --evaluate --episodes 0
  python cartpole_final.py --record-gif
  python cartpole_final.py --record-mp4 --mp4-name best_run.mp4

"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Tuple, Optional

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium import spaces
try:
    from PIL import Image, ImageDraw, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ----------------------- Artifacts -----------------------
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)
MODEL_DIR = ARTIFACTS / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "dqn_cartpole.pth"
PLOT_PATH = ARTIFACTS / "cartpole_rewards.png"
GIF_PATH = ARTIFACTS / "cartpole_agent.gif"


# ----------------------- Seeding -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------- Custom CartPole with Better Colors -----------------------
class StyledCartPoleEnv(gym.Wrapper):
    """Wrapper to customize CartPole visualization with enhanced colors"""
    
    def __init__(self, env, 
                 cart_color=(50, 100, 200),      # Blue cart
                 pole_color=(200, 50, 50),        # Red pole
                 track_color=(100, 100, 100)):    # Gray track
        super().__init__(env)
        self.env = env
        self.cart_color = np.array(cart_color, dtype=np.uint8)
        self.pole_color = np.array(pole_color, dtype=np.uint8)
        self.track_color = np.array(track_color, dtype=np.uint8)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        """Override rendering to apply custom color scheme"""
        frame = self.env.render()
        
        if self.render_mode == "rgb_array" and frame is not None:
            # Apply color enhancement
            frame = self._apply_custom_colors(frame)
        
        return frame
    
    def _apply_custom_colors(self, frame):
        """Apply custom color scheme to the rendered frame"""
        # Convert to float for processing
        enhanced = frame.astype(np.float32)
        
        # Enhance contrast slightly
        enhanced = np.clip((enhanced - 128) * 1.15 + 128, 0, 255)
        
        # The gymnasium CartPole renders:
        # - Cart as dark gray/black (~0-50 range)
        # - Pole as dark gray/black 
        # - Background as white (255)
        # - Track as light gray (~200)
        
        # Create mask for dark pixels (cart and pole)
        dark_mask = (frame[:, :, 0] < 80) & (frame[:, :, 1] < 80) & (frame[:, :, 2] < 80)
        
        # Identify pole vs cart by position (pole is thinner, more vertical)
        height, width = frame.shape[:2]
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Pole is typically in upper portion and narrower
        pole_mask = dark_mask & (y_coords < height * 0.65)
        cart_mask = dark_mask & (y_coords >= height * 0.65)
        
        # Apply colors
        enhanced[pole_mask] = self.pole_color
        enhanced[cart_mask] = self.cart_color
        
        # Enhance track (light gray pixels)
        track_mask = (frame[:, :, 0] > 150) & (frame[:, :, 0] < 220) & \
                     (frame[:, :, 1] > 150) & (frame[:, :, 1] < 220)
        enhanced[track_mask] = self.track_color
        
        return enhanced.astype(np.uint8)


def make_styled_cartpole(render_mode=None, 
                         cart_color=(50, 100, 200), 
                         pole_color=(200, 50, 50),
                         track_color=(100, 100, 100)):
    """Create CartPole with custom color scheme
    
    Args:
        cart_color: RGB tuple for cart (default: blue)
        pole_color: RGB tuple for pole (default: red)
        track_color: RGB tuple for track (default: gray)
    """
    env = gym.make("CartPole-v1", render_mode=render_mode)
    return StyledCartPoleEnv(env, cart_color, pole_color, track_color)


# ----------------------- Network -----------------------
class QNetwork(nn.Module):
    def __init__(self, obs_size: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------- Replay Buffer -----------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []  # list of tuples
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            np.vstack(states),
            actions,
            rewards.astype(np.float32),
            np.vstack(next_states),
            dones.astype(np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ----------------------- DQN Agent -----------------------
class DQNAgent:
    def __init__(
        self,
        obs_size: int,
        n_actions: int,
        lr: float,
        gamma: float,
        device: torch.device,
    ):
        self.device = device
        self.gamma = gamma
        self.q = QNetwork(obs_size, n_actions).to(device)
        self.q_target = QNetwork(obs_size, n_actions).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.n_actions = n_actions

    def select_action(self, state, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.q(state_t)
            return int(torch.argmax(q_vals, dim=1).item())

    def optimize(self, batch, loss_fn=nn.MSELoss()) -> float:
        states, actions, rewards, next_states, dones = batch
        states_v = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states_v = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_v = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_v = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_v = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_vals = self.q(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = self.q_target(next_states_v).max(1)[0]
            target = rewards_v + self.gamma * next_q * (1 - dones_v)

        loss = loss_fn(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self):
        self.q_target.load_state_dict(self.q.state_dict())


# ----------------------- Training -----------------------
def train(
    episodes: int,
    max_steps: int,
    batch_size: int,
    buffer_size: int,
    prefill: int,
    gamma: float,
    lr: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    target_update_freq: int,
    target_update_mode: str,
    seed: int,
    early_stop_avg: Optional[float],
    early_stop_window: int,
) -> Tuple[list, list]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_size, n_actions, lr=lr, gamma=gamma, device=device)
    replay = ReplayBuffer(buffer_size)

    # Prefill replay with random transitions for stability
    if prefill > 0:
        state, _ = env.reset()
        for _ in range(prefill):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.push(state, action, reward, next_state, done)
            state = next_state if not done else env.reset()[0]

    epsilon = epsilon_start
    rewards_history = []
    losses = []
    total_steps = 0
    best_avg = -float("inf")

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(max_steps):
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            total_steps += 1

            # Optimize
            if len(replay) >= batch_size:
                batch = replay.sample(batch_size)
                loss = agent.optimize(batch)
                losses.append(loss)

            # Target network update modes
            if target_update_mode == "steps" and total_steps % target_update_freq == 0:
                agent.update_target()

            if done:
                break

        rewards_history.append(ep_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Episode-based target update (alternative mode)
        if target_update_mode == "episodes" and ep % target_update_freq == 0:
            agent.update_target()

        # Logging every 10 episodes
        if ep % 10 == 0:
            window_mean = (
                np.mean(rewards_history[-early_stop_window:])
                if len(rewards_history) >= early_stop_window
                else np.mean(rewards_history)
            )
            print(
                f"Episode {ep}/{episodes} | Reward: {ep_reward:.1f} | Avg({early_stop_window}): "
                f"{window_mean:.2f} | Eps: {epsilon:.3f}"
            )
            if window_mean > best_avg:
                best_avg = window_mean
                torch.save(agent.q.state_dict(), MODEL_PATH)

        # Early stopping criterion
        if early_stop_avg is not None and len(rewards_history) >= early_stop_window:
            recent_avg = np.mean(rewards_history[-early_stop_window:])
            if recent_avg >= early_stop_avg:
                print(
                    f"Solved (avg >= {early_stop_avg}) in {ep} episodes! Saving final model."
                )
                torch.save(agent.q.state_dict(), MODEL_PATH)
                break

    env.close()

    # Final save if not saved yet
    if not MODEL_PATH.exists():
        torch.save(agent.q.state_dict(), MODEL_PATH)

    plot_rewards(rewards_history)
    print("Training complete.")
    print(f"Model stored at {MODEL_PATH}")
    print(f"Reward plot stored at {PLOT_PATH}")
    return rewards_history, losses


# ----------------------- Plotting -----------------------
def plot_rewards(rewards: list):
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, label="Episode reward", alpha=0.8)
    if len(rewards) >= 20:
        smooth = np.convolve(rewards, np.ones(20) / 20, mode="valid")
        plt.plot(range(19, len(rewards)), smooth, label="20-ep avg", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("CartPole DQN Training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()


# ----------------------- Evaluation -----------------------
def evaluate(episodes: int, render: bool = False, seed: int = 42):
    if not MODEL_PATH.exists():
        print("No model found; train first.")
        return
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(obs_size, n_actions, lr=1e-3, gamma=0.99, device=device)
    agent.q.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    agent.q.eval()

    returns = []
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total = 0
        done = False
        truncated = False
        while not (done or truncated):
            action = agent.select_action(state, epsilon=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        returns.append(total)
        print(f"Eval Episode {ep}: Return={total:.1f}")
    env.close()
    print(f"Average return: {np.mean(returns):.2f}")
    return returns


# ----------------------- Recording (GIF / MP4) -----------------------
def record_gif(gif_path: Path = GIF_PATH, seed: int = 42):
    if not MODEL_PATH.exists():
        print("No model found; train first.")
        return
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use styled environment with custom colors
    env = make_styled_cartpole(
        render_mode="rgb_array",
        cart_color=(60, 120, 216),   # Bright blue cart
        pole_color=(220, 60, 60),     # Red pole
        track_color=(80, 80, 80)      # Dark gray track
    )
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(obs_size, n_actions, lr=1e-3, gamma=0.99, device=device)
    agent.q.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    agent.q.eval()

    frames = []
    state, _ = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        frame = env.render()
        frames.append(frame)
        action = agent.select_action(state, epsilon=0.0)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"GIF saved to {gif_path}")


def record_mp4(filename: str = "cartpole_run.mp4", seed: int = 42):
    if not MODEL_PATH.exists():
        print("No model found; train first.")
        return
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use styled environment with custom colors
    env = make_styled_cartpole(
        render_mode="rgb_array",
        cart_color=(60, 120, 216),   # Bright blue cart
        pole_color=(220, 60, 60),     # Red pole
        track_color=(80, 80, 80)      # Dark gray track
    )
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(obs_size, n_actions, lr=1e-3, gamma=0.99, device=device)
    agent.q.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    agent.q.eval()

    frames = []
    state, _ = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        frame = env.render()
        if isinstance(frame, tuple):
            frame = frame[0]
        frames.append(frame)
        action = agent.select_action(state, epsilon=0.0)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()

    writer = imageio.get_writer(ARTIFACTS / filename, fps=30, codec="libx264", quality=8)
    for f in frames:
        if f.dtype != np.uint8:
            f = (255 * np.clip(f, 0, 1)).astype("uint8")
        writer.append_data(f)
    writer.close()
    print(f"MP4 saved to {ARTIFACTS / filename}")


# ----------------------- CLI -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Unified DQN for CartPole (inverted pendulum)")
    p.add_argument("--episodes", type=int, default=500, help="Training episodes (default 500). Set 0 to skip training.")
    p.add_argument("--max-steps", type=int, default=500, help="Max steps per episode.")
    p.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    p.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer capacity.")
    p.add_argument("--prefill", type=int, default=5000, help="Initial random transitions to prefill replay buffer.")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    p.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    p.add_argument("--eps-start", type=float, default=1.0, help="Initial epsilon.")
    p.add_argument("--eps-end", type=float, default=0.05, help="Minimum epsilon.")
    p.add_argument("--eps-decay", type=float, default=0.995, help="Multiplicative epsilon decay per episode.")
    p.add_argument("--target-update-freq", type=int, default=1000, help="Frequency for target net update (depends on mode).")
    p.add_argument(
        "--target-update-mode",
        choices=["steps", "episodes"],
        default="steps",
        help="Update target by steps or by episodes.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--early-stop-avg", type=float, default=450.0, help="Avg reward threshold for early stop (set negative to disable).")
    p.add_argument("--early-stop-window", type=int, default=50, help="Window size for early-stop averaging.")
    p.add_argument("--evaluate", action="store_true", help="Evaluate saved model (no exploration).")
    p.add_argument("--eval-episodes", type=int, default=5, help="Episodes to evaluate.")
    p.add_argument("--record-gif", action="store_true", help="Record a GIF of the trained agent.")
    p.add_argument("--record-mp4", action="store_true", help="Record an MP4 video of the trained agent.")
    p.add_argument("--mp4-name", type=str, default="cartpole_run.mp4", help="Filename for MP4 inside artifacts.")
    return p.parse_args()


def main():
    args = parse_args()

    start = time.time()
    if args.episodes > 0:
        early = args.early_stop_avg if args.early_stop_avg >= 0 else None
        rewards, losses = train(
            episodes=args.episodes,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            prefill=args.prefill,
            gamma=args.gamma,
            lr=args.lr,
            epsilon_start=args.eps_start,
            epsilon_end=args.eps_end,
            epsilon_decay=args.eps_decay,
            target_update_freq=args.target_update_freq,
            target_update_mode=args.target_update_mode,
            seed=args.seed,
            early_stop_avg=early,
            early_stop_window=args.early_stop_window,
        )
        print(f"Training took {time.time() - start:.2f} seconds")
    else:
        print("Skipping training (--episodes=0)")

    if args.evaluate:
        evaluate(args.eval_episodes, render=False, seed=args.seed)

    if args.record_gif:
        record_gif(GIF_PATH, seed=args.seed)

    if args.record_mp4:
        record_mp4(args.mp4_name, seed=args.seed)


if __name__ == "__main__":
    main()
