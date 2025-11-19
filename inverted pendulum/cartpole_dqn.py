import argparse
from pathlib import Path
import random

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS / "cartpole_model.pth"
PLOT_PATH = ARTIFACTS / "cartpole_rewards.png"
GIF_PATH = ARTIFACTS / "cartpole_agent.gif"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.layers(x)


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            states,
            actions,
            rewards.astype(np.float32),
            next_states,
            dones.astype(np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def train_dqn(episodes=400, batch_size=64, gamma=0.99, lr=1e-3):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_net = DQN(state_size, action_size).to(device)
    target_net = DQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer()

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    target_update = 10

    rewards_history = []
    losses = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = policy_net(state_tensor)
                    action = torch.argmax(q_values, dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                loss = optimize_model(policy_net, target_net, optimizer, batch, gamma, device)
                losses.append(loss)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        print(f"Episode {ep + 1}/{episodes}: reward={total_reward:.1f}  epsilon={epsilon:.3f}")

        if (ep + 1) % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env.close()

    torch.save(policy_net.state_dict(), MODEL_PATH)
    plot_rewards(rewards_history)

    print("\nTraining complete!")
    print(f"Average reward (last 20 eps): {np.mean(rewards_history[-20:]):.2f}")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Reward plot saved to {PLOT_PATH} (also shown on screen)")

    return policy_net, device


def optimize_model(policy_net, target_net, optimizer, batch, gamma, device):
    states, actions, rewards, next_states, dones = batch
    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    q_values = policy_net(states).gather(1, actions).squeeze()
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        target = rewards + gamma * next_q * (1 - dones)

    loss = nn.MSELoss()(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def plot_rewards(rewards):
    plt.figure(figsize=(9, 4))
    plt.plot(rewards, label="Reward per episode")
    if len(rewards) >= 10:
        smooth = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        plt.plot(range(9, len(rewards)), smooth, label="10-ep avg", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("CartPole DQN training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.show()


def record_agent(model_path=MODEL_PATH, gif_path=GIF_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy_net = DQN(state_size, action_size).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    frames = []
    state, _ = env.reset()
    done = False
    while not done:
        frames.append(env.render())
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = policy_net(state_tensor).argmax(dim=1).item()
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
    env.close()

    imageio.mimsave(gif_path, frames, fps=30)
    print(f"Saved CartPole animation to {gif_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple DQN for CartPole (assignment-friendly).")
    parser.add_argument("--episodes", type=int, default=400, help="Training episodes (default 400).")
    parser.add_argument("--record", action="store_true", help="Record a GIF using the saved model.")
    args = parser.parse_args()

    if not args.record:
        train_dqn(episodes=args.episodes)
    else:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("No trained model found. Run training first.")
        record_agent()


if __name__ == "__main__":
    main()

