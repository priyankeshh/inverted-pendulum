# dqn_cartpole.py (Gymnasium compatible)
import gymnasium as gym
import math
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time

# --- Hyperparameters ---
ENV_NAME = "CartPole-v1"
SEED = 42
GAMMA = 0.99
LR = 5e-4
BATCH_SIZE = 128
BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 5000
TARGET_UPDATE_FREQ = 1000  # steps
MAX_EPISODES = 700
MAX_STEPS = 500
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.997
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.vstack(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.vstack(next_states), np.array(dones, dtype=np.uint8))
    def __len__(self):
        return len(self.buffer)

# --- Q-network ---
class QNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x):
        return self.net(x)

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, env):
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n
        self.q = QNetwork(obs_size, n_actions).to(DEVICE)
        self.q_target = QNetwork(obs_size, n_actions).to(DEVICE)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=LR)
        self.replay = ReplayBuffer(BUFFER_SIZE)
        self.n_actions = n_actions
        self.steps_done = 0

    def select_action(self, state, eps):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        else:
            state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            q_vals = self.q(state_v)
            return int(torch.argmax(q_vals, dim=1).item())

    def optimize(self):
        if len(self.replay) < BATCH_SIZE:
            return None
        states, actions, rewards, next_states, dones = self.replay.sample(BATCH_SIZE)

        states_v = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        next_states_v = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
        actions_v = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards_v = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        dones_v = torch.tensor(dones, dtype=torch.uint8).to(DEVICE)

        q_vals = self.q(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            q_next = self.q_target(next_states_v).max(1)[0]
            q_target = rewards_v + GAMMA * q_next * (1 - dones_v)

        loss = nn.functional.mse_loss(q_vals, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.q_target.load_state_dict(self.q.state_dict())

# --- Training loop ---
def train():
    env = gym.make(ENV_NAME)
    env.reset(seed=SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    agent = DQNAgent(env)

    # Fill replay with some random transitions
    state, _ = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()
        next_state, reward, done, truncated, _ = env.step(action)
        finished = done or truncated
        agent.replay.push(state, action, reward, next_state, finished)
        state = next_state if not finished else env.reset()[0]

    episode_rewards = []
    losses = []
    eps = EPS_START
    total_steps = 0
    best_avg = -float('inf')

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(MAX_STEPS):
            action = agent.select_action(state, eps)
            next_state, reward, done, truncated, _ = env.step(action)
            finished = done or truncated
            agent.replay.push(state, action, reward, next_state, finished)
            loss = agent.optimize()
            if loss is not None:
                losses.append(loss)
            state = next_state
            ep_reward += reward
            total_steps += 1

            if total_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target()

            if finished:
                break

        episode_rewards.append(ep_reward)
        eps = max(EPS_END, eps * EPS_DECAY)

        # Logging
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}, Reward: {ep_reward:.1f}, Avg(50): {avg_reward:.2f}, Eps: {eps:.3f}")
            if avg_reward > best_avg:
                best_avg = avg_reward
                torch.save(agent.q.state_dict(), os.path.join(MODEL_DIR, "dqn_cartpole.pth"))

        if len(episode_rewards) >= 50 and np.mean(episode_rewards[-50:]) >= 475:
            print(f"Solved in {episode} episodes!")
            break

    env.close()

    # Save training plots
    plt.figure(figsize=(10,4))
    plt.plot(episode_rewards)
    plt.title("Episode rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.tight_layout()
    plt.savefig("rewards.png")
    print("Training finished. Plots saved: rewards.png")
    return episode_rewards, losses

# --- Evaluation / play with trained policy ---
def evaluate(n_episodes=5, render=False):
    env = gym.make(ENV_NAME, render_mode="human" if render else None)
    agent = DQNAgent(env)
    model_path = os.path.join(MODEL_DIR, "dqn_cartpole.pth")
    if not os.path.exists(model_path):
        print("No saved model found. Train first.")
        return
    agent.q.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.q.eval()

    returns = []
    for ep in range(n_episodes):
        state, _ = env.reset()
        total = 0
        done = False
        truncated = False
        while not (done or truncated):
            if render:
                env.render()
            action = agent.select_action(state, eps=0.0)
            state, reward, done, truncated, _ = env.step(action)
            total += reward
        returns.append(total)
        print(f"Eval episode {ep+1}: return {total}")
    env.close()
    print(f"Average return: {np.mean(returns)}")
    return returns

# --- Record a trained agent using imageio (recommended) ---
def record_video(filename="dqn_cartpole_video.mp4"):
    import imageio         # pip install imageio imageio-ffmpeg
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    agent = DQNAgent(env)
    model_path = os.path.join(MODEL_DIR, "dqn_cartpole.pth")
    if not os.path.exists(model_path):
        print("No saved model found. Train first.")
        return

    agent.q.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.q.eval()

    frames = []
    state, _ = env.reset()
    done = False
    truncated = False

    # collect frames
    while not (done or truncated):
        frame = env.render()   # should return an RGB array in Gymnasium
        if isinstance(frame, tuple):  # just in case some env returns (frame, info)
            frame = frame[0]
        frames.append(frame)
        action = agent.select_action(state, eps=0.0)
        state, reward, done, truncated, _ = env.step(action)

    env.close()

    # write video with imageio (uses ffmpeg backend provided by imageio-ffmpeg)
    writer = imageio.get_writer(filename, fps=30, codec='libx264', quality=8)
    for f in frames:
        # ensure uint8
        if f.dtype != 'uint8':
            f = (255 * np.clip(f, 0, 1)).astype('uint8')
        writer.append_data(f)
    writer.close()
    print(f"🎥 Video saved as {filename}")



if __name__ == "__main__":
    start = time.time()
    rewards, losses = train()
    print("Training took {:.2f} seconds".format(time.time() - start))
    evaluate(n_episodes=5, render=False)

record_video()