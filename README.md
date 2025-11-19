# Reinforcement Learning: Inverted Pendulum (CartPole-v1) DQN

**Student:** [Your Name]  
**Course:** [Course Code/Name]  
**Date:** November 20, 2025

---

## Project Overview

This project implements a **Deep Q-Network (DQN)** to solve the CartPole-v1 environment from Gymnasium. The objective is to train a reinforcement learning agent to stabilize an inverted pendulum mounted on a moving cart—a classic control problem that demonstrates the effectiveness of neural network-based Q-learning.

### Problem Statement

The inverted pendulum problem requires balancing a pole attached by an un-actuated joint to a cart moving along a frictionless track. The agent must learn to:
- Move the cart left or right to prevent the pole from falling
- Maintain balance for as long as possible (max 500 steps)
- Generalize from experience to handle different initial states

This is challenging because:
1. The system is **unstable** at equilibrium (pole naturally falls)
2. Actions have **delayed consequences** (small movements now affect future stability)
3. The agent must learn **predictive control** from trial and error

---

## Implementation Details

### Algorithm: Deep Q-Network (DQN)

The implementation uses the following key components:

#### 1. Neural Network Q-Function Approximator
- **Architecture:** Fully connected neural network
  - Input layer: 4 neurons (cart position, cart velocity, pole angle, pole angular velocity)
  - Hidden layer 1: 128 neurons + ReLU activation
  - Hidden layer 2: 128 neurons + ReLU activation
  - Output layer: 2 neurons (Q-values for left/right actions)

#### 2. Experience Replay Buffer
- **Capacity:** 100,000 transitions
- **Prefill:** 5,000 random transitions before training begins
- **Purpose:** Breaks correlation between consecutive samples, stabilizes learning

#### 3. Target Network
- **Update frequency:** Every 1,000 steps (configurable)
- **Purpose:** Provides stable Q-value targets, prevents oscillation

#### 4. Exploration Strategy
- **Method:** ε-greedy policy
- **Initial ε:** 1.0 (100% random exploration)
- **Final ε:** 0.01 (1% exploration)
- **Decay:** Multiplicative decay factor of 0.997 per episode

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Episodes | 500 | Maximum training episodes |
| Batch Size | 128 | Mini-batch size for SGD |
| Learning Rate | 0.0005 | Adam optimizer learning rate |
| Gamma (γ) | 0.99 | Discount factor for future rewards |
| Replay Buffer | 100,000 | Maximum stored transitions |
| Target Update | 1,000 steps | Frequency of target network sync |
| Early Stop Threshold | 475.0 | Average reward over 50 episodes |

---

## File Structure

```
inverted pendulum/
├── cartpole_final.py      # Main DQN implementation
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── artifacts/            # Generated during training (not committed)
    ├── models/
    │   └── dqn_cartpole.pth
    ├── cartpole_rewards.png
    ├── cartpole_agent.gif
    └── *.mp4
```

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- PowerShell (Windows) or Bash (Linux/macOS)

### Installation Steps

```powershell
# Navigate to project directory
cd "inverted pendulum"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\activate  # Windows PowerShell
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install --default-timeout=120 -r requirements.txt
```

---

## Usage

### 1. Train the Agent (Default Settings)

```powershell
python cartpole_final.py
```

**Output:**
- Console: Episode-by-episode rewards and training progress
- `artifacts/models/dqn_cartpole.pth`: Saved model weights
- `artifacts/cartpole_rewards.png`: Reward curve plot

### 2. Train with Custom Hyperparameters

```powershell
# Example: Longer training with different epsilon decay
python cartpole_final.py --episodes 600 --eps-decay 0.995 --prefill 8000

# Example: Episode-based target updates instead of step-based
python cartpole_final.py --target-update-mode episodes --target-update-freq 10

# Example: Disable early stopping
python cartpole_final.py --early-stop-avg -1
```

### 3. Evaluate Trained Model

```powershell
# Run 5 evaluation episodes without exploration
python cartpole_final.py --episodes 0 --evaluate --eval-episodes 5
```

### 4. Record Agent Performance

```powershell
# Generate GIF animation
python cartpole_final.py --episodes 0 --record-gif

# Generate MP4 video (higher quality)
python cartpole_final.py --episodes 0 --record-mp4 --mp4-name demo.mp4
```

### 5. Full Training + Evaluation + Recording Pipeline

```powershell
python cartpole_final.py --episodes 500 --evaluate --eval-episodes 3 --record-gif
```

---

## Results

### Training Performance

*Expected behavior (results may vary due to randomness):*

- **Episodes 1-50:** Agent learns basic movements, rewards fluctuate (10-100 range)
- **Episodes 50-150:** Policy improves, rewards increase (100-300 range)
- **Episodes 150-300:** Agent approaches optimal policy (300-500 range)
- **Episodes 300+:** Consistently achieves maximum reward (500 steps)

**Success Criterion:** CartPole-v1 is considered "solved" when the agent achieves an average reward ≥ 195 over 100 consecutive episodes. With our configuration, the agent typically reaches an average of 475+ over 50 episodes.

### Visualizations

1. **Reward Curve:** Shows episode rewards with 20-episode moving average smoothing
2. **GIF/Video:** Demonstrates trained agent balancing the pole successfully

---

## Key Learning Outcomes

This project demonstrates:

1. **Function Approximation:** Using neural networks to estimate Q-values in continuous state spaces
2. **Experience Replay:** Breaking temporal correlation to improve sample efficiency
3. **Target Networks:** Stabilizing training by decoupling Q-value estimation from updates
4. **Exploration vs Exploitation:** Balancing random exploration with greedy policy exploitation
5. **Hyperparameter Tuning:** Impact of learning rate, batch size, replay buffer size, and epsilon decay

---

## Technical Notes

### Why DQN Works for CartPole

- **State space:** Continuous 4D space (position, velocity, angle, angular velocity)
- **Action space:** Discrete (left or right)
- **Episode length:** Limited to 500 steps (prevents infinite loops)
- **Reward structure:** +1 per timestep (encourages longer episodes)

The DQN algorithm is well-suited because:
- Neural network handles continuous states
- Q-learning works naturally with discrete actions
- Experience replay improves sample efficiency in this low-dimensional environment

### Potential Extensions

- **Double DQN:** Reduce overestimation bias in Q-value updates
- **Dueling DQN:** Separate value and advantage streams
- **Prioritized Experience Replay:** Sample important transitions more frequently
- **Rainbow DQN:** Combine multiple improvements (Double, Dueling, Noisy Nets, etc.)
- **Transfer to harder tasks:** MountainCar, LunarLander, Atari games

---

## Command-Line Arguments Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--episodes` | int | 500 | Training episodes (0 to skip training) |
| `--max-steps` | int | 500 | Max steps per episode |
| `--batch-size` | int | 128 | Mini-batch size |
| `--buffer-size` | int | 100000 | Replay buffer capacity |
| `--prefill` | int | 5000 | Random transitions to prefill buffer |
| `--gamma` | float | 0.99 | Discount factor |
| `--lr` | float | 0.0005 | Learning rate |
| `--eps-start` | float | 1.0 | Initial epsilon |
| `--eps-end` | float | 0.01 | Minimum epsilon |
| `--eps-decay` | float | 0.997 | Epsilon decay per episode |
| `--target-update-freq` | int | 1000 | Target network update frequency |
| `--target-update-mode` | str | steps | Update by "steps" or "episodes" |
| `--seed` | int | 42 | Random seed |
| `--early-stop-avg` | float | 475.0 | Avg reward threshold for early stop |
| `--early-stop-window` | int | 50 | Window size for averaging |
| `--evaluate` | flag | False | Run evaluation mode |
| `--eval-episodes` | int | 5 | Episodes to evaluate |
| `--record-gif` | flag | False | Record GIF animation |
| `--record-mp4` | flag | False | Record MP4 video |
| `--mp4-name` | str | cartpole_run.mp4 | MP4 filename |

---

## References

- Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Gymnasium Documentation: https://gymnasium.farama.org/
- PyTorch Documentation: https://pytorch.org/docs/

---

## Acknowledgments

This implementation was developed as part of a reinforcement learning course assignment, demonstrating the practical application of DQN to a classic control problem. The code is self-contained and suitable for educational purposes.

---

**For questions or clarifications, please contact [your email/contact information].**

