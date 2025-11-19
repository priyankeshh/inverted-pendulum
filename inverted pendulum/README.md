## Simple CartPole (Inverted Pendulum) DQN Assignment

This repo now contains one straightforward script you can point to for your reinforcement learning assignment. Running it:
- trains a DQN agent on Gymnasium’s `CartPole-v1` environment
- prints per-episode rewards right in the terminal so you can copy them into your report
- pops up a Matplotlib reward graph when training ends
- can optionally save a quick GIF of the agent balancing the pole (handy for slides)

### Files
- `cartpole_dqn.py` – the only script you need to show. It holds the model, replay buffer, training loop, plotting, and optional recording logic.
- `requirements.txt` – pip dependencies.
- `artifacts/` – created automatically when saving the trained model, reward plot, or GIF (feel free to delete if you want a cleaner folder before submission).

### How to Run (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --default-timeout=120 -r requirements.txt  # retry if your internet drops

# Train for 400 episodes (you’ll see rewards streaming in the terminal)
python cartpole_dqn.py --episodes 400
```
When training finishes:
- A Matplotlib window opens with the reward curve (screenshot it for your report).
- Files saved under `artifacts/`:
  - `cartpole_model.pth` (weights)
  - `cartpole_rewards.png` (same plot in case you close the window)

### Optional GIF
```powershell
python cartpole_dqn.py --record
```
This uses the last saved model (`artifacts/cartpole_model.pth`) and writes `artifacts/cartpole_agent.gif`. You can attach that to your submission if your instructor asks for visual proof.

### What to Mention in Your Write-up
1. Briefly explain the inverted pendulum problem and why balancing it is challenging.
2. Summarize how DQN works (neural network Q-approximation + replay buffer + target network).
3. Paste a small table of episode rewards (the script prints them).
4. Include the reward curve plot and note when the policy stabilizes.
5. Drop the GIF or describe how the pole behaves after training.

That’s it—one file, clean outputs, and easy to present as your own work.

