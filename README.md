<p align="center"><img width="90%" src="images/Reinforcement-Learning.png"></p>

From the basics to deep reinforcement learning, this repo provides easy-to-read code examples. One file for each algorithm. Please feel free to create a Pull Request, or open an issue!

## Algorithms

**Grid World** ([`1-grid-world/`](./1-grid-world))

1. Policy Iteration  — [`1-policy_iteration.py`](./1-grid-world/1-policy_iteration.py)
2. Value Iteration   — [`2-value_iteration.py`](./1-grid-world/2-value_iteration.py)
3. SARSA             — [`3-sarsa.py`](./1-grid-world/3-sarsa.py)
4. Q-Learning        — [`4-q_learning.py`](./1-grid-world/4-q_learning.py)
5. Deep SARSA        — [`5-deep_sarsa.py`](./1-grid-world/5-deep_sarsa.py)
6. REINFORCE         — [`6-reinforce.py`](./1-grid-world/6-reinforce.py)

**CartPole** ([`2-cartpole/`](./2-cartpole))

7. DQN  — [`1-dqn.py`](./2-cartpole/1-dqn.py)
8. A2C  — [`2-a2c.py`](./2-cartpole/2-a2c.py)
9. PPO  — [`3-ppo.py`](./2-cartpole/3-ppo.py)

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo>
cd reinforcement-learning
uv sync
```

## Running

```bash
# Grid World
cd 1-grid-world && uv run python 3-sarsa.py

# CartPole — train
cd 2-cartpole && uv run python 1-dqn.py

# CartPole — watch training (slower)
cd 2-cartpole && uv run python 1-dqn.py --render

# CartPole — replay a trained checkpoint
cd 2-cartpole && uv run python 1-dqn.py --test
```

## Updates

Modernized from the 2017 original:

- **Framework**: Keras + TensorFlow 1.0 → PyTorch 2.11
- **Env**: gym 0.8 → gymnasium 1.2
- **Rendering**: tkinter → pygame (cross-platform with no system Tk)
- **Tooling**: `requirements.txt` → `pyproject.toml` + `uv`
- **Scope**: pruned to 9 core algorithms; dropped Monte Carlo / DDQN / A3C / Atari / mountaincar; added PPO
- **Layout**: flat `1-grid-world/3-sarsa.py` instead of nested `1-grid-world/4-sarsa/sarsa_agent.py`
- **Docs**: each algorithm file now opens with a paper citation and the core update equation
