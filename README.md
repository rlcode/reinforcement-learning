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

**Atari** ([`3-atari/`](./3-atari))

10. DQN  — [`1-dqn.py`](./3-atari/1-dqn.py)
11. PPO  — [`2-ppo.py`](./3-atari/2-ppo.py)

## Benchmarks

Trained on a **MacBook Pro 14" (Apple M3, 8 GB unified memory)**, macOS 26.2, Python 3.11, PyTorch 2.11 with the MPS backend. CPU / GPU figures are read from Activity Monitor on the `python3.11` process after the run has stabilized (~5 min in); peak RAM is the process's real memory at its high-water mark. Final score is the mean per-game return over the last 20 episodes of training.

### Atari — Breakout (10M agent steps, `ALE/Breakout-v5` with sticky actions)

| Algorithm | Params | Train time | Final mean (per-game) | Peak RAM | CPU% | GPU% | W&B |
|-----------|--------|------------|-----------------------|----------|------|------|-----|
| DQN       | 1.69M  | ~9h        | 93.5 ± 9.6            | 5.27 GB  | ~60  | ~55  | [report](https://api.wandb.ai/links/rlcode/ljkn7ahp) |
| PPO       | 1.69M  | ~3.8h      | 261.9 ± 6.4           | 1.98 GB  | ~62  | ~55  | [report](https://api.wandb.ai/links/rlcode/jbdsbn6t) |

> Single seed per row, mean ± std over the final 20 logged episodes. `Params` counts only trainable network weights. `CPU%` is the single-process value reported by Activity Monitor (sum across cores, so >100% means multi-core use); `GPU%` is the same column for the Apple GPU. Sticky actions (`repeat_action_probability=0.25`) make absolute scores lower than the deterministic `*-v4` environments often cited in older papers.

### Atari — Montezuma's Revenge

Mac Studio (Apple M4 Max, 64 GB), `ALE/MontezumaRevenge-v5`, single seed. **Two protocols, not cross-comparable:** sticky-action RL vs deterministic restore-based search.

| Method | Protocol | Score (single seed) | Frames | Link |
|--------|----------|---------------------|--------|------|
| PPO + RND ([`1-ppo-rnd.py`](./4-atari-hard/1-ppo-rnd.py)) | sticky, RL policy | ~3,120 | 65M | [report](https://api.wandb.ai/links/rlcode/3j0nfk9s) |
| Go-Explore — exploration ([`2-go-explore.py`](./4-atari-hard/2-go-explore.py)) | deterministic search | 31,000 (replay-verified) | 500M | [run](https://wandb.ai/rlcode/rl-atari-hard-go-explore/runs/m6ox4l3m) |
| Go-Explore — robustification ([`3-robustify.py`](./4-atari-hard/3-robustify.py)) | sticky, RL policy | — (no from-reset score) | 5M | — |

> **RND** (Burda et al. 2018): first key ~327k steps with 512 envs (128 never scored in 50M — parallel breadth is the lever); plateaued above the PPO baseline 2497, below RND's 8152 (~30× more experience). **Exploration** (Ecoffet et al. 2019/2021): best end-of-episode trajectory from a knowledge-free cell archive, no NN — a search result, not an RL score (Nature ref 24,758). **Robustification** (backward algorithm, Salimans & Chen 2018): bootstraps with a first-key demo + 128 envs but the curriculum plateaus ~22% on one machine — no from-reset score, a scale ceiling vs the original hundreds–thousands of envs.

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

### Logging to Weights & Biases (Atari only)

Both Atari scripts (`1-dqn.py`, `2-ppo.py`) can stream training metrics to your own [Weights & Biases](https://wandb.ai/) account. One-time login, then pass `--wandb`:

```bash
uv run wandb login   # paste the API key from https://wandb.ai/authorize
cd 3-atari && uv run python 2-ppo.py --env breakout --wandb
cd 3-atari && uv run python 1-dqn.py --env breakout --wandb
```

Runs land in *your* `rl-atari-ppo` / `rl-atari-dqn` project — nothing is shared by default. Omit `--wandb` and the script runs without ever touching the network.

## Updates

Modernized from the 2017 original:

- **Framework**: Keras + TensorFlow 1.0 → PyTorch 2.11
- **Env**: gym 0.8 → gymnasium 1.2
- **Rendering**: tkinter → pygame (cross-platform with no system Tk)
- **Tooling**: `requirements.txt` → `pyproject.toml` + `uv`
- **Scope**: pruned to 9 core algorithms; dropped Monte Carlo / DDQN / A3C / Atari / mountaincar; added PPO
- **Layout**: flat `1-grid-world/3-sarsa.py` instead of nested `1-grid-world/4-sarsa/sarsa_agent.py`
- **Docs**: each algorithm file now opens with a paper citation and the core update equation
