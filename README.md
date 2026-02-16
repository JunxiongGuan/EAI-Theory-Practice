# EAI-Theory-Practice
Embodied AI Theory and Practice repository with tutorials, paper notes/comments, and practical algorithm demos for RL, DiT, VLA, and humanoid control.

## GRPO demo
This repository now includes a minimal educational implementation of **GRPO (Group Relative Policy Optimization)** in `grpo.py`.

### What is implemented
- Group-relative advantage normalization: `A_i = (r_i - mean(r_group)) / (std(r_group) + eps)`
- PPO-style clipped objective with policy ratio clipping
- Entropy bonus for exploration
- A simple contextual bandit environment for quick verification
- Reproducible random seeds (environment and policy maintain independent RNG)

### Run
```bash
python grpo.py
```

### Tests
```bash
python -m unittest discover -s tests -v
```

Expected output should show improved training/evaluation rewards and passing tests.
