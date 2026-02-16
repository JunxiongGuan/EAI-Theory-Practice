# EAI-Theory-Practice
Embodied AI Theory And Practice such as course tutorial, comment on papers, and demo about RL, DiT, VLA, even humanoid control.

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
