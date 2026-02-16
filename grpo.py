"""Minimal GRPO (Group Relative Policy Optimization) implementation.

This module provides:
1. Group-relative advantage computation.
2. A tiny discrete policy trained with a PPO-style clipped GRPO objective.

It is designed for educational use in this repository.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple


EPS = 1e-8


def softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def group_relative_advantages(rewards: Sequence[float]) -> List[float]:
    """Compute normalized group-relative advantages.

    For a group of rewards r_i, GRPO commonly normalizes by group statistics:
        A_i = (r_i - mean(r)) / (std(r) + eps)
    """
    if len(rewards) == 0:
        return []
    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = math.sqrt(var_r)
    return [(r - mean_r) / (std_r + EPS) for r in rewards]


@dataclass
class Sample:
    state: int
    action: int
    old_logp: float
    reward: float
    advantage: float


class DiscretePolicy:
    """A tiny tabular-softmax policy for discrete state/action spaces."""

    def __init__(self, num_states: int, num_actions: int, seed: int = 42):
        random.seed(seed)
        self.num_states = num_states
        self.num_actions = num_actions
        self.logits = [
            [random.uniform(-0.01, 0.01) for _ in range(num_actions)]
            for _ in range(num_states)
        ]

    def probs(self, state: int) -> List[float]:
        return softmax(self.logits[state])

    def sample_action(self, state: int) -> Tuple[int, float]:
        probs = self.probs(state)
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                return i, math.log(max(p, EPS))
        # numeric fallback
        return len(probs) - 1, math.log(max(probs[-1], EPS))

    def log_prob(self, state: int, action: int) -> float:
        return math.log(max(self.probs(state)[action], EPS))

    def grpo_update(
        self,
        batch: Sequence[Sample],
        lr: float = 0.05,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
    ) -> float:
        """One gradient-ascent step with a clipped GRPO surrogate.

        Surrogate:
            L = E[min(r_t A_t, clip(r_t, 1-e, 1+e) A_t)] + c_ent * H(pi)
        where r_t = pi(a|s)/pi_old(a|s)
        """
        # accumulate gradients for logits[state][action]
        grad = [[0.0 for _ in range(self.num_actions)] for _ in range(self.num_states)]
        objective = 0.0

        for item in batch:
            probs = self.probs(item.state)
            new_logp = math.log(max(probs[item.action], EPS))
            ratio = math.exp(new_logp - item.old_logp)

            unclipped = ratio * item.advantage
            clipped_ratio = min(max(ratio, 1.0 - clip_eps), 1.0 + clip_eps)
            clipped_obj = clipped_ratio * item.advantage

            use_unclipped = unclipped <= clipped_obj
            weight = ratio if use_unclipped else clipped_ratio
            objective += min(unclipped, clipped_obj)

            # policy gradient for log-softmax:
            # d log pi(a|s) / d logits[k] = 1[a=k] - pi(k|s)
            for k in range(self.num_actions):
                indicator = 1.0 if k == item.action else 0.0
                grad[item.state][k] += weight * item.advantage * (indicator - probs[k])

            # entropy bonus gradient (approximate ascent on H)
            # H = -sum p log p ; dH/dlogit_k = -p_k (log p_k + H)
            h = -sum(p * math.log(max(p, EPS)) for p in probs)
            for k in range(self.num_actions):
                grad[item.state][k] += entropy_coef * (-probs[k] * (math.log(max(probs[k], EPS)) + h))

        n = max(len(batch), 1)
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.logits[s][a] += lr * grad[s][a] / n

        return objective / n


class ContextualBanditEnv:
    """Simple environment for demonstrating GRPO.

    - State sampled uniformly from [0, num_states)
    - Correct action is state % num_actions
    - Reward is 1 for correct action else 0
    """

    def __init__(self, num_states: int, num_actions: int, seed: int = 7):
        self.num_states = num_states
        self.num_actions = num_actions
        random.seed(seed)

    def sample_state(self) -> int:
        return random.randrange(self.num_states)

    def reward(self, state: int, action: int) -> float:
        return 1.0 if action == (state % self.num_actions) else 0.0


def collect_grouped_samples(
    env: ContextualBanditEnv,
    policy: DiscretePolicy,
    group_size: int,
    groups_per_epoch: int,
) -> List[Sample]:
    samples: List[Sample] = []
    for _ in range(groups_per_epoch):
        state = env.sample_state()
        group_rewards = []
        group_actions = []
        group_old_logps = []

        for _ in range(group_size):
            action, old_logp = policy.sample_action(state)
            r = env.reward(state, action)
            group_actions.append(action)
            group_old_logps.append(old_logp)
            group_rewards.append(r)

        advs = group_relative_advantages(group_rewards)
        for action, old_logp, reward, adv in zip(group_actions, group_old_logps, group_rewards, advs):
            samples.append(
                Sample(
                    state=state,
                    action=action,
                    old_logp=old_logp,
                    reward=reward,
                    advantage=adv,
                )
            )
    return samples


def train_grpo(
    epochs: int = 120,
    group_size: int = 6,
    groups_per_epoch: int = 48,
    num_states: int = 10,
    num_actions: int = 3,
    seed: int = 0,
) -> Tuple[DiscretePolicy, List[float]]:
    env = ContextualBanditEnv(num_states=num_states, num_actions=num_actions, seed=seed)
    policy = DiscretePolicy(num_states=num_states, num_actions=num_actions, seed=seed)

    avg_rewards: List[float] = []
    for _ in range(epochs):
        batch = collect_grouped_samples(env, policy, group_size=group_size, groups_per_epoch=groups_per_epoch)
        policy.grpo_update(batch)
        avg_rewards.append(sum(s.reward for s in batch) / max(1, len(batch)))

    return policy, avg_rewards


if __name__ == "__main__":
    policy, rewards = train_grpo()
    print(f"initial avg reward: {rewards[0]:.3f}")
    print(f"final avg reward:   {rewards[-1]:.3f}")
