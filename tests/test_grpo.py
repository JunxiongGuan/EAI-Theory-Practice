import unittest

from grpo import (
    DiscretePolicy,
    evaluate_policy,
    group_relative_advantages,
    train_grpo,
)


class TestGRPO(unittest.TestCase):
    def test_group_relative_advantages_sum_near_zero(self):
        adv = group_relative_advantages([1.0, 0.0, 1.0, 0.0])
        self.assertAlmostEqual(sum(adv), 0.0, places=6)

    def test_group_relative_advantages_empty(self):
        self.assertEqual(group_relative_advantages([]), [])

    def test_training_is_reproducible_with_seed(self):
        _, rewards1 = train_grpo(epochs=20, groups_per_epoch=20, seed=11)
        _, rewards2 = train_grpo(epochs=20, groups_per_epoch=20, seed=11)
        self.assertEqual(len(rewards1), len(rewards2))
        for a, b in zip(rewards1, rewards2):
            self.assertAlmostEqual(a, b, places=10)

    def test_policy_improves_after_training(self):
        num_states = 10
        num_actions = 3
        untrained = DiscretePolicy(num_states=num_states, num_actions=num_actions, seed=5)
        baseline = evaluate_policy(
            untrained, num_states=num_states, num_actions=num_actions, episodes=400, seed=202
        )

        trained, _ = train_grpo(
            epochs=80,
            group_size=6,
            groups_per_epoch=40,
            num_states=num_states,
            num_actions=num_actions,
            seed=5,
        )
        improved = evaluate_policy(
            trained, num_states=num_states, num_actions=num_actions, episodes=400, seed=202
        )
        self.assertGreater(improved, baseline)
        self.assertGreater(improved, 0.45)


if __name__ == "__main__":
    unittest.main()
