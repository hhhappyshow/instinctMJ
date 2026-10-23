"""Semantic checks for miss encoding, no progress cycling, and terminal accounting."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from instinct_mj.perception.forward_scan import forward_scan
from instinct_mj.tasks.navigation import mdp


class NavigationContracts(unittest.TestCase):
    def test_misses_are_finite_far_distances(self):
        data = SimpleNamespace(distances=torch.tensor([[-1., float('nan'), float('inf'), 3.1, 0., 1.5, 3.]]), hit_pos_w=torch.zeros(1, 7, 3))
        scan = forward_scan(data)
        self.assertEqual(scan['ray_valid_mask'].tolist(), [[False, False, False, False, True, True, True]])
        self.assertTrue(torch.equal(scan['ray_observation'], torch.tensor([[1., 1., 1., 1., 0., .5, 1.]])))

    def test_backtracking_cannot_repeat_progress_reward(self):
        term = SimpleNamespace(best_distance=torch.tensor([4.]), progress=torch.zeros(1), reward_step=-1)
        env = SimpleNamespace(step_dt=.1, common_step_counter=0, extras={}, cfg=SimpleNamespace(rewards={'progress': SimpleNamespace(weight=2.)}))
        total = 0.
        with patch.object(mdp, 'controller', return_value=term):
            for step, distance in enumerate([3., 4., 3., 4., 3., 2.]):
                term.distance = lambda distance=distance: torch.tensor([distance])
                env.common_step_counter = step
                value = mdp.reward(env, 'progress')
                total += value.item()*env.step_dt
                self.assertEqual(value.item(), mdp.reward(env, 'progress').item())
        self.assertAlmostEqual(total, 2.)

    def test_terminal_causes_are_exclusive(self):
        env = SimpleNamespace(episode_length_buf=torch.tensor([200]*4), max_episode_length=200)
        term = SimpleNamespace(distance=lambda: torch.tensor([0., 0., 0., 4.]))
        with patch.object(mdp, 'controller', return_value=term), patch.object(mdp, 'collision', return_value=torch.tensor([True, False, False, False])), patch.object(mdp, 'fall', return_value=torch.tensor([True, True, False, False])):
            matrix = torch.stack([mdp.outcome(env, k) for k in ('collision','fall','success','timeout')])
        self.assertTrue(torch.equal(matrix, torch.eye(4, dtype=torch.bool)))


if __name__ == '__main__':
    unittest.main()
