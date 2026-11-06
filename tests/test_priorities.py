import unittest

import numpy as np
from primoretarget.optimizer import Objective, PrioritizedOptimizer
from primoretarget.plugins import InteractionEEF, contiguous_windows


class PriorityTests(unittest.TestCase):
    def test_feet_then_eef_then_free_conflict(self):
        # P0 fixes first coordinate; P1 can solve second; free term cannot undo either.
        objectives = [
            Objective("foot", lambda q, t: q[:1] - 1, 0),
            Objective("eef", lambda q, t: q[:2] - [8, 2], 1),
            Objective("free", lambda q, t: q - [-9, -8, 3], None),
        ]
        result = PrioritizedOptimizer(objectives, iterations=50, step_size=0.5, smoothness=0).optimize(np.zeros((1, 3)))
        np.testing.assert_allclose(result.values[0], [1, 2, 3], atol=1e-4)

    def test_human_center_window_and_human_target(self):
        human = np.array([[0.20, 0, 0], [0.15, 0, 0], [0.14, 0.01, 0], [0.20, 0, 0]])
        plugin = InteractionEEF(human, np.zeros(3))
        np.testing.assert_array_equal(plugin.active, [False, True, True, False])
        self.assertEqual(contiguous_windows(plugin.active), [[1, 3]])
        objective = plugin.objective(lambda q, t: q)
        self.assertEqual(objective.priority, 1)
        np.testing.assert_array_equal(objective.residual(np.zeros(3), 0), np.zeros(3))
        np.testing.assert_allclose(objective.residual(np.zeros(3), 2), -human[2])

    def test_bounds_do_not_postclip_priority_solution(self):
        objectives = [
            Objective("foot", lambda q, t: np.array([q.sum() - 1]), 0),
            Objective("eef", lambda q, t: q[:1] - 3, 1),
        ]
        result = PrioritizedOptimizer(
            objectives, iterations=20, step_size=0.2, lower=np.zeros(2), upper=np.ones(2), smoothness=0
        ).optimize(np.array([[0.4, 0.6]]))
        self.assertLess(np.max(np.abs(result.values.sum(axis=1) - 1)), 1e-5)
        self.assertTrue(np.all(result.values >= 0))
        self.assertTrue(np.all(result.values <= 1 + 1e-8))

    def test_no_mixed_warmup(self):
        with self.assertRaises(ValueError):
            PrioritizedOptimizer([], warmup=2)

    def test_bound_can_move_inward(self):
        objective = Objective("eef", lambda q, t: q - 0.5, 1)
        result = PrioritizedOptimizer([objective], iterations=10, lower=[0.0], upper=[1.0], smoothness=0).optimize(
            np.array([[1.0]])
        )
        np.testing.assert_allclose(result.values, [[0.5]], atol=1e-5)

    def test_nonlinear_top_constraint_retraction(self):
        foot = Objective("foot", lambda q, t: np.array([q @ q - 1]), 0)
        eef = Objective("eef", lambda q, t: q - [0, 1], 1)
        result = PrioritizedOptimizer([foot, eef], iterations=30, smoothness=0).optimize(np.array([[1.0, 0.0]]))
        self.assertLess(abs(float(result.values[0] @ result.values[0]) - 1), 2e-6)
        np.testing.assert_allclose(result.values, [[0, 1]], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
