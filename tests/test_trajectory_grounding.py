import unittest

import mujoco
import numpy as np
from primoretarget.grounding import make_ground_plan
from primoretarget.optimizer import Objective
from primoretarget.trajectory import TrajectoryOptimizer


class TrajectoryTests(unittest.TestCase):
    def test_linear_hierarchy_preserves_particular_solution_before_trust_scaling(self):
        feet = Objective("feet", lambda q, t: q[:1] - 2, 0)
        eef = Objective("eef", lambda q, t: q[:1] + q[1:2] - 1, 1)
        solver = TrajectoryOptimizer([feet, eef], step_size=0.15)
        dx, basis = solver._hierarchy(np.zeros(2), 0, [[feet], [eef]], np.zeros(2, bool))
        # Trust scaling applies to the complete step, never separately per level.
        np.testing.assert_allclose(dx, [2, -1], atol=3e-5)
        self.assertEqual(basis.shape, (2, 0))

    def test_soft_collision_cannot_override_feet_or_eef(self):
        # Different target signs/magnitudes with the same solver configuration.
        for target in (-0.4, 0.3, 0.7):
            objectives = [
                Objective("feet", lambda q, t: q[:1], 0),
                Objective("eef", lambda q, t, target=target: q[1:2] - target if 6 <= t < 9 else np.zeros(1), 1),
                Objective("collision", lambda q, t, target=target: q[1:2] + target, 2, 100),
                Objective("posture", lambda q, t: q, 2),
            ]
            result = TrajectoryOptimizer(objectives, iterations=8, step_size=0.5).optimize(np.zeros((15, 2)))
            np.testing.assert_allclose(result.values[:, 0], 0, atol=1e-6)
            np.testing.assert_allclose(result.values[6:9, 1], target, atol=1e-5)

    def test_window_target_changes_both_sides_without_extending_mask(self):
        n = 61
        active = np.zeros(n, bool)
        active[29:32] = True
        objective = [
            Objective("feet", lambda q, t: q[:1], 0),
            Objective("eef", lambda q, t: q[1:2] - 1 if active[t] else np.zeros(1), 1),
            Objective("posture", lambda q, t: q, 2),
        ]
        x = (
            TrajectoryOptimizer(objective, iterations=8, step_size=0.5, velocity_weight=25, acceleration_weight=400)
            .optimize(np.zeros((n, 2)))
            .values
        )
        np.testing.assert_allclose(x[29:32, 1], 1, atol=1e-5)
        self.assertGreater(x[20, 1], 0.05)
        self.assertGreater(x[40, 1], 0.05)
        self.assertLess(np.max(np.abs(np.diff(x[:, 1]))), 0.15)
        np.testing.assert_allclose(x[:, 0], 0, atol=1e-8)
        np.testing.assert_array_equal(np.flatnonzero(active), [29, 30, 31])

    def test_posture_can_control_eef_orientation_null_space(self):
        n = 25
        reference = np.zeros((n, 3))
        reference[:, 2] = 0.2
        objective = [
            Objective("feet", lambda q, t: q[:1], 0),
            Objective("eef", lambda q, t: q[1:2] - 0.5 if 10 <= t < 15 else np.zeros(1), 1),
            Objective("posture", lambda q, t: q - reference[t], 2),
        ]
        x = TrajectoryOptimizer(objective, iterations=8, step_size=0.5).optimize(reference).values
        np.testing.assert_allclose(x[:, 2], 0.2, atol=1e-5)
        np.testing.assert_allclose(x[10:15, 1], 0.5, atol=1e-5)

    def test_ground_targets_use_sole_and_keep_swing(self):
        xml = """<mujoco><worldbody>
          <body name="left_ankle_roll_link"><geom name="left_foot1_collision" type="capsule" size=".01" fromto="-.05 0 -.025 .13 0 -.025"/></body>
          <body name="right_ankle_roll_link"><geom name="right_foot1_collision" type="capsule" size=".01" fromto="-.05 0 -.025 .13 0 -.025"/></body>
        </worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        n = 61
        pos = np.zeros((n, 2, 3))
        pos[:, :, 2] = 0.18
        rot = np.broadcast_to(np.eye(3), (n, 2, 3, 3)).copy()
        human = np.zeros((n, 2, 3))
        human[:, :, 2] = 0.07
        human[20:41, 0, 2] += 0.08 * np.sin(np.linspace(0, np.pi, 21))
        plan = make_ground_plan(pos, rot, human, model, 50)
        # FK ankle center is 18cm high, but the actual sole is 3.5cm below it.
        np.testing.assert_allclose(plan.positions[plan.support, 2], 0.035, atol=1e-7)
        self.assertGreater(plan.positions[:, 0, 2].max(), 0.1)
        self.assertTrue(np.all(plan.positions[:, :, 2] >= 0.035 - 1e-8))


if __name__ == "__main__":
    unittest.main()
