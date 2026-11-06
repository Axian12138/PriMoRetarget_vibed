"""Optional task-specific objectives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .optimizer import Objective


@dataclass
class InteractionEEF:
    """Add EEF target tracking only during source hand/object proximity."""
    source_hand_xyz: np.ndarray
    target_xyz: np.ndarray
    proximity: float = 0.12
    priority: int = 1
    weight: float = 1.0

    def objective(self, eef_position):
        active = np.linalg.norm(self.source_hand_xyz - self.target_xyz, axis=-1) < self.proximity
        def residual(q, frame):
            return self.weight ** 0.5 * (np.asarray(eef_position(q, frame)) - self.target_xyz[frame]) if active[frame] else np.zeros(3)
        return Objective("interaction_eef", residual, self.priority)
