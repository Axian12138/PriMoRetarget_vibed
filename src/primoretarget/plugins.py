"""Human-derived interaction windows and their prioritized EEF objectives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .optimizer import Objective


def contiguous_windows(active):
    """Half-open source-frame intervals; no padding, hysteresis or gap filling."""
    edges = np.diff(np.r_[False, np.asarray(active, dtype=bool), False].astype(int))
    return [[int(a), int(b)] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)]


@dataclass
class InteractionEEF:
    """Object-center proximity selects frames; source wrist is the EEF target.

    The target exists at every frame. Detection changes only its smooth weight;
    priority 1 is a weighted tracking objective below hard priority-0 feet.
    """

    source_hand_xyz: np.ndarray
    target_xyz: np.ndarray
    proximity: float = 0.15
    priority: int = 1
    weight: float = 1.0
    fps: float = 50.0
    window_weight: float = 1000.0
    transition_seconds: float = 0.3

    @property
    def weights(self):
        """Positive baseline plus a C1 Gaussian shoulder around each window.

        Fixed seconds, not take-specific frame padding. The detection mask is
        unchanged; the envelope is one inside each interval and decays outside.
        """
        seconds = np.arange(len(self.source_hand_xyz)) / self.fps
        envelope = np.zeros(len(seconds))
        for start, stop in contiguous_windows(self.active):
            distance = np.maximum(np.maximum(start / self.fps - seconds, seconds - (stop - 1) / self.fps), 0.0)
            envelope = np.maximum(envelope, np.exp(-0.5 * (distance / self.transition_seconds) ** 2))
        return self.weight + (self.window_weight - self.weight) * envelope

    @property
    def distances(self):
        return np.linalg.norm(self.source_hand_xyz - self.target_xyz, axis=-1)

    @property
    def active(self):
        return self.distances <= self.proximity

    def objective(self, eef_position, eef_jacobian=None, *, name="interaction_eef"):
        scale = np.sqrt(self.weights)

        def residual(q, frame):
            return scale[frame] * (np.asarray(eef_position(q, frame)) - self.source_hand_xyz[frame])

        def jacobian(q, frame):
            return scale[frame] * eef_jacobian(q, frame)

        return Objective(name, residual, self.priority, 1.0, jacobian if eef_jacobian is not None else None)
