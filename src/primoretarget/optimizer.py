"""Frame-wise hierarchical least squares in higher-priority null spaces."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize._numdiff import approx_derivative

Residual = Callable[[np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class Objective:
    name: str
    residual: Residual
    priority: int | None
    weight: float = 1.0
    jacobian: Residual | None = None


@dataclass(frozen=True)
class OptimizationResult:
    values: np.ndarray
    losses: dict[str, float]


class PrioritizedOptimizer:
    """Ascending priorities; all lower terms stay in the preceding null space.

    Damped SVD solves each level's remaining linearized residual. Exact numerical
    null spaces protect higher levels. Lexicographic line search checks nonlinear
    losses; bounds constrain the step before acceptance. No mixed warmup.
    """

    def __init__(
        self,
        objectives: Sequence[Objective],
        *,
        damping=1e-5,
        step_size=0.15,
        iterations=30,
        warmup=0,
        smoothness=0.01,
        smoothness_priority=None,
        lower=None,
        upper=None,
        progress=None,
    ):
        if damping <= 0 or step_size <= 0 or iterations < 0 or warmup != 0:
            raise ValueError("invalid parameters; prioritized optimization requires warmup=0")
        self.objectives = tuple(objectives)
        self.damping, self.step_size = damping, step_size
        self.iterations, self.smoothness = iterations, smoothness
        self.smoothness_priority = smoothness_priority
        self.lower, self.upper, self.progress = lower, upper, progress

    @staticmethod
    def _loss(objective, x, frame):
        r = np.asarray(objective.residual(x, frame), dtype=float).ravel()
        return 0.5 * objective.weight * float(r @ r)

    def _terms(self, objective, x, frame):
        def fun(z):
            return np.asarray(objective.residual(z, frame), dtype=float).ravel()

        r = fun(x)
        j = (
            approx_derivative(fun, x, method="3-point")
            if objective.jacobian is None
            else np.asarray(objective.jacobian(x, frame), dtype=float)
        )
        scale = np.sqrt(objective.weight)
        return r * scale, np.atleast_2d(j) * scale

    def optimize(self, initial):
        x = np.asarray(initial, dtype=float).copy()
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("initial values must be finite [frames, dofs]")
        lower = np.broadcast_to(-np.inf if self.lower is None else self.lower, x.shape[1:])
        upper = np.broadcast_to(np.inf if self.upper is None else self.upper, x.shape[1:])
        if np.any(x < lower - 1e-8) or np.any(x > upper + 1e-8):
            raise ValueError("initial motion violates hard joint bounds")
        levels = sorted({o.priority for o in self.objectives if o.priority is not None})
        groups = [[o for o in self.objectives if o.priority == level] for level in levels]
        free = [o for o in self.objectives if o.priority is None]
        if free:
            groups.append(free)
        for iteration in range(self.iterations):
            updated = x.copy()
            for t in range(len(x)):
                dx = np.zeros(x.shape[1])
                smooth_target = (x[max(0, t - 1)] + x[min(len(x) - 1, t + 1)]) / 2
                frame_groups = [list(group) for group in groups]
                if self.smoothness and len(x) > 1:
                    temporal = Objective(
                        "temporal_smoothing",
                        lambda q, t, target=smooth_target: q - target,
                        None,
                        self.smoothness,
                        lambda q, t: np.eye(len(q)),
                    )
                    if frame_groups:
                        index = levels.index(self.smoothness_priority) if self.smoothness_priority in levels else -1
                        frame_groups[index].append(temporal)
                    else:
                        frame_groups = [[temporal]]
                fixed = np.zeros(x.shape[1], dtype=bool)
                for _ in range(x.shape[1] + 1):
                    dx = self._direction(frame_groups, x[t], t, fixed)
                    blocked = ((x[t] <= lower + 1e-8) & (dx < 0)) | ((x[t] >= upper - 1e-8) & (dx > 0))
                    if not np.any(blocked & ~fixed):
                        break
                    fixed |= blocked
                scale = min(1.0, self.step_size / max(1e-12, np.max(np.abs(dx))))
                positive, negative = dx > 1e-12, dx < -1e-12
                if positive.any():
                    scale = min(scale, float(np.min((upper[positive] - x[t, positive]) / dx[positive])))
                if negative.any():
                    scale = min(scale, float(np.min((lower[negative] - x[t, negative]) / dx[negative])))
                before = [sum(self._loss(o, x[t], t) for o in group) for group in frame_groups]
                for _ in range(12):
                    candidate = x[t] + max(0.0, scale) * dx
                    # Retract the finite step onto the top-level nonlinear
                    # constraint manifold before judging lower-priority losses.
                    for _ in range(6):
                        top_loss = sum(self._loss(o, candidate, t) for o in frame_groups[0])
                        if top_loss <= before[0] + 1e-12:
                            break
                        correction = self._direction(frame_groups[:1], candidate, t, np.zeros(len(candidate), bool))
                        candidate += correction
                    if np.any(candidate < lower - 1e-9) or np.any(candidate > upper + 1e-9):
                        scale *= 0.5
                        continue
                    after = [sum(self._loss(o, candidate, t) for o in group) for group in frame_groups]
                    accepted = True
                    for old, new in zip(before, after, strict=True):
                        tolerance = 1e-10 + 1e-6 * old
                        if new > old + tolerance:
                            accepted = False
                            break
                        if new < old - tolerance:
                            break
                    if accepted:
                        updated[t] = candidate
                        break
                    scale *= 0.5
            change = float(np.max(np.abs(updated - x)))
            x = updated
            if self.progress:
                self.progress(iteration + 1, change)
            if change < 1e-7:
                break
        losses = {o.name: sum(self._loss(o, row, t) for t, row in enumerate(x)) / len(x) for o in self.objectives}
        return OptimizationResult(x, losses)

    def _direction(self, groups, point, frame, fixed):
        dx = np.zeros(len(point))
        basis = np.eye(len(point))[:, ~fixed]
        for group in groups:
            terms = [self._terms(o, point, frame) for o in group]
            r = np.concatenate([v[0] for v in terms])
            j = np.concatenate([v[1] for v in terms], axis=0)
            a = j @ basis
            if not a.size:
                continue
            u, s, vt = np.linalg.svd(a, full_matrices=True)
            rank = int(np.sum(s > max(1e-9, (s[0] if len(s) else 0) * 1e-7)))
            if rank:
                dx += (
                    basis
                    @ vt[:rank].T
                    @ ((s[:rank] / (s[:rank] ** 2 + self.damping)) * (u[:, :rank].T @ (-r - j @ dx)))
                )
            basis = basis @ vt[rank:].T
        return dx
