"""Frame-wise prioritized optimization following PriMoRetarget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize._numdiff import approx_derivative


Residual = Callable[[np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class Objective:
    name: str
    residual: Residual
    priority: int | None
    weight: float = 1.0


@dataclass(frozen=True)
class OptimizationResult:
    values: np.ndarray
    losses: dict[str, float]


class PrioritizedOptimizer:
    """Optimize independent frame variables with prioritized error-vector projection.

    Objectives with integer priorities are projected in ascending order. Objectives
    with priority=None are free terms and are added directly to the update.
    """

    def __init__(self, objectives: Sequence[Objective], *, damping: float = 1e-3,
                 step_size: float = 0.05, iterations: int = 20, warmup: int = 3,
                 smoothness: float = 0.0):
        if damping <= 0 or step_size <= 0 or iterations < 0 or warmup < 0:
            raise ValueError("invalid optimizer parameters")
        self.objectives = tuple(objectives)
        self.damping, self.step_size = damping, step_size
        self.iterations, self.warmup, self.smoothness = iterations, warmup, smoothness

    @staticmethod
    def _loss(objective: Objective, x: np.ndarray, frame: int) -> float:
        r = np.asarray(objective.residual(x, frame), dtype=float).ravel()
        return 0.5 * objective.weight * float(r @ r)

    def _gradient(self, objective: Objective, x: np.ndarray, frame: int) -> tuple[np.ndarray, np.ndarray]:
        fun = lambda z: np.asarray(objective.residual(z, frame), dtype=float).ravel()
        r = fun(x)
        j = approx_derivative(fun, x, method="2-point")
        return j.T @ r * objective.weight, j * np.sqrt(objective.weight)

    def optimize(self, initial: np.ndarray) -> OptimizationResult:
        x = np.asarray(initial, dtype=float).copy()
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("initial values must be finite [frames, dofs]")
        levels = sorted({o.priority for o in self.objectives if o.priority is not None})
        for iteration in range(self.iterations):
            updated = x.copy()
            for t in range(len(x)):
                dx = np.zeros(x.shape[1])
                if iteration < self.warmup:
                    gradients = [self._gradient(o, x[t], t)[0] for o in self.objectives]
                    dx = -sum(gradients, np.zeros_like(dx))
                else:
                    projector = np.eye(x.shape[1])
                    for level in levels:
                        group = [o for o in self.objectives if o.priority == level]
                        for objective in group:
                            grad, _ = self._gradient(objective, x[t], t)
                            dx -= projector @ grad
                        jacobians = [self._gradient(o, x[t], t)[1] for o in group]
                        if jacobians:
                            j = np.concatenate(jacobians, axis=0)
                            projector -= j.T @ np.linalg.solve(j @ j.T + self.damping * np.eye(j.shape[0]), j @ projector)
                    for objective in self.objectives:
                        if objective.priority is None:
                            dx -= self._gradient(objective, x[t], t)[0]
                if self.smoothness and len(x) > 1:
                    left, right = x[max(t - 1, 0)], x[min(t + 1, len(x) - 1)]
                    dx += self.smoothness * (left + right - 2 * x[t])
                updated[t] += self.step_size * np.clip(dx, -1.0, 1.0)
            x = updated
        losses = {o.name: sum(self._loss(o, row, t) for t, row in enumerate(x)) / len(x)
                  for o in self.objectives}
        return OptimizationResult(x, losses)
