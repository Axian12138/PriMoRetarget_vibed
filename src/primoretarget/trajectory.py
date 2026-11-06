"""Prioritized trajectory solve with truly coupled sparse temporal objectives."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr

from .optimizer import OptimizationResult, PrioritizedOptimizer


class TrajectoryOptimizer(PrioritizedOptimizer):
    """P0/P1 are frame constraints; lower terms share one trajectory system.

    D1/D2 act on all optimization variables together. Window constraints therefore
    influence preceding/following free frames, without extending the EEF mask.
    Posture, orientation, collision and temporal preferences compete in the same
    lower level, rather than losing posture behind a full-rank local smoothness J.
    """

    def __init__(
        self,
        objectives,
        *,
        fps=50.0,
        velocity_weight=None,
        acceleration_weight=None,
        temporal_scale=None,
        constraint_priorities=(0, 1),
        **kwargs,
    ):
        super().__init__(objectives, smoothness=0, **kwargs)
        self.fps = fps
        self.velocity_weight = 0.16 * fps**2 if velocity_weight is None else velocity_weight
        self.acceleration_weight = 0.01 * fps**4 if acceleration_weight is None else acceleration_weight
        self.temporal_scale = temporal_scale
        self.constraint_priorities = constraint_priorities

    def _hierarchy(self, point, t, groups, fixed):
        dx = np.zeros(len(point))
        basis = np.eye(len(point))[:, ~fixed]
        for group in groups:
            terms = [self._terms(o, point, t) for o in group]
            r = np.concatenate([v[0] for v in terms])
            j = np.concatenate([v[1] for v in terms])
            a = j @ basis
            if not a.size:
                continue
            u, s, vt = np.linalg.svd(a, full_matrices=True)
            rank = int(np.sum(s > max(1e-9, (s[0] if len(s) else 0) * 1e-7)))
            if rank:
                correction = (
                    basis
                    @ vt[:rank].T
                    @ ((s[:rank] / (s[:rank] ** 2 + self.damping)) * (u[:, :rank].T @ (-r - j @ dx)))
                )
                dx += correction
            basis = basis @ vt[rank:].T
        return dx, basis

    def _project_top(self, x, top, lower, upper):
        x = x.copy()
        project_groups = [top]
        project_objectives = top
        for t in range(len(x)):
            frame_lower = lower[t] if lower.ndim == 2 else lower
            frame_upper = upper[t] if upper.ndim == 2 else upper
            for _ in range(25):
                terms = [self._terms(o, x[t], t) for o in project_objectives]
                r = np.concatenate([v[0] for v in terms])
                if np.max(np.abs(r), initial=0) < 2e-7:
                    break
                fixed = np.zeros(x.shape[1], bool)
                for _ in range(4):
                    dx, _ = self._hierarchy(x[t], t, project_groups, fixed)
                    blocked = ((x[t] <= frame_lower + 1e-8) & (dx < 0)) | ((x[t] >= frame_upper - 1e-8) & (dx > 0))
                    if not np.any(blocked & ~fixed):
                        break
                    fixed |= blocked
                scale = min(1.0, 0.15 / max(np.max(np.abs(dx)), 1e-12))
                old = float(r @ r)
                for _ in range(14):
                    trial = x[t] + scale * dx
                    if np.any(trial < frame_lower - 1e-8) or np.any(trial > frame_upper + 1e-8):
                        scale *= 0.5
                        continue
                    rr = np.concatenate([self._terms(o, trial, t)[0] for o in project_objectives])
                    if rr @ rr < old:
                        x[t] = trial
                        break
                    scale *= 0.5
        return x

    def optimize(self, initial):
        x = np.asarray(initial, dtype=float).copy()
        reference = x.copy()
        n, d = x.shape
        lower = np.broadcast_to(-np.inf if self.lower is None else self.lower, (n, d))
        upper = np.broadcast_to(np.inf if self.upper is None else self.upper, (n, d))
        if np.any(x < lower - 1e-8) or np.any(x > upper + 1e-8):
            raise ValueError("initial joint limits")
        high_levels = sorted({o.priority for o in self.objectives if o.priority in self.constraint_priorities})
        groups = [[o for o in self.objectives if o.priority == p] for p in high_levels]
        low = [o for o in self.objectives if o.priority not in self.constraint_priorities]
        if not groups:
            raise ValueError("trajectory hierarchy needs priority-0 constraint")
        # This is a highest-priority feasibility projection, not mixed warmup.
        x = self._project_top(x, groups[0], lower, upper)
        temporal_scale = np.ones(d) if self.temporal_scale is None else self.temporal_scale
        first = sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n))
        second = sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))
        t1 = sparse.kron(first, sparse.diags(temporal_scale), format="csr") * np.sqrt(self.velocity_weight)
        t2 = sparse.kron(second, sparse.diags(temporal_scale), format="csr") * np.sqrt(self.acceleration_weight)
        temporal = sparse.vstack([t1, t2], format="csr")
        fixed = np.zeros((n, d), bool)
        for iteration in range(self.iterations):
            for _active_pass in range(4):
                parts = []
                bases = []
                local_matrices = []
                local_rhs = []
                for t in range(n):
                    particular, basis = self._hierarchy(x[t], t, groups, fixed[t])
                    terms = [self._terms(o, x[t], t) for o in low]
                    r = np.concatenate([v[0] for v in terms])
                    j = np.concatenate([v[1] for v in terms])
                    parts.append(particular)
                    bases.append(basis)
                    local_matrices.append(sparse.csr_matrix(j @ basis))
                    local_rhs.append(-r - j @ particular)
                particular = np.asarray(parts)
                global_basis = sparse.block_diag([sparse.csr_matrix(b) for b in bases], format="csr")
                matrix = sparse.vstack(
                    [sparse.block_diag(local_matrices, format="csr"), temporal @ global_basis], format="csr"
                )
                rhs = np.r_[np.concatenate(local_rhs), -temporal @ (x - reference + particular).ravel()]
                solution = lsmr(matrix, rhs, atol=1e-7, btol=1e-7, maxiter=160)[0]
                dx = particular + (global_basis @ solution).reshape(n, d)
                blocked = ((x <= lower + 1e-8) & (dx < 0)) | ((x >= upper - 1e-8) & (dx > 0))
                if not np.any(blocked & ~fixed):
                    break
                fixed |= blocked
            # Permit inward departure from an old active bound on the next step.
            fixed = ((x <= lower + 1e-8) & (dx < 0)) | ((x >= upper - 1e-8) & (dx > 0))
            # A local active bound must not veto unrelated frames in the global
            # trajectory. Scalar scaling retains each frame's linear null space.
            ratios = np.ones_like(dx)
            np.divide(upper - x, dx, out=ratios, where=dx > 1e-12)
            negative = np.ones_like(dx)
            np.divide(lower - x, dx, out=negative, where=dx < -1e-12)
            bound_scale = np.clip(np.minimum(ratios, negative).min(axis=1), 0, 1)
            dx *= bound_scale[:, None]
            trust = min(1.0, self.step_size / max(np.max(np.abs(dx)), 1e-12))
            before = [sum(self._loss(o, x[t], t) for o in group for t in range(n)) for group in groups]
            low_before = sum(self._loss(o, x[t], t) for o in low for t in range(n)) + 0.5 * np.sum(
                (temporal @ (x - reference).ravel()) ** 2
            )
            updated = x.copy()
            for _ in range(10):
                candidate = self._project_top(x + trust * dx, groups[0], lower, upper)
                if np.any(candidate < lower - 1e-8) or np.any(candidate > upper + 1e-8):
                    trust *= 0.5
                    continue
                after = [sum(self._loss(o, candidate[t], t) for o in group for t in range(n)) for group in groups]
                low_after = sum(self._loss(o, candidate[t], t) for o in low for t in range(n)) + 0.5 * np.sum(
                    (temporal @ (candidate - reference).ravel()) ** 2
                )
                accepted = True
                for old, new in zip([*before, low_before], [*after, low_after], strict=True):
                    tol = 1e-9 + old * 1e-5
                    if new > old + tol:
                        accepted = False
                        break
                    if new < old - tol:
                        break
                if accepted:
                    updated = candidate
                    break
                trust *= 0.5
            change = float(np.max(np.abs(updated - x)))
            x = updated
            if self.progress:
                self.progress(iteration + 1, change)
            if change < 2e-6:
                break
        losses = {o.name: sum(self._loss(o, row, t) for t, row in enumerate(x)) / n for o in self.objectives}
        return OptimizationResult(x, losses)
