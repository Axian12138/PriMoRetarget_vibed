"""Grounded foot-pose targets from source steps and human support evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation

from .plugins import contiguous_windows


@dataclass
class GroundPlan:
    positions: np.ndarray
    rotations: np.ndarray
    support: np.ndarray
    human_floor_reference: np.ndarray
    floor_z: float


def capsule_endpoints(model, side):
    """Capsule endpoints in ankle-roll body coordinates, with their radii."""
    body = model.body(side + "_ankle_roll_link").id
    positions, radii = [], []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body or not (model.geom(geom).name or "").startswith(side + "_foot"):
            continue
        if int(model.geom_type[geom]) != 3:  # MuJoCo capsule
            continue
        rot = Rotation.from_quat(model.geom_quat[geom][[1, 2, 3, 0]]).as_matrix()
        for sign in [-1, 1]:
            positions.append(model.geom_pos[geom] + sign * model.geom_size[geom, 1] * rot[:, 2])
            radii.append(model.geom_size[geom, 0])
    if not positions:
        raise ValueError("no foot collision capsules for " + side)
    return np.asarray(positions), np.asarray(radii)


def sole_heights(model, data, side, floor_z=0.0):
    body = model.body(side + "_ankle_roll_link").id
    endpoints, radii = capsule_endpoints(model, side)
    world = endpoints @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
    return world[:, 2] - radii - floor_z


def make_ground_plan(foot_pos, foot_rot, human_feet, model, fps, floor_z=0.0):
    """Infer stance, anchor it to the floor, and retain human swing clearance.

    Human feet are joint points with a nonzero anatomical offset: calibrate each
    side from low, slow samples. Do not apply this offset to robot ankle centers.
    Contact is inferred, not a claim of measured load bearing. In stance, robot
    capsule sole extrema are 0--2 mm over the plane; swing stays above it.
    PCHIP through stance plateaus and swing samples makes pose targets C1.
    """
    n = len(foot_pos)
    speed = np.linalg.norm(np.gradient(human_feet, 1 / fps, axis=0), axis=-1)
    heights = human_feet[:, :, 2]
    refs = np.zeros(2)
    support = np.zeros((n, 2), bool)
    target_pos = foot_pos.copy()
    euler = np.unwrap(Rotation.from_matrix(foot_rot.reshape(-1, 3, 3)).as_euler("xyz").reshape(n, 2, 3), axis=0)
    target_rot = np.empty_like(foot_rot)
    for side, word in enumerate(["left", "right"]):
        low = (heights[:, side] <= np.quantile(heights[:, side], 0.35)) & (speed[:, side] < 0.25)
        if not low.any():
            raise ValueError("no low-speed human support evidence: " + word)
        refs[side] = np.median(heights[low, side])
        active = (heights[:, side] < refs[side] + 0.02) & (speed[:, side] < 0.25)
        # Remove very short evidence runs. No forced bilateral-support fallback.
        for start, stop in contiguous_windows(active):
            if stop - start < max(3, round(0.08 * fps)):
                active[start:stop] = False
        support[:, side] = active
        angles = euler[:, side].copy()
        lift = np.maximum(heights[:, side] - refs[side], 0.0)
        for start, stop in contiguous_windows(active):
            target_pos[start:stop, side, :2] = np.median(foot_pos[start:stop, side, :2], axis=0)
            angles[start:stop, :2] = 0.0
            angles[start:stop, 2] = np.median(angles[start:stop, 2])
            lift[start:stop] = 0.0
        samples = np.unique(np.r_[0, n - 1, np.arange(0, n, 4), np.flatnonzero(active)])
        for axis in range(2):
            target_pos[:, side, axis] = PchipInterpolator(samples, target_pos[samples, side, axis])(np.arange(n))
        angles = PchipInterpolator(samples, angles[samples], axis=0)(np.arange(n))
        lift = PchipInterpolator(samples, lift[samples])(np.arange(n))
        rotations = Rotation.from_euler("xyz", angles).as_matrix()
        endpoints, radii = capsule_endpoints(model, word)
        local_z = np.einsum("nij,kj->nki", rotations, endpoints)[:, :, 2] - radii
        target_pos[:, side, 2] = floor_z - local_z.min(axis=1) + lift
        target_rot[:, side] = rotations
    return GroundPlan(target_pos, target_rot, support, refs, float(floor_z))
