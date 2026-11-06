"""Run prioritized refinement of a seven-field G1 motion archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .optimizer import Objective, PrioritizedOptimizer
from .scene import SceneKinematics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--robot", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--output", type=Path, default=Path("outputs/20261008_172423_003"))
    p.add_argument("--eef", action="store_true", help="enable optional interaction EEF objective")
    p.add_argument("--target", nargs=3, type=float, help="override the target xyz read from the task XML")
    p.add_argument("--interaction-radius", type=float, default=0.25,
                   help="source hand to target distance for activating EEF tracking (meters)")
    args = p.parse_args()
    archive = args.episode / "retarget/g1_motion.npz"
    with np.load(archive, allow_pickle=False) as z:
        motion = {k: np.asarray(z[k]).copy() for k in z.files}
    q0 = motion["joint_pos"].astype(float)
    pos = motion["body_pos_w"].astype(float)
    quat = motion["body_quat_w"].astype(float)
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    kinematics = SceneKinematics(args.scene, args.robot)
    if len(kinematics.joint_names) != q0.shape[1]:
        raise ValueError(f"G1 model has {len(kinematics.joint_names)} joints but archive has {q0.shape[1]}")
    feet = ("left_ankle_roll_link", "right_ankle_roll_link")
    hands = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    kinematics.forward(pos[0, 0], quat[0, 0][[3, 0, 1, 2]], q0[0])
    try:
        xml_target = kinematics.site_pos("task_00_goal")
        target_source = "task_00_goal site"
    except ValueError:
        xml_target = kinematics.body_pos("task_00_target")
        target_source = "task_00_target body"
    target = np.asarray(args.target, dtype=float) if args.target is not None else xml_target

    def evaluate(q, t):
        rootq = quat[t, 0][[3, 0, 1, 2]]
        kinematics.forward(pos[t, 0], rootq, q)
        return kinematics

    # Detect support from low vertical foot speed and proximity to the floor.
    fkfeet = np.empty((len(q0), 2, 3))
    for t in range(len(q0)):
        fk = evaluate(q0[t], t)
        fkfeet[t] = [fk.body_pos(n) for n in feet]
    speed = np.linalg.norm(np.gradient(fkfeet, 1 / fps, axis=0), axis=-1)
    support = (speed < 0.18) & (fkfeet[..., 2] < 0.12)
    anchors = fkfeet.copy()
    for side in range(2):
        active = np.flatnonzero(support[:, side])
        if len(active):
            for t in active:
                lo = max(0, t - 3); hi = min(len(q0), t + 4)
                anchors[t, side] = np.median(fkfeet[lo:hi, side], axis=0)

    def feet_residual(q, t):
        fk = evaluate(q, t)
        out = []
        for side, name in enumerate(feet):
            if support[t, side]:
                out.extend(fk.body_pos(name) - anchors[t, side])
            else:
                out.append(min(0.0, fk.body_pos(name)[2] - 0.025))
        return np.asarray(out)

    def collision_residual(q, t):
        fk = evaluate(q, t)
        return fk.geom_penetrations(0.025)

    objectives = [Objective("foot_support", feet_residual, 0, 1.0),
                  Objective("obstacle_clearance", collision_residual, 0, 2.0)]
    if args.eef:
        hand_track = np.empty((len(q0), 3))
        for t in range(len(q0)):
            evaluate(q0[t], t)
            hand_track[t] = kinematics.body_pos(hands[0])
        active = np.linalg.norm(hand_track - target, axis=-1) < args.interaction_radius
        def eef_residual(q, t):
            if not active[t]: return np.zeros(3)
            fk = evaluate(q, t)
            return fk.body_pos(hands[0]) - target
        objectives.append(Objective("interaction_eef", eef_residual, 1, 1.0))

    # Joint limits are enforced as a free objective and hard clipped after each solve.
    lower = np.array([kinematics.model.joint(n).range[0] for n in kinematics.joint_names])
    upper = np.array([kinematics.model.joint(n).range[1] for n in kinematics.joint_names])
    objectives.append(Objective("joint_limits", lambda q, t: np.r_[np.minimum(q - lower, 0), np.maximum(q - upper, 0)], None, 4.0))
    optimizer = PrioritizedOptimizer(objectives, iterations=args.iterations, warmup=2,
                                     damping=2e-3, step_size=0.02, smoothness=0.01)
    result = optimizer.optimize(q0)
    result.values[:] = np.clip(result.values, lower, upper)

    # Refresh all body kinematics from the refined motion.
    for t, q in enumerate(result.values):
        evaluate(q, t)
        motion["body_pos_w"][t] = kinematics.data.xpos[:motion["body_pos_w"].shape[1]]
        motion["body_quat_w"][t] = kinematics.data.xquat[:motion["body_quat_w"].shape[1]]
    motion["joint_pos"] = result.values.astype(np.float32)
    dt = 1 / fps
    motion["joint_vel"] = np.gradient(result.values, dt, axis=0).astype(np.float32)
    motion["body_lin_vel_w"] = np.gradient(motion["body_pos_w"], dt, axis=0).astype(np.float32)
    rotations = Rotation.from_quat(motion["body_quat_w"][..., [1, 2, 3, 0]])
    av = np.zeros_like(motion["body_pos_w"])
    av[:-1] = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / dt
    av[-1] = av[-2]
    motion["body_ang_vel_w"] = av.astype(np.float32)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "g1_motion.npz", **motion)
    report = {"losses": result.losses, "support_frames": int(support.sum()), "frames": len(q0),
              "eef_enabled": args.eef, "target_xyz": target.tolist(), "target_source": target_source,
              "interaction_radius_m": args.interaction_radius,
              "interaction_frames": int(active.sum()) if args.eef else 0,
              "iterations": args.iterations}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
