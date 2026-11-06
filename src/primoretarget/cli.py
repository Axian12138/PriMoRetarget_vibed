"""Refine G1 with priority-0 feet and human-derived priority-1 EEF windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .grounding import capsule_endpoints, make_ground_plan, sole_heights
from .optimizer import Objective
from .plugins import InteractionEEF, contiguous_windows
from .scene import SceneKinematics
from .trajectory import TrajectoryOptimizer


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--robot", type=Path, required=True)
    p.add_argument("--interaction-body", required=True, help="scene body whose center defines interaction proximity")
    p.add_argument(
        "--human-world",
        type=Path,
        required=True,
        help="aligned human BVH sidecar: positions_w, joint_names, source_frame, fps",
    )
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--eef-baseline-weight", type=float, default=1.0)
    p.add_argument("--eef-window-weight", type=float, default=1000.0)
    p.add_argument("--eef-transition-seconds", type=float, default=0.3)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--eef", action="store_true")
    p.add_argument(
        "--no-eef", dest="eef", action="store_false", help="disable EEF constraints for a diagnostic ablation"
    )
    p.add_argument(
        "--interaction-radius",
        type=float,
        default=0.15,
        help="human wrist to object CENTER distance, meters; inclusive threshold",
    )
    p.add_argument("--clearance", type=float, default=0.015)
    p.add_argument(
        "--collision-weight",
        type=float,
        default=100.0,
        help="lower-level nonpenetration cost; ground/EEF priorities remain unchanged",
    )
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--stop-frame", type=int)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if args.interaction_radius <= 0:
        p.error("interaction radius must be positive")
    if args.eef_baseline_weight <= 0 or args.eef_window_weight < args.eef_baseline_weight or args.eef_transition_seconds <= 0:
        p.error("EEF weights require 0 < baseline <= window and positive transition seconds")
    archive = args.episode / "retarget/g1_motion.npz"
    with np.load(archive, allow_pickle=False) as z:
        motion = {k: np.asarray(z[k]).copy() for k in z.files}
    q0 = motion["joint_pos"].astype(float)
    pos = motion["body_pos_w"].astype(float)
    quat = motion["body_quat_w"].astype(float)  # serialized WXYZ, not XYZW
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    n = len(q0)
    stop = n if args.stop_frame is None else args.stop_frame
    if not 0 <= args.start_frame < stop <= n:
        p.error("invalid source frame range")
    with np.load(args.human_world, allow_pickle=False) as z:
        human = z["positions_w"].copy()
        human_names = list(z["joint_names"])
        if len(human) != n or float(z["fps"]) != fps or not np.array_equal(z["source_frame"], np.arange(n)):
            raise ValueError("human source frames/FPS must match G1 exactly")
        if str(z["source_bvh_sha256"]) != sha256(args.episode / "human_motion.bvh"):
            raise ValueError("human sidecar does not refer to the input BVH")
    k = SceneKinematics(args.scene, args.robot, optimize_root=True, contact_bodies=(args.interaction_body,))
    if q0.shape[1] != len(k.joint_names) or pos.shape[1] != len(k.body_ids):
        raise ValueError("serialized G1 joint/body shape mismatch")
    lower = np.array([k.model.joint(name).range[0] for name in k.joint_names])
    upper = np.array([k.model.joint(name).range[1] for name in k.joint_names])
    initial = np.c_[np.zeros((n, 6)), q0]
    variable_lower = np.r_[np.full(6, -np.inf), lower]
    variable_upper = np.r_[np.full(6, np.inf), upper]
    root_rotations = Rotation.from_quat(quat[:, 0][:, [1, 2, 3, 0]])
    max_position_error, max_quaternion_error = 0.0, 0.0
    feet = ("left_ankle_roll_link", "right_ankle_roll_link")
    foot_pos, foot_rot = np.empty((n, 2, 3)), np.empty((n, 2, 3, 3))
    source_hand_rot = np.empty((n, 2, 3, 3))
    collision_before = np.empty(n)
    for t in range(n):
        k.forward(pos[t, 0], quat[t, 0], q0[t])
        max_position_error = max(
            max_position_error, float(np.linalg.norm(k.data.xpos[k.body_ids] - pos[t], axis=-1).max())
        )
        predicted = k.data.xquat[k.body_ids]
        max_quaternion_error = max(
            max_quaternion_error,
            float(
                np.minimum(
                    np.linalg.norm(predicted - quat[t], axis=-1), np.linalg.norm(predicted + quat[t], axis=-1)
                ).max()
            ),
        )
        for side, name in enumerate(feet):
            foot_pos[t, side] = k.body_pos(name)
            foot_rot[t, side] = k.data.xmat[k.model.body(name).id].reshape(3, 3)
            source_hand_rot[t, side] = k.data.xmat[
                k.model.body(("left", "right")[side] + "_wrist_yaw_link").id
            ].reshape(3, 3)
        collision_before[t] = np.max(k.geom_penetrations(args.clearance), initial=0.0)
    if max_position_error > 1e-4 or max_quaternion_error > 1e-4:
        raise ValueError(f"input FK mismatch: {max_position_error}, {max_quaternion_error}")
    target = k.body_pos(args.interaction_body)
    plane_ids = np.flatnonzero(k.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    floor_z = float(k.data.geom_xpos[plane_ids, 2].min())
    human_feet = human[:, [human_names.index("LeftFoot"), human_names.index("RightFoot")]]
    ground = make_ground_plan(foot_pos, foot_rot, human_feet, k.model, fps, floor_z)
    foot_pos, foot_rot = ground.positions, ground.rotations
    foot_shapes = [capsule_endpoints(k.model, word) for word in ["left", "right"]]
    interactions = {}
    for side in ("Left", "Right"):
        name = side + "Hand"
        if name not in human_names:
            raise ValueError(f"human wrist {name} missing")
        interactions[side.lower()] = InteractionEEF(
            human[:, human_names.index(name)], target, proximity=args.interaction_radius, priority=1,
            fps=fps, weight=args.eef_baseline_weight, window_weight=args.eef_window_weight,
            transition_seconds=args.eef_transition_seconds,
        )
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    windows = {
        side: {
            "source_joint": side.capitalize() + "Hand",
            "robot_eef": side + "_wrist_yaw_link",
            "frames": int(plugin.active.sum()),
            "windows_half_open": contiguous_windows(plugin.active),
            "windows_seconds_half_open": [[a / fps, b / fps] for a, b in contiguous_windows(plugin.active)],
            "min_center_distance_m": float(plugin.distances.min()),
        }
        for side, plugin in interactions.items()
    }
    report = {
        "frames": n,
        "fps": fps,
        "optimized_source_frames": [args.start_frame, stop],
        "priority_order": {
            "0": ["foot_positions", "stance_sole_level", "swing_floor_clearance"],
            "1": ["interaction_eef"],
            "2": [
                "obstacle_clearance",
                "global_velocity",
                "global_acceleration",
                "source_posture",
                "hand_orientation",
                "foot_orientation_preference",
            ],
        },
        "eef_enabled": args.eef,
        "interaction_radius_m": args.interaction_radius,
        "interaction_detection": "aligned HUMAN wrist to task OBJECT CENTER <= threshold; no padding",
        "eef_optimization_target": "per-frame HUMAN wrist world xyz, not object center or robot baseline",
        "target_xyz": target.tolist(),
        "interactions": windows,
        "root_variables": "world translation and rotation-vector deltas; constrained by grounded priority-0 foot targets",
        "foot_grounding": {
            "floor_z_m": floor_z,
            "human_reference_height_m": ground.human_floor_reference.tolist(),
            "support_frames": ground.support.sum(axis=0).tolist(),
            "source": "human low-height/low-speed inferred stance; robot capsule sole geometry; swing retained",
        },
        "trajectory_solver": "one sparse cross-frame system; all-frame EEF targets with smooth window weights",
        "priority_mode": "P0 hard feet; P1 weighted soft EEF jointly solved with P2 posture/collision and temporal regularizers below P0",
        "eef_weight_schedule": {
            "baseline": args.eef_baseline_weight,
            "window": args.eef_window_weight,
            "transition_seconds": args.eef_transition_seconds,
            "envelope": "C1 Gaussian distance-to-window shoulders, high plateau within detected windows",
            "target_scope": "entire trajectory for both hands",
        },
        "temporal_objective": {
            "velocity_weight": 0.16 * fps**2,
            "acceleration_weight": 0.01 * fps**4,
            "reference": "source correction differences; all variables optimized together",
        },
        "joint_bounds": "robot model limits only; no source-relative wrist clamp",
        "collision_contract": {
            "floor_contacts": "actual soles checked at priority 0; thin floor meshes allowed",
            "object_contact_margin_m": 0.0,
            "obstacle_margin_m": args.clearance,
        },
        "quaternion_order": "wxyz",
        "body_names": [k.model.body(int(i)).name for i in k.body_ids],
        "joint_names": list(k.joint_names),
        "input_fk": {"max_position_error_m": max_position_error, "max_quaternion_error": max_quaternion_error},
        "inputs": {str(path.resolve()): sha256(path) for path in (archive, args.scene, args.robot, args.human_world)},
        "visualization_started": False,
        "dynamics_verified": False,
    }
    (out / "input_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        out / "ground_plan.npz",
        positions_w=ground.positions,
        rotations_w=ground.rotations,
        support=ground.support,
        floor_z=ground.floor_z,
        source_frame=np.arange(n),
    )
    np.savez_compressed(
        out / "interaction_windows.npz",
        source_frame=np.arange(n),
        fps=fps,
        left_active=interactions["left"].active,
        right_active=interactions["right"].active,
        left_distances=interactions["left"].distances,
        right_distances=interactions["right"].distances,
        left_eef_target_w=interactions["left"].source_hand_xyz,
        right_eef_target_w=interactions["right"].source_hand_xyz,
        left_eef_weight=interactions["left"].weights,
        right_eef_weight=interactions["right"].weights,
        object_center_w=target,
    )
    print(json.dumps({"interactions": windows, "input_fk": report["input_fk"]}, indent=2), flush=True)
    if args.check_only:
        return

    def evaluate(q, t):
        k.root_rotation_vec = q[3:6]
        root_quat = (Rotation.from_rotvec(q[3:6]) * root_rotations[t]).as_quat()[[3, 0, 1, 2]]
        k.forward(pos[t, 0] + q[:3], root_quat, q[6:])
        return k

    def feet_residual(q, t):
        evaluate(q, t)
        r = []
        for side, name in enumerate(feet):
            r.extend(k.body_pos(name) - foot_pos[t, side])
            current = k.data.xmat[k.model.body(name).id].reshape(3, 3)
            if ground.support[t, side]:
                r.extend((current[:, 2] - np.array([0.0, 0.0, 1.0])) * 0.1)
            else:
                points, radii = foot_shapes[side]
                heights = (points @ current.T + k.body_pos(name))[:, 2] - radii - floor_z
                r.append(min(0.0, float(heights.min())))
        return np.asarray(r)

    def feet_jacobian(q, t):
        evaluate(q, t)
        rows = []
        for side, name in enumerate(feet):
            jp, jr = np.zeros((3, k.model.nv)), np.zeros((3, k.model.nv))
            mujoco.mj_jacBody(k.model, k.data, jp, jr, k.model.body(name).id)
            rows.append(k.jacobian_columns(jp))
            if ground.support[t, side]:
                normal = k.data.xmat[k.model.body(name).id].reshape(3, 3)[:, 2]
                cross = np.array([[0, -normal[2], normal[1]], [normal[2], 0, -normal[0]], [-normal[1], normal[0], 0]])
                rows.append(-0.1 * cross @ k.jacobian_columns(jr))
            else:
                points, radii = foot_shapes[side]
                current = k.data.xmat[k.model.body(name).id].reshape(3, 3)
                world = points @ current.T + k.body_pos(name)
                heights = world[:, 2] - radii - floor_z
                index = int(np.argmin(heights))
                if heights[index] < 0:
                    point_jac = np.zeros((3, k.model.nv))
                    mujoco.mj_jac(k.model, k.data, point_jac, None, world[index], k.model.body(name).id)
                    rows.append(k.jacobian_columns(point_jac)[2:3])
                else:
                    rows.append(np.zeros((1, len(q))))
        return np.concatenate(rows)

    collision_cache = {}

    def collision_terms(q, t):
        state = (t, q.tobytes())
        if collision_cache.get("state") != state:
            evaluate(q, t)
            collision_cache["state"] = state
            collision_cache["terms"] = k.collision_terms(args.clearance)
        return collision_cache["terms"]

    objectives = [Objective("feet", feet_residual, 0, jacobian=feet_jacobian)]
    if args.eef:
        for side, plugin in interactions.items():

            def eef_position(q, t, side=side):
                return evaluate(q, t).body_pos(side + "_wrist_yaw_link")

            def eef_jacobian(q, t, side=side):
                return evaluate(q, t).body_jacobian(side + "_wrist_yaw_link")

            objectives.append(plugin.objective(eef_position, eef_jacobian, name=side + "_interaction_eef"))
    objectives.append(
        Objective(
            "obstacle_clearance",
            lambda q, t: collision_terms(q, t)[0],
            2,
            weight=args.collision_weight,
            jacobian=lambda q, t: collision_terms(q, t)[1],
        )
    )
    posture_scale = np.r_[np.full(6, 3.0), np.ones(len(k.joint_names))]
    for i, name in enumerate(k.joint_names):
        if "_wrist_" in name:
            posture_scale[i + 6] = 8.0
    objectives.append(
        Objective(
            "source_posture",
            lambda q, t: (q - initial[t]) * posture_scale,
            2,
            1.0,
            lambda q, t: np.diag(posture_scale),
        )
    )

    def hand_orientation_residual(q, t):
        evaluate(q, t)
        rows = []
        for side, word in enumerate(["left", "right"]):
            current = k.data.xmat[k.model.body(word + "_wrist_yaw_link").id].reshape(3, 3)
            rows.extend(Rotation.from_matrix(current @ source_hand_rot[t, side].T).as_rotvec())
        return np.asarray(rows)

    def hand_orientation_jacobian(q, t):
        evaluate(q, t)
        rows = []
        for word in ["left", "right"]:
            jr = np.zeros((3, k.model.nv))
            mujoco.mj_jacBody(k.model, k.data, None, jr, k.model.body(word + "_wrist_yaw_link").id)
            rows.append(k.jacobian_columns(jr))
        return np.concatenate(rows)

    objectives.append(Objective("hand_orientation", hand_orientation_residual, 2, 4.0, hand_orientation_jacobian))

    def foot_orientation_preference(q, t):
        evaluate(q, t)
        return np.concatenate(
            [
                Rotation.from_matrix(k.data.xmat[k.model.body(name).id].reshape(3, 3) @ foot_rot[t, side].T).as_rotvec()
                for side, name in enumerate(feet)
            ]
        )

    def foot_orientation_preference_jacobian(q, t):
        evaluate(q, t)
        rows = []
        for name in feet:
            jr = np.zeros((3, k.model.nv))
            mujoco.mj_jacBody(k.model, k.data, None, jr, k.model.body(name).id)
            rows.append(k.jacobian_columns(jr))
        return np.concatenate(rows)

    objectives.append(
        Objective(
            "foot_orientation_preference", foot_orientation_preference, 2, 25.0, foot_orientation_preference_jacobian
        )
    )
    # Frame-local closures use global source frames, including bounded test runs.
    selected = [
        Objective(
            o.name,
            lambda q, t, o=o: o.residual(q, t + args.start_frame),
            o.priority,
            o.weight,
            lambda q, t, o=o: o.jacobian(q, t + args.start_frame),
        )
        for o in objectives
    ]
    optimizer = TrajectoryOptimizer(
        selected,
        iterations=args.iterations,
        warmup=0,
        damping=1e-5,
        step_size=0.15,
        fps=fps,
        constraint_priorities=(0,),
        velocity_weight=0.16 * fps**2,
        acceleration_weight=0.01 * fps**4,
        temporal_scale=np.r_[np.full(3, 4.0), np.full(3, 2.0), np.ones(len(k.joint_names))],
        lower=variable_lower,
        upper=variable_upper,
        progress=lambda iteration, change: print(f"iteration {iteration}: max_joint_update={change:.6g}", flush=True),
    )
    result = optimizer.optimize(initial[args.start_frame : stop])
    parameters = initial.copy()
    parameters[args.start_frame : stop] = result.values
    q = parameters[:, 6:]
    foot_error = np.zeros((n, 2))
    eef_before, eef_after = np.empty((n, 2)), np.empty((n, 2))
    collision_after = np.empty(n)
    absolute_sole = np.empty((n, 2, 2))
    for t in range(n):
        evaluate(initial[t], t)
        for side_index, (side, plugin) in enumerate(interactions.items()):
            eef_before[t, side_index] = np.linalg.norm(k.body_pos(side + "_wrist_yaw_link") - plugin.source_hand_xyz[t])
        evaluate(parameters[t], t)
        motion["body_pos_w"][t] = k.data.xpos[k.body_ids]
        motion["body_quat_w"][t] = k.data.xquat[k.body_ids]
        for side_index, (side, plugin) in enumerate(interactions.items()):
            eef_after[t, side_index] = np.linalg.norm(k.body_pos(side + "_wrist_yaw_link") - plugin.source_hand_xyz[t])
            foot_error[t, side_index] = np.linalg.norm(k.body_pos(feet[side_index]) - foot_pos[t, side_index])
            heights = sole_heights(k.model, k.data, side, floor_z)
            absolute_sole[t, side_index] = [heights.min(), heights.max()]
        collision_after[t] = np.max(k.geom_penetrations(args.clearance), initial=0.0)
    motion["joint_pos"] = q.astype(np.float32)
    motion["joint_vel"] = np.gradient(q, 1 / fps, axis=0).astype(np.float32)
    motion["body_lin_vel_w"] = np.gradient(motion["body_pos_w"], 1 / fps, axis=0).astype(np.float32)
    body_quats = motion["body_quat_w"].astype(float)
    # scipy Rotation expects [N,4], and XYZW; flatten before batch operations.
    rotations = Rotation.from_quat(body_quats.reshape(-1, 4)[:, [1, 2, 3, 0]])
    matrices = rotations.as_matrix().reshape(n, len(k.body_ids), 3, 3)
    relative = matrices[1:] @ np.swapaxes(matrices[:-1], -1, -2)
    av = np.zeros_like(motion["body_pos_w"])
    av[:-1] = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(n - 1, len(k.body_ids), 3) * fps
    av[-1] = av[-2]
    motion["body_ang_vel_w"] = av.astype(np.float32)
    np.savez_compressed(out / "g1_motion.npz", **motion)
    metrics = {}
    for i, (side, plugin) in enumerate(interactions.items()):
        active = plugin.active
        metrics[side] = {"frames": int(active.sum())}
        if active.any():
            metrics[side].update(
                {
                    "before_mean_m": float(eef_before[active, i].mean()),
                    "before_max_m": float(eef_before[active, i].max()),
                    "after_mean_m": float(eef_after[active, i].mean()),
                    "after_max_m": float(eef_after[active, i].max()),
                }
            )
    report.update(
        {
            "losses": result.losses,
            "iterations": args.iterations,
            "eef_metrics": metrics,
            "foot_position_deviation_max_m": float(foot_error.max()),
            "stance_sole_height_range_m": [
                float(absolute_sole[ground.support].min()),
                float(absolute_sole[ground.support].max()),
            ],
            "sole_penetration_max_m": float(max(0.0, -absolute_sole[:, :, 0].min())),
            "wrist_change_max_deg": {
                word: np.rad2deg(
                    np.max(
                        np.abs(
                            q[:, [i for i, name in enumerate(k.joint_names) if name.startswith(word + "_wrist_")]]
                            - q0[:, [i for i, name in enumerate(k.joint_names) if name.startswith(word + "_wrist_")]]
                        ),
                        axis=0,
                    )
                ).tolist()
                for word in ["left", "right"]
            },
            "joint_change_max_rad": float(np.max(np.abs(q - q0))),
            "root_translation_change_max_m": float(np.linalg.norm(parameters[:, :3], axis=-1).max()),
            "joint_limit_violation_max_rad": float(max(np.max(lower - q), np.max(q - upper), 0)),
            "clearance_violation_before_max_m": float(collision_before.max()),
            "clearance_violation_after_max_m": float(collision_after.max()),
            "clearance_violation_before_mean_m": float(collision_before.mean()),
            "clearance_violation_after_mean_m": float(collision_after.mean()),
            "output_sha256": sha256(out / "g1_motion.npz"),
        }
    )
    np.savez_compressed(
        out / "validation_curves.npz",
        source_frame=np.arange(n),
        eef_before=eef_before,
        eef_after=eef_after,
        foot_position_deviation=foot_error,
        clearance_before=collision_before,
        clearance_after=collision_after,
        absolute_sole=absolute_sole,
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
