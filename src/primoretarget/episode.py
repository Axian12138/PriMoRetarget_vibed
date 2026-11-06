"""Generate aligned human data and run PriMo from one episode directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .bvh_fk import load_bvh_skeleton

DEFAULT_ROBOT = Path(
    "assets/robots/g1_29dof_dex3/g1_29dof_dex3_offsethand.xml"
)


def relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise argparse.ArgumentTypeError("use a path relative to the repository root")
    return path


def prepare_human(episode: Path, output: Path) -> dict:
    from scipy.spatial.transform import Rotation

    source = episode / "human_motion.bvh"
    metadata = json.loads((episode / "retarget/retarget.json").read_text())
    skeleton = load_bvh_skeleton(source)
    with np.load(episode / "retarget/g1_motion.npz", allow_pickle=False) as motion:
        frames = len(motion["joint_pos"])
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    if len(skeleton.positions) != frames or not np.isclose(skeleton.fps, fps):
        raise ValueError("BVH/G1 frame counts and FPS must agree")
    if metadata["source_frames"] != frames or metadata["output_frames"] != frames:
        raise ValueError("episode retarget metadata must preserve source frames")
    # Acquisition-format convention: Axis BVH centimetres, X up -> world Z up.
    reference = skeleton.positions[..., [2, 0, 1]] * 0.01
    alignment = metadata["root_alignment"]
    rotation = Rotation.from_quat(alignment["rotation_xyzw"])
    shift = np.asarray(alignment["actual_start_xyz"]) - rotation.apply(alignment["reference_start_xyz"])
    world = reference @ rotation.as_matrix().T + shift
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = shift
    np.savez_compressed(
        output / "human_world.npz",
        positions_w=world,
        joint_names=np.asarray(skeleton.joint_names),
        edges=skeleton.edges,
        fps=fps,
        source_frame=np.arange(frames),
        T_world_from_ref=matrix,
        source_bvh_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    return {"frames": frames, "fps": fps, "bvh_axis_order": "ZXY", "bvh_scale_m": 0.01}


def resolve_scene(episode: Path, manifest: dict, assets_root: Path | None) -> Path:
    link = relative_path(manifest["mujoco_conversion"]["task_xml"])
    roots = [assets_root] if assets_root is not None else [episode, *episode.parents]
    for root in roots:
        candidate = root / link
        if candidate.is_file():
            return Path(os.path.normpath(candidate))
    raise FileNotFoundError(f"missing fixed scene asset {link}; supply --assets-root using a relative path")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, type=relative_path)
    parser.add_argument("--output", type=relative_path)
    parser.add_argument("--robot", type=relative_path, default=DEFAULT_ROBOT)
    parser.add_argument("--assets-root", type=relative_path)
    parser.add_argument(
        "--prepare-only", action="store_true", help="generate the human sidecar and command without solving"
    )
    args = parser.parse_args()
    episode = args.episode
    manifest = json.loads((episode / "episode_manifest.json").read_text())
    config = json.loads((episode / "scene_config.json").read_text())
    task = config["task"]["id"]
    scene = resolve_scene(episode, manifest, args.assets_root)
    if not args.robot.is_file():
        parser.error(f"missing G1 model {args.robot}; supply --robot with a relative asset path")
    output = args.output or (
        Path("outputs")
        / datetime.now().strftime("%Y-%m-%d")
        / "primo_episode"
        / "runs"
        / datetime.now().strftime("%H-%M-%S-%f")
    )
    if output.exists():
        parser.error(f"output already exists: {output}; choose a fresh run directory")
    output.mkdir(parents=True)
    info = prepare_human(episode, output)
    command = [
        "--episode",
        str(episode),
        "--scene",
        str(scene),
        "--robot",
        str(args.robot),
        "--human-world",
        str(output / "human_world.npz"),
        "--interaction-body",
        task + "_target",
        "--eef",
        "--interaction-radius",
        "0.15",
        "--eef-baseline-weight",
        "1",
        "--eef-window-weight",
        "1000",
        "--eef-transition-seconds",
        "0.3",
        "--iterations",
        "30",
        "--clearance",
        "0.015",
        "--collision-weight",
        "100",
        "--output",
        str(output),
    ]
    provenance = {
        "episode": str(episode),
        "scene_asset": str(scene),
        "robot_asset": str(args.robot),
        "human_world": str(output / "human_world.npz"),
        "output_motion": str(output / "g1_motion.npz"),
        "source_format": info,
        "solver_arguments": command,
        "inputs": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                episode / "human_motion.bvh",
                episode / "retarget/g1_motion.npz",
                episode / "retarget/retarget.json",
                scene,
                args.robot,
            ]
        },
        "status": "prepared" if args.prepare_only else "running",
    }
    provenance_file = output / "episode_pipeline.json"
    provenance_file.write_text(json.dumps(provenance, indent=2) + "\n")
    label = "Prepared; motion will be written to" if args.prepare_only else "Output"
    print(f"{label}: {output / 'g1_motion.npz'}", flush=True)
    if args.prepare_only:
        return
    from primoretarget.cli import main as solve

    sys.argv = ["primoretarget", *command]
    solve()
    provenance["status"] = "executed_pending_user_review"
    provenance_file.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
