"""Read named BVH joints and evaluate forward kinematics without robot assets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BvhSkeleton:
    """Positions in the source BVH axes and units; end sites are omitted."""

    positions: np.ndarray
    joint_names: tuple[str, ...]
    edges: np.ndarray
    fps: float


def load_bvh_skeleton(path: Path) -> BvhSkeleton:
    """Preserve declared Euler order, bone offsets and per-joint translations."""
    from scipy.spatial.transform import Rotation

    hierarchy, motion = path.read_text().split("MOTION", maxsplit=1)
    tokens = iter(re.findall(r"[^\s{}]+|[{}]", hierarchy))
    names, parents, offsets, channels = [], [], [], []

    def expect(value: str) -> None:
        actual = next(tokens)
        if actual != value:
            raise ValueError(f"BVH expected {value!r}, got {actual!r}")

    def joint(parent: int) -> None:
        name = next(tokens)
        index = len(names)
        names.append(name)
        parents.append(parent)
        offsets.append(None)
        channels.append(None)
        expect("{")
        for token in tokens:
            if token == "}":
                break
            if token == "OFFSET":
                offsets[index] = [float(next(tokens)) for _ in range(3)]
            elif token == "CHANNELS":
                channels[index] = [next(tokens) for _ in range(int(next(tokens)))]
            elif token == "JOINT":
                joint(index)
            elif token == "End":
                expect("Site")
                expect("{")
                expect("OFFSET")
                for _ in range(3):
                    float(next(tokens))
                expect("}")
            else:
                raise ValueError(f"unsupported BVH hierarchy token {token!r}")
        if offsets[index] is None or channels[index] is None:
            raise ValueError(f"BVH joint {name!r} lacks offsets or channels")

    expect("HIERARCHY")
    expect("ROOT")
    joint(-1)
    if next(tokens, None) is not None or len(set(names)) != len(names):
        raise ValueError("BVH must have one root and unique joint names")
    header = re.fullmatch(r"\s*Frames:\s*(\d+)\s+Frame Time:\s*([\d.eE+-]+)\s+(.*)", motion, flags=re.DOTALL)
    if header is None:
        raise ValueError("invalid BVH motion header")
    frames, frame_time = int(header[1]), float(header[2])
    values = np.fromstring(header[3], sep=" ")
    width = sum(map(len, channels))
    if (
        frames <= 0
        or not np.isfinite(frame_time)
        or frame_time <= 0
        or values.size != frames * width
        or not np.isfinite(values).all()
    ):
        raise ValueError("invalid BVH frame count, time or channel values")
    values = values.reshape(frames, width)
    positions = np.empty((frames, len(names), 3))
    rotations = np.empty((frames, len(names), 3, 3))
    column = 0
    for index, joint_channels in enumerate(channels):
        local_position = np.tile(offsets[index], (frames, 1)).astype(float)
        angles, axes = [], []
        for channel in joint_channels:
            if channel[0] not in "XYZ":
                raise ValueError(f"invalid BVH channel {channel!r}")
            if channel.endswith("position"):
                local_position[:, "XYZ".index(channel[0])] += values[:, column]
            elif channel.endswith("rotation"):
                axes.append(channel[0])
                angles.append(values[:, column])
            else:
                raise ValueError(f"unsupported BVH channel {channel!r}")
            column += 1
        local_rotation = np.tile(np.eye(3), (frames, 1, 1))
        if axes:
            angle_array = np.stack(angles, axis=1)
            local_rotation = Rotation.from_euler("".join(axes), angle_array, degrees=True).as_matrix()
        parent = parents[index]
        if parent < 0:
            positions[:, index] = local_position
            rotations[:, index] = local_rotation
        else:
            positions[:, index] = positions[:, parent] + np.einsum("tij,tj->ti", rotations[:, parent], local_position)
            rotations[:, index] = rotations[:, parent] @ local_rotation
    edges = np.asarray(
        [(parent, index) for index, parent in enumerate(parents) if parent >= 0], dtype=np.int32
    ).reshape(-1, 2)
    return BvhSkeleton(positions, tuple(names), edges, 1 / frame_time)
