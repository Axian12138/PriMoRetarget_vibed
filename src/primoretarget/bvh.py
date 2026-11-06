"""Minimal BVH world-position reader for named joints."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def read_joint_positions(path: str | Path, names: tuple[str, ...]) -> tuple[np.ndarray, float]:
    lines = Path(path).read_text().splitlines()
    hierarchy, motion = lines.index("HIERARCHY"), lines.index("MOTION")
    tokens = re.findall(r"[{}]|[A-Za-z_][A-Za-z_0-9]*|[-+0-9.eE]+", "\n".join(lines[hierarchy + 1:motion]))
    joints = []
    i = 0
    while i < len(tokens):
        if tokens[i] in ("ROOT", "JOINT"):
            name = tokens[i + 1]; i += 2
            while i < len(tokens) and tokens[i] != "OFFSET": i += 1
            offset = np.array(list(map(float, tokens[i + 1:i + 4]))); i += 4
            while i < len(tokens) and tokens[i] != "CHANNELS": i += 1
            count = int(tokens[i + 1]); channels = tokens[i + 2:i + 2 + count]; i += 2 + count
            joints.append((name, offset, channels))
        else: i += 1
    frame_line = next(i for i, line in enumerate(lines) if line.startswith("Frames:"))
    frames = int(lines[frame_line].split()[1]); frame_time = float(lines[frame_line + 1].split()[2])
    values = np.asarray([[float(v) for v in line.split()] for line in lines[frame_line + 2:frame_line + 2 + frames]])
    out = np.zeros((frames, len(names), 3)); cursor = 0
    positions = np.zeros((frames, 3)); rotations = np.broadcast_to(np.eye(3), (frames, 3, 3)).copy()
    found = {}
    for name, offset, channels in joints:
        width = len(channels); block = values[:, cursor:cursor + width]; cursor += width
        translation = np.tile(offset, (frames, 1)); order, angles = [], []
        for j, channel in enumerate(channels):
            axis, kind = channel[:-7].lower(), channel[-7:].lower()
            if kind == "position": translation[:, "xyz".index(axis)] += block[:, j]
            else: order.append(axis.upper()); angles.append(block[:, j])
        local_rot = Rotation.from_euler("".join(order), np.stack(angles, axis=1), degrees=True).as_matrix() if angles else rotations
        positions = positions + np.einsum("nij,nj->ni", rotations, translation)
        rotations = rotations @ local_rot
        found[name] = positions.copy()
    for k, name in enumerate(names):
        if name not in found: raise ValueError(f"BVH joint {name!r} not found")
        out[:, k] = found[name] * 0.01
    return out, 1.0 / frame_time
