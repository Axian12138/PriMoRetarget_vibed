"""Combined MuJoCo scene and G1 kinematics."""
from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


class SceneKinematics:
    def __init__(self, scene_xml: str | Path, robot_xml: str | Path):
        scene_xml, robot_xml = Path(scene_xml).resolve(), Path(robot_xml).resolve()
        scene, robot = ET.parse(scene_xml).getroot(), ET.parse(robot_xml).getroot()
        scene_assets = scene.find("asset")
        robot_assets = robot.find("asset")
        for asset in list(robot_assets or []):
            if asset.get("file"):
                resource = robot_xml.parent / asset.get("file")
                if not resource.exists():
                    resource = robot_xml.parent / "meshes" / asset.get("file")
                asset.set("file", str(resource.resolve()))
            scene_assets.append(asset)
        defaults = robot.find("default")
        if defaults is not None:
            scene.insert(list(scene).index(scene_assets), defaults)
        scene_world = scene.find("worldbody")
        robot_world = robot.find("worldbody")
        for body in list(robot_world):
            scene_world.append(body)
        # Preserve scene mesh references relative to the scene XML directory.
        compiler = scene.find("compiler")
        if compiler is not None and compiler.get("meshdir"):
            compiler.set("meshdir", str((scene_xml.parent / compiler.get("meshdir")).resolve()))
        merged_xml = scene_xml.parent / ".primoretarget_merged.xml"
        try:
            ET.ElementTree(scene).write(merged_xml, encoding="utf-8", xml_declaration=True)
            self.model = mujoco.MjModel.from_xml_path(str(merged_xml))
        finally:
            merged_xml.unlink(missing_ok=True)
        self.data = mujoco.MjData(self.model)
        self.joint_names = tuple(n for n in ("left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint",
            "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint",
            "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
            "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
            "right_wrist_pitch_joint", "right_wrist_yaw_joint") if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0)
        self.qpos_indices = np.array([self.model.joint(n).qposadr[0] for n in self.joint_names])
        self.robot_geom_ids = np.array([i for i in range(self.model.ngeom)
            if self.model.geom_contype[i] and self.model.geom_bodyid[i] != 0 and
            self.model.body(self.model.geom_bodyid[i]).name not in ("task_00_target",)])
        self.scene_geom_ids = np.array([i for i in range(self.model.ngeom)
            if self.model.geom_contype[i] and self.model.geom_bodyid[i] == 0])
        self.scene_centers = self.model.geom_pos[self.scene_geom_ids].copy()
        self.scene_radii = np.array([np.linalg.norm(self.model.geom_size[i]) for i in self.scene_geom_ids])

    def forward(self, root_pos: np.ndarray, root_quat_wxyz: np.ndarray, joints: np.ndarray) -> None:
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_quat_wxyz
        self.data.qpos[self.qpos_indices] = joints
        mujoco.mj_forward(self.model, self.data)

    def body_pos(self, name: str) -> np.ndarray:
        return self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()

    def site_pos(self, name: str) -> np.ndarray:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if idx < 0:
            raise ValueError(f"site {name!r} not found in scene")
        return self.data.site_xpos[idx].copy()

    def geom_penetrations(self, margin: float = 0.015) -> np.ndarray:
        penalties = np.zeros(len(self.robot_geom_ids), dtype=float)
        # Static scene mesh bounds are broad-phase filters; exact MuJoCo distance
        # queries run only for obstacle candidates near each robot geom.
        for k, rid in enumerate(self.robot_geom_ids):
            center = self.data.geom_xpos[rid]
            radius = np.linalg.norm(self.model.geom_size[rid])
            candidates = np.flatnonzero(np.linalg.norm(self.scene_centers - center, axis=1)
                                        < radius + self.scene_radii + margin + 0.35)
            for candidate in candidates:
                sid = self.scene_geom_ids[candidate]
                dist = mujoco.mj_geomDistance(self.model, self.data, int(rid), int(sid), margin, None)
                penalties[k] = max(penalties[k], margin - dist)
        return np.maximum(penalties, 0.0)
