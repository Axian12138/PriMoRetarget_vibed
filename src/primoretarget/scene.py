"""Combined MuJoCo scene and G1 kinematics."""

from __future__ import annotations

import copy
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


class SceneKinematics:
    def __init__(self, scene_xml: str | Path, robot_xml: str | Path, *, optimize_root=False, contact_bodies=()):
        self.contact_bodies = frozenset(contact_bodies)
        scene_xml, robot_xml = Path(scene_xml).resolve(), Path(robot_xml).resolve()
        scene, robot = ET.parse(scene_xml).getroot(), ET.parse(robot_xml).getroot()
        scene_assets = scene.find("asset")
        robot_assets = robot.find("asset")
        robot_compiler = robot.find("compiler")
        robot_meshdir = robot_compiler.get("meshdir", ".") if robot_compiler is not None else "."
        for source in list(robot_assets or []):
            asset = copy.deepcopy(source)
            if asset.get("file"):
                resource = robot_xml.parent / robot_meshdir / asset.get("file")
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
        with tempfile.TemporaryDirectory(prefix="primoretarget_model_") as temporary:
            merged_xml = Path(temporary) / "model.xml"
            ET.ElementTree(scene).write(merged_xml, encoding="utf-8", xml_declaration=True)
            self.model = mujoco.MjModel.from_xml_path(str(merged_xml))
        self.data = mujoco.MjData(self.model)
        self.joint_names = tuple(
            n
            for n in (
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            )
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0
        )
        self.qpos_indices = np.array([self.model.joint(n).qposadr[0] for n in self.joint_names])
        self.dof_indices = np.array([self.model.joint(n).dofadr[0] for n in self.joint_names])
        root = self.model.body("pelvis").id
        free = [
            i
            for i in range(self.model.njnt)
            if self.model.jnt_bodyid[i] == root and self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if len(free) != 1:
            raise ValueError("expected one floating-base joint on pelvis")
        self.root_qpos = int(self.model.jnt_qposadr[free[0]])
        self.root_dof = int(self.model.jnt_dofadr[free[0]])
        self.optimize_root = optimize_root
        self.root_rotation_vec = np.zeros(3)
        self.body_ids = np.array([root, *[self.model.jnt_bodyid[self.model.joint(n).id] for n in self.joint_names]])
        descendants = {root}
        for i in range(root + 1, self.model.nbody):
            if int(self.model.body_parentid[i]) in descendants:
                descendants.add(i)
        self.robot_geom_ids = np.array(
            [
                i
                for i in range(self.model.ngeom)
                if self.model.geom_contype[i] and int(self.model.geom_bodyid[i]) in descendants
            ]
        )
        self.scene_geom_ids = np.array(
            [
                i
                for i in range(self.model.ngeom)
                if self.model.geom_contype[i]
                and int(self.model.geom_bodyid[i]) not in descendants
                and self.model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE
            ]
        )
        self.data.qpos[:] = self.model.qpos0
        mujoco.mj_forward(self.model, self.data)
        self.scene_centers = self.data.geom_xpos[self.scene_geom_ids].copy()
        self.scene_radii = self.model.geom_rbound[self.scene_geom_ids].copy()
        floor_z = float(self.data.geom_xpos[self.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE, 2].min())
        self.floor_mesh_ids = set()
        for geom in self.scene_geom_ids:
            if self.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = int(self.model.geom_dataid[geom])
            start, count = int(self.model.mesh_vertadr[mesh]), int(self.model.mesh_vertnum[mesh])
            vertices = (
                self.model.mesh_vert[start : start + count] @ self.data.geom_xmat[geom].reshape(3, 3).T
                + self.data.geom_xpos[geom]
            )
            lo, hi = vertices.min(axis=0), vertices.max(axis=0)
            if lo[2] >= floor_z - 0.001 and hi[2] <= floor_z + 0.001 and np.min(hi[:2] - lo[:2]) > 0.2:
                self.floor_mesh_ids.add(int(geom))
        self._state = None

    def forward(self, root_pos: np.ndarray, root_quat_wxyz: np.ndarray, joints: np.ndarray) -> None:
        state = np.r_[root_pos, root_quat_wxyz, joints]
        if self._state is not None and np.array_equal(state, self._state):
            return
        self.data.qpos[self.root_qpos : self.root_qpos + 3] = root_pos
        self.data.qpos[self.root_qpos + 3 : self.root_qpos + 7] = root_quat_wxyz
        self.data.qpos[self.qpos_indices] = joints
        mujoco.mj_forward(self.model, self.data)
        self._state = state.copy()

    def body_pos(self, name: str) -> np.ndarray:
        return self.data.xpos[self.model.body(name).id].copy()

    def body_jacobian(self, name: str) -> np.ndarray:
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jac, None, self.model.body(name).id)
        return self.jacobian_columns(jac)

    def jacobian_columns(self, jac):
        """Map MuJoCo local angular velocity to world root rotation-vector parameters."""
        joints = jac[:, self.dof_indices]
        if not self.optimize_root:
            return joints
        v = self.root_rotation_vec
        theta = np.linalg.norm(v)
        cross = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        if theta < 1e-6:
            left_jac = np.eye(3) + 0.5 * cross + cross @ cross / 6
        else:
            left_jac = (
                np.eye(3) + (1 - np.cos(theta)) / theta**2 * cross + (theta - np.sin(theta)) / theta**3 * cross @ cross
            )
        root_rotation = self.data.xmat[self.model.body("pelvis").id].reshape(3, 3)
        start = self.root_dof
        return np.c_[jac[:, start : start + 3], jac[:, start + 3 : start + 6] @ root_rotation.T @ left_jac, joints]

    def collision_terms(self, margin: float = 0.015) -> tuple[np.ndarray, np.ndarray]:
        """Worst signed distance per robot geom, with closest-point Jacobians.

        The infinite floor is handled by the separate foot objective. Mesh
        broad phase uses compiled world centers and geom_rbound, not mesh sizes.
        """
        penalties = np.zeros(len(self.robot_geom_ids))
        jacobians = np.zeros((len(penalties), len(self.joint_names) + (6 if self.optimize_root else 0)))
        for k, rid in enumerate(self.robot_geom_ids):
            nearest_gap = np.inf
            candidates = np.flatnonzero(
                np.linalg.norm(self.scene_centers - self.data.geom_xpos[rid], axis=1)
                < self.model.geom_rbound[rid] + self.scene_radii + margin
            )
            for candidate in candidates:
                sid = int(self.scene_geom_ids[candidate])
                if sid in self.floor_mesh_ids and "_foot" in (self.model.geom(int(rid)).name or ""):
                    continue  # floor contact is constrained by the actual sole objective
                pair_margin = (
                    0.0 if self.model.body(int(self.model.geom_bodyid[sid])).name in self.contact_bodies else margin
                )
                segment = np.zeros(6)
                dist = mujoco.mj_geomDistance(self.model, self.data, int(rid), sid, margin, segment)
                gap = dist - pair_margin
                if gap >= nearest_gap or gap >= 0:
                    continue
                nearest_gap = gap
                penalties[k] = max(0.0, -gap)
                jac = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, self.data, jac, None, segment[:3], int(self.model.geom_bodyid[rid]))
                if abs(dist) > 1e-9:
                    normal = (segment[:3] - segment[3:]) / dist
                    jacobians[k] = -normal @ self.jacobian_columns(jac)
        return penalties, jacobians

    def site_pos(self, name: str) -> np.ndarray:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if idx < 0:
            raise ValueError(f"site {name!r} not found in scene")
        return self.data.site_xpos[idx].copy()

    def geom_penetrations(self, margin: float = 0.015) -> np.ndarray:
        return self.collision_terms(margin)[0]
