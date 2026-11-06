# PriMoRetarget_vibed

Standalone G1 episode refinement. All runtime code and the skill live in this
repository; NumPy, SciPy and MuJoCo are the only Python dependencies.

## Episode entry

From this repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
./scripts/primo_episode_to_g1.sh --episode data/episodes/example
```

The launcher uses the local virtual environment when present, otherwise
`python3`. It sets `PYTHONPATH=src` and calls `primoretarget.episode`.
The installed equivalent is `primoretarget-episode`.

Put the G1 XML and referenced mesh files in
`assets/robots/g1_29dof_dex3/`, or supply a relative `--robot` path.
Put scene assets under the episode package's manifest-relative location,
for example `data/scenes/`. `--assets-root` can specify that package root.
Only the episode is motion input; model and scene files are fixed assets.

```bash
./scripts/primo_episode_to_g1.sh \
  --episode data/episodes/example --output outputs/example-run
```

The episode adapter regenerates `human_world.npz` using the local BVH reader,
then runs the solver once to generate `g1_motion.npz`, ground plan, interaction
mask/weights and reports. It never needs an earlier optimized motion or sidecar.
Output paths and configuration paths are relative to this repository root.

Skill: [`primo-episode-to-g1`](skills/primo-episode-to-g1/SKILL.md).
Input/schema contract: [contract](skills/primo-episode-to-g1/references/contract.md).

## Optimization

P0 grounded feet are hard constraints. P1 human wrist targets exist throughout
the motion, with positive smooth interaction weights (1 outside, 1000 inside,
0.3 second Gaussian shoulders, 15 cm center-distance detection). Weighted soft
P1 tracking, P2 source posture/hand and foot orientation, collision costs and
whole-trajectory D1/D2 terms are solved jointly below the foot constraints.
Weights can therefore trade tracking accuracy against smoothness.

There are no motion-specific frame rules, source-relative wrist clamps,
strict extra collision level or residual-driven reruns. The default iteration
budget is 30. Joint bounds come from the robot model. Ground targets use actual
capsule soles and human stance/swing evidence. NPZ serialization retains 29
non-finger joints, 30 bodies and WXYZ quaternions; velocities are regenerated.

The current adapter assumes Axis BVH (centimetres, X up), synchronized source
frames, G1, a planar floor and one stationary target. Numerical completion does
not imply naturalness, collision-free grasping or dynamic feasibility.
`--prepare-only` creates the sidecar/command without generating a final motion.
Independent reproduction must use a fresh execution, not earlier artifacts.
