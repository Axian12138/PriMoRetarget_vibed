# PriMoRetarget_vibed

Prioritized G1 refinement with real sole grounding and a coupled trajectory solve.
Consumes an existing seven-field G1 archive and verified aligned human BVH world
sidecar. No other retarget solver is invoked by this CLI.

## Hierarchy

- **0:** grounded foot-step positions; upward horizontal stance soles; swing
  floor clearance. Target stance positions are anchored and human swing lift is
  retained. Actual robot capsule radii/endpoints define the floor offsets.
- **1:** weighted human-wrist EEF targets at every frame. Detection
  uses each human wrist to object-center distance <= 0.15 m. Fitting uses the
  per-frame human wrist world XYZ, not the object center. Window weights have a
  high plateau and Gaussian shoulders; far from windows the weight remains positive.
- **2:** posture, wrist/hand and foot orientation, obstacle clearance and true
  cross-frame velocity/acceleration preferences share the same sparse solve.

`trajectory.py` constructs frame-local high-priority particular solutions and
null spaces for hard feet, then solves one block sparse system for weighted EEF
and lower regularizers over the entire trajectory.
P1 is weighted soft tracking, not a per-frame equality: otherwise a scalar
weight cancels from the constrained solution and cannot trade off with smoothing.
D1/D2 and smooth EEF weights jointly shape the entire trajectory. The wrist
posture preference is active in this lower system; it is not projected away by
a preceding full-rank per-frame averaging objective. Root translation and
world rotation-vector deltas allow proximal compensation while satisfying feet.
Wrist yaw about the EEF origin does not control that origin's XYZ, so orientation
and joint preferences are essential for meaningful hand posture.

`grounding.py` calibrates anatomical human foot-point height from low/slow
samples, distinguishes stance/swing, and generates C1 target paths. Stance
capsule sole extrema are approximately 0--2 mm over the floor. Foot yaw is a
preference rather than an unconditional frozen old orientation, preserving
feasibility of grounded gait. Actual output sole heights must be checked
independently; equality to an old floating foot trajectory is not grounding.

## Run

With NumPy, SciPy and MuJoCo, from this submodule:

```bash
PYTHONPATH=src python -m primoretarget.cli \
  --episode /path/to/episodes/20261008_172423_003 \
  --scene /path/to/tasks/task_00.xml \
  --robot /path/to/g1_29dof_dex3_offsethand.xml \
  --human-world /path/to/human_world.npz \
  --eef --interaction-body target_body --interaction-radius .15 --output /path/to/new-run
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v
```

`--check-only` validates inputs and exports plans without optimization/servers.
`--no-eef` enables a diagnostic comparison. `--start-frame`/`--stop-frame` limit
the solve while retaining source frames outside; short probes do not validate
full-sequence continuity. Human sidecars require world-meter `positions_w`,
`joint_names` (LeftHand/RightHand/LeftFoot/RightFoot), `source_frame`, `fps` and
`source_bvh_sha256`, synchronized exactly to G1 source frames.

Scene/robot floating-base offsets and body indices come from model identities;
input and output serialization use WXYZ quaternions and the real 29-joint /
30-body order. CLI outputs `g1_motion.npz`, `ground_plan.npz`,
`interaction_windows.npz`, `validation_curves.npz`, `input_audit.json` and
`report.json`. No mixed warmup, post-solve clipping, live rendering or physics
rollout is performed. Independent checks must cover the actual sole geometry,
window influence, wrist changes and seams; numerical EEF precision does not
prove mocap accuracy, contact, naturalness or dynamic feasibility.

The legacy minimal BVH reader is not used. `optimizer.py` retains the frame-local
helper/API; the corrected CLI uses `TrajectoryOptimizer` for time coupling.

## General design and scope

Use one invocation with the fixed three-level hierarchy above. Iterative
linearization, trust steps and constraint retraction are internal to this one
solve; they are not a sequence of motion-specific repair passes. There are no
hardcoded interaction frames, source-relative wrist angle clamps, strict extra
collision level or automatic residual-driven reruns. Only the robot's physical
joint limits are hard bounds. Scene target identity is explicit input.

The soft level balances source posture (stronger wrist prior), original hand
orientation, foot yaw, scene clearance and global first/second differences.
Velocity/acceleration weights scale with fps squared/fourth power, so the
physical smoothing preference is not tied to a 50 Hz take. Mild overlap or
posture mismatch is accepted and reported rather than spawning another fix.
No finite optimization can guarantee every unreachable or contradictory input
succeeds; report residuals after the fixed iteration budget.

This adapter currently supports G1, a synchronized human world sidecar, one
stationary interaction body and a planar support floor with stance/swing gait.
It is not a general kneeling, stair, moving-object or finger-grasp solver.
The mathematical trajectory hierarchy is independent of episode identity;
transfer beyond these input assumptions requires an adapter, not per-frame
patches. General behavior is covered by synthetic grounding, window coupling,
orientation-nullspace and conflicting soft-objective tests; one real clip does
not establish generalization.

## Continuous EEF weight schedule

Targets exist for both hands throughout the recording. The raw 15 cm detection
mask is metadata, never a residual/Jacobian gate. Default least-squares weights
are 1 away from interactions and 1000 inside them; Gaussian shoulders use a
fixed 0.3 second scale before/after each interval. Multiple intervals combine
their envelopes without inserting take-specific frame rules. The CLI exposes
`--eef-baseline-weight`, `--eef-window-weight`, `--eef-transition-seconds`.
The archived window NPZ includes both hands' weight arrays.

The CLI protects P0 feet exactly and solves weighted P1 EEF, P2 posture/collision
and temporal costs together below P0. These are weighted soft priorities; P1 is
no longer a strict exact per-frame constraint. This permits deliberate EEF error
versus smoothness tradeoffs and makes the weight schedule effective.
