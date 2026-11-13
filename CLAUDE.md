# beverage-disposal

Context file for anyone (human or AI assistant) working in this repository. It is committed,
public, and must stay accurate: see "Maintenance rules".

## Purpose

A simulated SO-101 arm learns, with reinforcement learning (RL), to pick up a beverage container and
move it from A to B. A coded sequencer around that learned skill decides where a container goes: a
practically empty one (under 30 mL) goes into a bin, anything else, or anything unsure, goes to a
separate set-aside area on the table. The fullness decision is a tested rule, not learned: a weighing
pad plus a database of empty weights for known container types, and a CoinFT tilt test for unknown
ones. Containers: the 12 fl oz can first, then the 16.9 fl oz (500 mL) PET bottle. Tiers: 1 =
simulation with statistics, 2 = a real arm, 3 = wider containers and more sensors.

- **MuJoCo** simulates the physics and is where all training happens.
- **Gazebo + ROS 2** validates the trained policy on a *different* physics engine (a real
  sim-to-sim transfer test) through a ROS 2 policy node.
- This is a ground-up rebuild of an earlier scripted version, which faked the grasp by
  teleporting the can to the gripper. Here the grasp must be physical.
- The owner drives every decision and must understand every file. Understanding beats speed.

## Current status

- **Iteration 0 (foundations):** 0.1 to 0.5 (repo skeleton, uv environment and pytest tiers,
  ruff and pre-commit, living docs and staleness tests, vendored SO-101 model) are built
  test-first, with red output recorded before each piece of code; `main` is that history. The
  earlier first build, written without a red-first record, is kept on the branch `first-build`.
  The tests are organized into topic folders. 0.6 (step and render smoke test, with a viewer
  script) is built the same way. The narrow 0.7 (headless Gazebo Harmonic in a Docker container on the
  development Mac) is built test-first and merged: the image builds from
  `docker/gazebo/Dockerfile` (base image pinned by digest, installed package versions recorded), and
  a falling ball steps in Gazebo to the same 0.0495405 m drop at 0.1 s as in MuJoCo. A ball landing on the
  ground and a block on a slope also agree with physics formulas and with MuJoCo, and the measured differences
  between the engines (impact, friction combination, static friction) are kept for 7.5.
- **Iteration 1 (kinematics and end-effector control):** named by the owner on 2026-09-25 and built
  one sub-iteration per session, each on its own branch (rule 9). 1.1 (model tests and the TCP site) is
  built test-first and merged into `main`: a loader adds the TCP site to the vendored model, and MJCF and
  URDF are tested to agree on link poses and masses. 1.2 (forward kinematics) is built test-first and
  merged into `main`: `beverage_disposal.core.kinematics.fk` turns the five arm joint angles into the
  4x4 TCP pose with NumPy alone, reading the generated, frozen chain file `core/so101_chain.json`; it
  matches MuJoCo's TCP pose to 1e-9 over 1000 random poses. The kinematics tests live in
  `tests/kinematics/`.
- **Iteration 1, 1.3 to 1.6: built test-first and merged into `main` on 2026-10-14**, one merge commit
  per branch: `iter1/1.3-jacobian`
  (a 4x5 `jacobian` over the five arm joints, rows x, y, z, pitch; pitch is the signed in-plane angle of the
  TCP z axis), `iter1/1.4-numerical-ik` (`core/ik.py`: damped least squares, tip position first, pitch in
  the exact null space, joints parked on their limits; `mink` and a closed-form planar solution as test
  oracles), `iter1/1.5-ee-delta-controller` (`core/controller.py`), `iter1/1.6-closed-loop-pd` (a 20 Hz
  MuJoCo loop, step-response metrics and a stored reference file for matching Gazebo later). Fast and medium
  tiers pass on the merged result (the Gazebo tier was not run). Found on the way: straight-down
  pitch together with a tip position is reachable only below about z = 0.09 m, so a can cannot be grasped
  top-down at its middle.
- **Iteration 2, 2.1a (minimal scene): built test-first on 2026-11-13** (`core/config.py`, `core/containers.py`,
  `sim/scene.py`, `configs/scene.toml`, `tests/scene/`): a table, one can (settles at half height, no creep under a
  small push) and the arm in one model; solver settings elliptic cone and impratio 10 because the default let the
  can creep. **2.1b (layout) built the same day** (`core/layout.py`, `core/reach.py`, bin, pad, set-aside marker and
  spawn box in `configs/scene.toml`): a reach check requires every pick, pad, set-aside and bin pose to be reachable
  at 60 degrees of pitch or more; top-down is reachable only to about x = 0.26 m at the can's middle height, which
  sized the spawn box (0.12 m square) and the bin walls (0.06 m). **2.4 (grasp physics) built the same day** (`sim/pads.py`, `sim/grasp.py`, `tests/grasp/`): the vendored jaws could not hold
  the can, so two flat pads replace the jaw collision, the wrist rolls 90 degrees and the can is taken near its rim (0.10 m);
  a prescribed grasp lifts empty and full cans at normal friction, with two strict, documented limits (a full can at half
  the friction; softer contacts). **2.5 (scripted pick-and-place) built the same day** (`core/interfaces.py`, `core/scripted.py`, `core/baselines.py`,
  `sim/rollout.py`, `tests/scripted/`): a nine-phase script puts an empty or a full can on the pad, in the bin or in the
  set-aside area from 12 of 12 seeded spawns each, at 0.6 N m of squeeze; a random policy and a never-squeeze script score
  zero, and perception noise of 5 mm breaks it (8 of 8 at 2 mm). **2.3 (recorder and task tests) built the same day** (`core/trajectory.py`, `core/task_tests.py`, `sim/recorder.py`,
  `tests/tasktests/`): six pass-fail tests and five indicators on a recorded episode, with thresholds measured on 24 scripted
  episodes. **2.2 (container registry) built the same day** (`core/registry.py`, `configs/containers.toml`, `tests/containers/`): container
  types are data tables with provenance, the scene names one, and an unknown type is met with a request to enrol it. **2.6a (pad and weight rule) built the same day** (`core/pad.py`, `core/fullness.py`, `core/fullness_eval.py`,
  `core/statistics.py`, `configs/fullness.toml`, `tests/fullness/`): a pad model, a rule that bins under 26.5 g net (30 mL at 0.95 g/mL
  less a 2 g margin), Wilson bounds, and a Monte Carlo: in simulation, 0 of 100 000 binned at 30, 31 and 35 mL of the lightest
  drink (bound 3.8e-5); the sweep breaks at about four times the nominal noise. **2.6b (statics wrench) built the same day** (`core/statics.py`, `tests/statics/`): the liquid's torque on a tilted grasp from a
  rigid-cylinder model with closed-form checks; at 60 degrees about 0.6 mN m per mL, and the plausible parameter errors move 30 mL's
  torque by about 4 mL's worth (model M, unvalidated). **2.6c (tilt routine) built the same day** (`core/tilt.py`, `core/coinft.py`, `core/tilt_config.py`, `configs/tilt.toml`,
  `tests/tilt/`): a CoinFT model and a tilt verdict that bins an unknown container only if the volume bound is under 30 mL (about 14 mL
  of margin with the guessed sensor, so cans under about 16 mL), with a pad-only fallback; 0 of 500 binned at 30, 31 and 35 mL.
  Computed, not run through MuJoCo. **2.7 (safety supervisor) built the same day** (`core/supervisor.py`, `core/supervised.py`, `core/safety_config.py`, `configs/safety.toml`,
  `tests/safety/`): checks for a too-wide or too-heavy container, an unknown type, a failed test lift, arm force and gripper torque, bad or
  missing observations, timeout and a moved container, all reported by light, sound and message; a check that throws is a stop. Faults injected
  into a running episode (NaN, dead sensor, no finish, throwing supervisor, NaN action) all end in a safe stop. **2.8 (sequencer, harness, results file, demo) built the same day** (`core/sequencer.py`, `core/acceptance.py`, `sim/skills.py`, `sim/harness.py`, `sim/demo.py`, `scripts/run_acceptance.py`, `scripts/record_demo.py`, `tests/sequencer/`): the sequencer carries a container to the pad, weighs it, decides and carries it on, under the supervisor; the harness runs it over a volume grid with seeded runs and writes a results file (commit, config hash, versions, seeds, counts). First run (can, known type, 381 trials at each of 0, 30, 31 and 35 mL, scripted skill, git cfc6544 plus uncommitted 2.8 files): in simulation, 0 of 381 binned at each of 30, 31 and 35 mL (Wilson 95 % upper bound 0.998 %); 342 of 381 empties binned (the other 39 were set aside or stopped; the summary does not split them yet); pick-and-place completed in 1389 of 1524 runs (lower bound 89.6 %, just under the 90 % target; one found cause: seed 4 at 30 mL times out on the first leg). Not run: the unknown-type (tilt) grid and the bottle. Iteration 2 is built continuously from 2026-11-13 with no
  teaching pauses (the owner studies separately): 2.1a, 2.1b, 2.4, 2.5, 2.3, 2.2, 2.6a, 2.6b, 2.6c, 2.7, 2.8. The
  order comes from the iteration 2 plan review (2026-10-29) and keeps the full scope, tilt routine included.
  2.1 is split into the minimal scene (table, can, one config home for container size and mass) and the layout
  (bin, set-aside area, pad, reach map). 2.6 is split into the pad model with the weight rule, the statics model
  and the tilt routine. 2.2 is data plus enrolment logic. 2.7 adds fault-injection tests that must end in
  set-aside or a safe stop. The "binned at most 1 %" promise is tested as a measured miss rate per volume bin
  (at least 381 trials in the bins near 30 mL for a zero-miss 95 % Wilson bound below 1 %), simulation only, and
  is worded "in simulation, k/N misses, Wilson 95 % upper bound X %". Config is frozen dataclasses plus TOML read
  with stdlib `tomllib`. The run is a work in progress by design; it stops at the end of iteration 2.
- **Plan review closed (2026-10-13).** Every decision was re-examined against the real alternatives. The
  fullness decision became a coded rule (weighing pad and database, CoinFT tilt for unknown containers, a
  30 mL promise), RL narrowed to one pick-and-place skill, the bottle was added after the can, and a real-arm
  tier (iteration 9) and headline metrics were added. The Locked decisions and Roadmap below are the result.
- Iterations 2 to 9 are planned in detail down to sub-iterations (see Roadmap) and not started.
- Last updated: 2026-11-13 (2.8 built; iteration 2 complete). Update this section in the same change as every sub-iteration.

## Locked decisions

Each label is a stable name; the owner's private decision journal uses the same labels.

- **RL end to end:** the policy is learned with RL, not scripted. A scripted controller exists
  only as a solvability proof and baseline.
- **Simulator roles:** MuJoCo for physics and training (headless, plain Gymnasium env, no ROS
  in the training loop). Gazebo for validation, running its own physics.
- **ROS-free core:** IK, the end-effector controller, the observation builder and the task test
  functions live in a plain-Python core library used by both the Gymnasium env and the ROS
  policy node, so they cannot drift apart. ROS is only an adapter at validation time.
- **Training hardware:** develop and debug on an Apple Silicon Mac (CPU); rent a cloud GPU for
  long runs once the environment is stable.
- **Platform (validation):** Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic, run headless in a Docker
  container (arm64 on the development Mac, reproducible from a Dockerfile). No VM and no Gazebo GUI:
  MuJoCo is where the arm is watched, and Gazebo is judged by numbers.
- **Tooling:** Python 3.12 (pinned `>=3.12,<3.13`), `uv` (ranges in `pyproject.toml`, exact
  versions in `uv.lock`, hatchling build backend, src layout), pytest with fast/medium/slow
  tiers, hypothesis, ruff (rules E, F, I, UP, B; 100 columns) and pre-commit.
  Versions locked 2026-09-24: mujoco 3.14.0, numpy 2.5.3, gymnasium 1.3.0, pytest 9.1.1,
  hypothesis 6.168.1, ruff 0.16.9, pre-commit 4.6.2; mink 1.3.0 added in 1.4.
- **Arm model:** official SO-101 MJCF/URDF (TheRobotStudio/SO-ARM100, "new" calibration),
  vendored untouched under `assets/so101/` with a provenance file (upstream commit
  `5f6d2b876a53a4872e405b991dd925556c9e38a4`, file checksums) and a NOTICE. Those files stay under
  their own Apache-2.0 license, so the repository is not purely public domain. Our loader
  (`beverage_disposal.sim.arm.load_arm`) reads the MJCF through MuJoCo's `MjSpec` editing API and adds
  what is missing (the TCP site in 1.1): plain MJCF cannot add a child to a body defined in an included
  file, and the vendored files are never edited. MJCF is primary for training; URDF is for Gazebo;
  consistency tests compare names, limits, link poses (within 10 um over random poses) and link masses.
  Facts about the vendored model, all tested: it has no floor, light or camera and uses MuJoCo's
  defaults (timestep 0.002 s); the joint stops hold; the gripper touches the shoulder when the elbow
  folds to about 1.5 rad, so the elbow's upper range limit (1.69 rad) is not reachable; and a fresh
  `MjData` renders blank until `mj_forward` (or a step) has computed the body positions. Upstream's
  `gripperframe` site sits on the fixed jaw's inner face, 7.9 mm off the wrist-roll axis; at gripper
  angle 0 the moving jaw's face mirrors it, so the jaws are 15.8 mm apart there. Every arm joint turns
  about its own body's z axis, and the shoulder-pan axis points straight down in the base frame, so a
  positive pan angle turns the arm clockwise seen from above.
- **Observations (for the pick-and-place skill):** state-based first, designed so a camera can be added
  later. Measured joint positions (no velocities, which are too noisy on the real servos), commanded
  joint targets, tip position (our FK), container position relative to the tip, the vector to the goal B
  (an input), previous action, time remaining, and a short frame-stacked history of positions and the
  previous action. No joint load and no destination flag: the skill never decides the destination. Fixed
  physical-range scaling (no running statistics).
- **Action space:** `Box(-1, 1, (4,))` = end-effector delta `[dx, dy, dz]` plus gripper. Deltas
  scale to about 1.5 cm per step (tunable) and add to the *previous commanded target*, with a
  clamp on target-to-measured-tip distance. Gripper is an absolute target with a slew-rate
  limit (squeeze force is roughly kp times target-minus-actual).
- **Control rate:** 20 Hz policy rate, with frame skip over the physics steps.
- **Kinematics and IK:** pure NumPy FK/Jacobian/IK (no MuJoCo needed at runtime), oracle-tested
  against MuJoCo. Damped least-squares with a null-space projector and strict task priority:
  tip position first, top-down pitch second, wrist roll locked. Pitch is the angle of the grasp
  axis below horizontal, positive down (top-down = +90 degrees). TCP is the midpoint between the
  finger pads at gripper angle 0: on the wrist-roll axis, at the depth of upstream's `gripperframe`
  (98.1 mm from the gripper body origin), fixed to the gripper body; its z axis points out of the tool
  towards the fingertips and its x axis towards the moving jaw. A closed-form planar IK and the `mink`
  library exist in tests only, as oracles. The geometry reaches the NumPy side as a frozen JSON chain
  file (the robot spec: per-joint offsets, rotations and limits, the TCP, the home pose), generated
  from MuJoCo by a script and checked for staleness; the generator refuses any joint that is not a
  hinge about its body's z axis through the body origin. FK works in the arm's base frame and returns
  the full 4x4 TCP pose. Home pose: every joint 0 except wrist flex at +90 degrees, which holds the TCP
  at about (0.232, 0.000, 0.075) m, pointing straight down.
- **Actuators:** start from the servo data in the official files, verify by step-response tests
  (rise time, overshoot, settling, steady-state error), store the reference numbers so Gazebo can
  be matched to them.
- **Grasp:** physical only. Tune pads, friction and solver settings within plausible physics;
  no weld or teleport fallback. Our own two pads replace the jaw meshes' collision in the scene (2.4), the wrist
  rolls 90 degrees and the can is taken near its rim; the gripper servo's torque limit (0.6 N m in the scene)
  sets the squeeze.
- **Sequencer and skill:** RL learns one skill, "pick at A, place at B" (B is an input), through its
  motion under contact. A plain, tested sequencer calls it twice (container to the weighing pad, then pad
  to the bin or the set-aside area) around the fullness rule. RL never decides the destination.
- **Fullness (task rule):** bin only if confident the container holds under 30 mL; if unsure, set aside;
  no capacity estimate. For every volume from 30 mL up, the container is binned at most 1 % of the time
  (a worst-case promise per volume; brimful containers are left to the weight check). A container of a
  known type is decided by its weight on the pad against the type's lightest plausible empty weight
  (bin only if the weight above it is under 30 g minus the scale's error margin). An unknown container
  heavier than a material-dependent limit is set aside; otherwise a coded CoinFT tilt test runs with
  the opening on the high side and the same strict threshold applies, and the robot asks the owner to
  enrol the type. The tilt routine is coded, not learned; RL only brings the container to its start pose.
  The set-aside area is separate from the pad. Tests always know the true volume.
- **Containers:** the 12 fl oz can first, then the 16.9 fl oz PET bottle (right after the can works). A
  container type is data in a registry (dimensions, lightest empty weight, grasp height and width, cap and
  tab rules); no logic branches on a specific container. One enrolled type covers every container of
  that type. For a bottle the camera must confirm the cap is on, else it is set aside.
- **Safety supervisor:** coded and tested; refuses or stops on a container that is too wide or too heavy,
  a failed test lift, or a force limit, and reports by light, sound and message.
- **Perception:** markerless camera: detect the container, recognise its type, find the tab, check a
  bottle's cap. Simulation uses ground truth plus a noise model (position noise, bias, delay,
  dropouts). Material and cap recognition for other containers is Tier 3.
- **Weight and tilt signals (replace servo-load weighing, which a study showed fails):** a weighing pad
  (a MuJoCo force sensor with noise and drift) and CoinFT force/torque pads on the gripper. Servo
  load is not an observation.
- **Container model:** rigid body plus a precomputed statics "liquid wrench" for the tilt test, checked
  occasionally against OpenFOAM. Container mass is a randomized range, not two classes.
- **Reward:** sparse success bonus (container released at B, upright, within tolerance, not spilled) plus
  potential-based progress shaping, small regularizers (weights tuned in 4.3), failure penalties (drop,
  tip-over, off the table, hard joint or table contact), early termination on those failures and on
  timeout. Adversarial-policy tests (hover, wave, push the container off, drop it, do nothing) must all
  score below the scripted controller.
- **Acceptance ("done") and metrics:** Tier 1: at least 300 trials per volume so that zero binned
  containers of 30 mL or more shows the 1 % promise at 95 % confidence (rule of three), and a pick-and-place
  success lower bound of 90 % over at least 5 seeds. Tier 2 (real arm): 30 trials per container type with
  a Wilson interval, no hard pass bar. The evaluation harness writes a results file (commit and seeds
  recorded) holding the headline metrics: pick-and-place success with a bootstrap lower bound, binned
  30 mL+ containers per volume, empties correctly binned, the MuJoCo-to-Gazebo gap, steps to 90 %
  success, success under the widest passing randomization, engineering numbers (tests, mutants killed,
  FK error, steps per second) and, if reached, the real-arm trials.
- **Architecture and code standards:** modules for perception, sequencer, the RL skill, the fullness
  rule and the safety supervisor; interfaces and data contracts (typed dataclasses and Protocols) are
  fixed first and their internals built test-first; core code is ROS-free and MuJoCo-free; mypy on
  `core/` only, ruff everywhere; no magic numbers (named constants or TOML); comments only where the
  code is complex or has a reason.
- **TdRL later:** pass-fail and indicative test functions define the task and drive a
  hand-shaped reward first. The learned-reward method (Test-driven RL, AAAI-26) is a later
  comparison against that baseline.
- **Algorithm:** SAC via Stable-Baselines3 (owner's decision, 2026-09-24; supersedes the earlier
  "working choice pending a PPO bake-off"). No PPO comparison is planned. PPO stays available as
  a fallback only if SAC cannot be made to work, and the training config keeps the algorithm a
  swappable setting. Because SAC is settled, the TdRL question (TdRL builds on SAC) is open for
  iteration 6. Reason given by the owner: SAC is more standard in robotics applications.
- **Policy deployment:** the trained policy is exported to ONNX and run with ONNX Runtime in the
  ROS policy node (no PyTorch on the ROS side); tests compare PyTorch and ONNX outputs. How the
  node reaches the simulator (thin node over standard ros2_control controllers vs a controller
  plugin) is decided at 7.3. The TdRL integration path is decided at 6.1.
- **Curriculum:** stages S0 reach, S1 grasp and lift, S2 place at a fixed B, S3 random B plus physics,
  then the randomization stages (S4, S5), promoted automatically when the lower confidence bound of
  evaluation success passes a threshold (unit-tested scheduler). Demonstrations or DORAEMON only if a
  stage stalls. The 4.3 run: SAC, at least 5 seeds, bootstrap intervals.
- **Tracking:** frozen dataclass configs read from TOML (stdlib `tomllib`, no new dependency; changed from
  YAML on 2026-11-13) and TensorBoard. Hydra and Weights & Biases are noted
  alternatives to revisit when sweeping.
- **Robustness (iteration 5, aimed at the Gazebo sim-to-sim test):** match what can be measured
  in Gazebo (for example actuator step responses) and randomize the residual uncertainty.
  Randomization ranges start narrow and widen automatically under the same confidence-bound
  promotion rule as the curriculum (stage S5). Container mass is a continuous range and the liquid
  amount is continuous (no two-class gap). Physics randomization covers friction, container mass and
  size, joint damping/friction/armature, actuator gains, gripper force limit and contact softness;
  sensor and timing randomization covers pad and CoinFT noise/drift, perception noise/bias/delay,
  encoder resolution and observation/action delay. Ranges
  come from a sensitivity study, and a "sim-to-sim gap budget" doc lists each MuJoCo/Gazebo
  difference and whether it is randomized, matched or accepted.
- **License:** The Unlicense. Vendored third-party files keep their own licenses.
- **Repo policy:** since 2026-11-13 assistants commit on a sub-iteration's branch and merge it into `main`
  with `git merge --no-ff`, staging named publishable files only; the owner reviews the history afterwards.
  Assistants never push. (Before that date the owner made every commit.)

## Process rules

1. **Iterations and sub-iterations.** Each iteration delivers a milestone; each has several
   sub-iterations.
2. **No pauses in the iteration 2 run.** Since 2026-11-13 the owner studies separately, so the
   sub-iterations of iteration 2 are built one after another without stopping to teach. The run stops
   at the end of iteration 2, or earlier when a decision belongs to the owner. Each sub-iteration still
   ends with the tests green, the docs updated and a commit.
3. **Strict test-driven development.** Write the tests first and watch them fail for the right
   reason, then the minimum code, then green, then refactor. RL-specific layers are in
   "Testing layers" below. The method follows the `test-driven-development` skill from
   obra/superpowers: no production code without a failing test first; every test names the break
   it catches; expected values are derived independently of the code under test; test behavior,
   not text; run the whole suite before calling anything done and report every failure.
4. **Standards first.** Prefer industry-standard tools and practices even when more complex, and
   explain what each one is and why it was chosen over the alternatives. Verify "standard"
   claims against current docs, not memory.
5. **Be honest about uncertainty.** Mark unverified facts as unverified; never claim something
   works unless it was run.
6. **Assistants commit on branches.** Each sub-iteration is committed on its own branch and merged
   with a merge commit (see Repo policy); the owner reviews the history and may revert any merge.
7. **Parallel work uses git worktrees.** Independent tasks run at the same time by separate
   agents, each in its own git worktree on its own branch (for example `iter1/fk`), created in a
   sibling directory next to the repository, never inside it. Each worktree has its own uv
   environment (`uv sync`). The main checkout stays on `main`. Work that depends on another
   task's code is not parallelized. Untracked local files exist only in the main checkout, so a
   worktree only sees what is committed. Agents leave their changes uncommitted until the owner
   says go, then commit on the task's branch only; the owner merges into `main`. To keep parallel
   branches from conflicting, only the coordinating session edits this file and the owner's
   local notes, after merges; other agents list the doc changes they need in their report.
8. **Comments read like a person wrote them.** Explain why, not what: a reason, a gotcha, a unit
   or a source. Skip comments that restate the code, skip essays, and skip banner decoration.
   One or two natural sentences is usually enough, and plenty of code needs none.
9. **Branches, never `main`.** No direct work on `main`. Each sub-iteration is a branch cut from
   `main` and named `iter<N>/<N.M>-<slug>` (`docs/`, `fix/` or `chore/` for other work), worked in
   the main checkout; worktrees are for parallel tasks only (rule 7). The owner merges into `main`
   with a merge commit (`git merge --no-ff`) so a sub-iteration stays one visible unit. A local
   `no-commit-to-branch` pre-commit hook refuses commits on `main`; it only takes effect after
   `uv run pre-commit install`, and it does not block the owner's merge commits (a merge uses a
   different git hook, tested).

### Testing layers

1. Pure unit tests (kinematics, observation builder, action clipping).
2. Model tests (MJCF loads, MJCF and URDF agree).
3. Gymnasium contract and determinism.
4. Solvability: a scripted controller passes; a random policy scores about zero.
5. Task-test functions checked on hand-built trajectories.
6. Learning smoke tests (probe environments, overfit one configuration).
7. Statistical acceptance: at least 5 (better 10) seeds, bootstrap confidence intervals.
8. Sim-to-sim: the same acceptance tests run in Gazebo.

Tiers: fast (seconds, every change), medium (minutes, before finishing a sub-iteration),
slow (hours, on a rented GPU when needed).

## Maintenance rules

- **Never let this file go stale.** Update it in the same change as any decision, requirement,
  architecture change, new command or status change.
- **Living docs.** The owner keeps local, untracked study notes: a decision journal (the owner's
  decisions and reasoning), one file per iteration covering every sub-iteration, "why we chose
  X over Y" notes, a resume-ready tech-stack list, a resources list (videos and articles), and a
  testing-strategy note. They are updated in the same change as the thing they describe, and
  local checks look for staleness. These notes are not in the public repo, and neither are
  their names or locations: those are described only in the owner's private context file.
- **Private notes.** Anything private goes in an untracked private context file that Claude
  Code loads alongside this file. Nothing private belongs here: this file must be enough to run
  the project on any machine. A public test checks it for absolute home paths and email
  addresses, and a local, unpublished check confirms that no published file names a private
  path.
- **Local-only ignore rules.** Ignore rules for the owner-only paths live in each clone's local
  exclude file (inside the git metadata folder), never in the published ignore file, so that
  no pushed file names them. The exact rules are recorded in the private context file.
- **Edits to shared files.** Re-read this file and the decision journal before editing and use
  targeted edits, never a full overwrite.

## Revisit triggers

Decisions and ideas to reopen at a specific point:

- Probe environments and observation/action normalization contract test: iterations 3 to 4.
- Seeds, confidence intervals, AdaStop for A-vs-B comparisons: iterations 4 and 6.
- Regression suite of fixed seeded scenarios: iterations 2 to 5.
- Whether to use TdRL on top of SAC: iteration 6. Reopen the SAC choice (PPO fallback) only if
  SAC cannot be made to learn in iteration 4.
- Distributional critic (TQC or Distributional SAC): train plain SAC first as the baseline. If 4.3
  or 4.4 shows unstable Q-values or large seed-to-seed variance, run a distributional variant as
  a controlled A/B against that baseline. Not started before then: the RL-algorithm-selection
  paper (arXiv 2407.20917) ranks distributional methods low priority when risk estimation is not
  needed. Whether SB3 or `sb3-contrib` provides a usable one is unverified. Squint (below) is a second
  source: a C51 distributional critic was one of its most useful changes on this exact arm.
- Two open points from the same paper, to check at 4.1: n-step returns instead of one-step TD
  (favored for sparse rewards or a violated Markov assumption), and seeding the replay buffer
  with scripted-controller demonstrations (the paper warns this can hurt without care).
- Squint (arXiv 2602.21203, 2026: SAC with a distributional critic, trained in ManiSkill3, zero-shot on
  the real SO-101 in 73 of 80 trials, with cans among its tasks). The owner wants its techniques used where
  possible. The owner's local research note (location in the private context file) maps each insight to
  the sub-iteration it affects; read that row before planning 2.1, 2.3, 3.1, 3.2, 3.4, 3.5, 4.1, 4.3, 4.4,
  5.2, 5.3, 5.4, 6.1, 7.2 to 7.9, and iteration 8. Code: https://github.com/aalmuzairee/squint (MIT).
- Range or ultrasound fullness sensor for varied containers (glass bottle, plastic bottle, metal flask,
  can): an idea the owner is interested in pursuing, perhaps in parallel with the main line, as a
  fallback if the weighing pad and the CoinFT tilt test are not enough to tell how much liquid a
  container holds. Parked, not started (the HC-SR04 veto is Tier 3). MuJoCo (`rangefinder`, checked in
  3.14) and Gazebo both only ray-cast, so the sensor model would live in the ROS-free core like the pad
  model, with MuJoCo as the natural place to train it. Real ultrasound is the hard part to model: beam
  cone, container opening, foam, through-wall coupling, and level versus volume for unknown container
  shapes. Natural homes: 2.6 (fullness) or after iteration 8.
- Asymmetric critic (the critic sees the true container mass during training): no longer needed for the
  decision; kept as an optional experiment for the grasp, for iteration 6 or after 4.4.
- CoinFT bench test (a soft gate): CoinFT is built in parallel with the main line. If it fails, fall back
  to the weighing pad alone. Revisit the 1 % promise only if the bench shows about 0.3 to 0.5 mN m of
  stop-to-stop error (a 5 % promise would then roughly halve the empties set aside).
- Sim-to-sim test protocol (none published; ours to define): iteration 7.
- Failure search and metamorphic testing: after iteration 5.
- Bridge choice (thin node over standard controllers vs a ros2_control controller plugin;
  `mujoco_ros2_control` only as the 7.4 integration test): decide at 7.3.
- `mink` library instead of our own IK: only if joint-limit or singularity problems appear. Until then
  mink is a test oracle only (added as a dev dependency in 1.4, test-first); it needs MuJoCo at runtime, so
  it cannot live in the ROS-free core.
- Hydra / Weights & Biases: when sweeping many runs.
- Scripted-controller demonstrations for imitation-assisted RL: only if the grasp stage stalls.
- A small C++ module for the pad pipeline (filter, tare, debounce, threshold, fail-safe state machine; no
  dynamic allocation, unit tested, called from Python): an embedded-skills idea saved for after iteration 2,
  once 2.6a works in Python. Not started.

## Commands

Requires [`uv`](https://docs.astral.sh/uv/) and a Python 3.12 (uv will find or fetch one).

```bash
uv sync                      # create .venv from uv.lock (exact, reproducible versions)
uv run pytest                # default run: fast + medium tiers (slow is excluded)
uv run pytest -m fast        # quick loop, run on every change (about 45 s today)
uv run pytest -m slow        # slow tier only (for example training runs)
uv run pytest tests/tooling  # one area of the tests
uv lock --check              # is uv.lock consistent with pyproject.toml?
uv run ruff check .          # lint (reports only; nothing is auto-fixed)
uv run ruff format .         # format
uv run pre-commit install    # once per clone: run ruff and the fast tier before every commit
uv run python scripts/view_so101.py                   # interactive viewer of the SO-101
uv run python scripts/view_so101.py --snapshot a.png  # picture of the arm, no window
uv run python scripts/generate_so101_chain.py        # rewrite the frozen chain file after a model change
uv run python scripts/run_acceptance.py --trials 381   # sequencer grid in simulation, writes runs/acceptance.json (about 45 min)
uv run python scripts/record_demo.py                   # GIF of one container through the sequencer, runs/demo.gif
uv run python scripts/measure_actuators.py         # re-measure the servo step responses (configs/actuator_step_response.json)
uv run pytest tests/gazebo   # Gazebo image tests: need a running Docker daemon, else they skip
```

Dependencies: ranges in `pyproject.toml`, exact versions in `uv.lock`. Runtime: `mujoco`,
`numpy`, `gymnasium`. Dev group: `pytest`, `hypothesis`, `packaging`, `ruff`, `pre-commit`,
`pillow`, `mink` (the IK oracle of 1.4; it pulls in `qpsolvers`, `daqp` and `scipy`). A test
fails if the installed environment drifts from `uv.lock`, and another fails if the "Versions
locked" sentence above disagrees with it, so update both together when a dependency changes. To add or upgrade a dependency, edit `pyproject.toml`, run
`uv lock` then `uv sync`, and commit both files.

Test tiers: an unmarked test is `fast`; a test may carry at most one of `fast`, `medium`,
`slow`; misspelled markers are errors. The pre-commit hooks run ruff check, ruff format, the
fast tier (an unmarked test counts as fast, so that is most of the suite) and the `main` guard. The format hook
rewrites files and stops the commit so you re-stage them.

## Layout

```
src/beverage_disposal/   Python package (placeholder name)
  core/                    the ROS-free core (no MuJoCo): sequencer.py (pick, weigh, decide, place), acceptance.py (results summary), supervisor.py (safety checks and indicators), supervised.py (a controller
                           under the supervisor), safety_config.py, tilt.py (the tilt verdict and pad-only fallback), coinft.py
                           (torque sensor model), tilt_config.py, statics.py (the liquid wrench of a tilted container), pad.py (weighing pad model), fullness.py (the weight rule),
                           fullness_eval.py (Monte Carlo and sweeps), fullness_config.py, statistics.py (Wilson bounds), registry.py (container types as data, enrolment), trajectory.py and task_tests.py (an episode record and the
                           task's pass-fail and indicative functions), interfaces.py (Observation, Controller, noise), scripted.py
                           (the scripted pick-and-place), baselines.py (random policy); config.py (frozen scene dataclasses read from TOML),
                           containers.py (ContainerSpec, provenance, liquid mass), layout.py (footprints, spawn sampler, set-aside test),
                           reach.py (reach check of the layout); kinematics.py (FK, Jacobian, pitch) and
                           so101_chain.json, the generated, frozen robot spec it reads; ik.py (solve_ik);
                           controller.py (EEDeltaController); step_response.py (rise, overshoot, settling)
  sim/                     the MuJoCo side: arm.py loads the vendored arm and adds the TCP site; scene.py (table, bin, pad, set-aside
                           marker, container and the arm in one model); pads.py (the gripper's two collision pads);
                           grasp.py (prescribed grasp); skills.py (the scripted skill as the sequencer calls it), harness.py (seeded grid and results record), demo.py (GIF frames); rollout.py (one judged episode); recorder.py (an episode as a Trajectory); loop.py
                           (20 Hz closed loop through the PD actuators); actuator_response.py (step tests)
docker/gazebo/           headless Gazebo validation image: Dockerfile, worlds/, step_world.py,
                         apt-versions.txt (recorded package versions, arm64)
tests/                   pytest tests (fast/medium/slow tiers), one folder per area:
  repo/                    what git publishes: ignore rules, top-level allowlist, no private text
  environment/             Python pin, dependencies, lockfile, third-party libraries
  tooling/                 pytest tiers, ruff, pre-commit hooks
  gazebo/                  the Docker image (builds, platform, pinned and recorded versions), the
                           falling ball, ground and slope checks against physics formulas and MuJoCo,
                           and proofs the tests can fail (medium tier)
  model/                   the vendored SO-101 model: files, stepping, joint stops, rendering,
                           viewer script, MJCF/URDF agreement; our arm model and its TCP site
  kinematics/              FK against MuJoCo, the home pose, pan and roll properties, the core without
                           MuJoCo, the chain file's generator and staleness, and their can-fail proofs
  control/                 the end-effector delta controller and the closed loop through the PD actuators
  sequencer/               the sequencer (ideal and simulated skill), the results summary, the harness and the demo recorder
  safety/                  the supervisor's checks with near-misses, the indicators and faults injected into an episode
  tilt/                    the CoinFT model, the tilt verdict, the volume bound and the pad-only fallback
  statics/                 the liquid wrench: closed-form checks, spill angle and the sensitivity sweep
  fullness/                the pad model, the weight rule, the Wilson bound and the false-bin Monte Carlo
  containers/              the registry, its weights with sources and the enrolment of unknown types
  tasktests/               the recorder and the task's pass-fail and indicative functions
  scripted/                the observation, the scripted pick-and-place, the goals and the baselines
  grasp/                   the gripper pads and the prescribed grasp, with its measured limits
  scene/                   the scene config and container record, the can on the table, the jaw opening, the
                           layout (bin, pad, set-aside area, spawn box) and its reach check
  support/                 shared helpers only (paths, git, scanning, the SO-101 facts and the
                           damaged-copy and mutated-package harnesses); no tests
  conftest.py              tier hook and shared fixtures
assets/                  vendored third-party files with provenance (the SO-101 model)
configs/                 configs (scene.toml, containers.toml, fullness.toml, tilt.toml, safety.toml; training later) and the stored actuator step-response reference (JSON)
scripts/                 helper scripts (view_so101.py, generate_so101_chain.py, measure_actuators.py, run_acceptance.py, record_demo.py)
pyproject.toml  uv.lock  .python-version   uv project, exact lockfile, pinned Python
.pre-commit-config.yaml                    git hooks (ruff, fast tests)
README.md  CLAUDE.md  LICENSE  .gitignore
```

Planned modules, not built yet (the tree above is what exists): `core/` gains the IK and controller
(iteration 1), `containers.py` (type registry, empty-weight database, enrolment), `fullness.py` (the rule),
`tilt.py` (the coded tilt routine and the statics model), `sequencer.py`, `supervisor.py`, `observation.py`
and `interfaces.py` (the Protocols and dataclasses that fix every boundary first); `sim/` gains the scene,
pad and CoinFT sensor models and the simulated perception; then `env/` (the Gymnasium environment for the
skill), `train/` (SAC configs, curriculum scheduler, evaluation harness, metrics writer), `ros/` (adapter
only: policy node, Gazebo bridge) and `hardware/` (Tier 2 drivers implementing the same Protocols).

A test allows only known top-level entries to be published (`PROJECT_TOP_LEVEL` in
`tests/repo/test_publishable_set.py`). When a legitimate new top-level file or folder appears, add
it there in the same change. Likewise only approved folders may sit directly in `tests/`
(`ALLOWED_IN_TESTS` in `tests/repo/test_tests_layout.py`): a new area of the project gets its own
folder and one line there, and shared helpers go in `tests/support/`, never loose in `tests/`.

Ignored by the published ignore file: `.venv`, `runs/`, caches, editor and OS files. Owner-only
files (study notes, private context file, unpublished tests) are ignored through the clone's
local exclude file and are not listed here.

## Roadmap

Each sub-iteration follows: failing test, minimum code, green, refactor, pause and teach,
update docs.

- **0 Foundations.** 0.1 repo skeleton; 0.2 uv env + pytest tiers; 0.3 ruff + pre-commit;
  0.4 living docs and staleness tests; 0.5 vendored SO-101 model + provenance; 0.6 step and
  render smoke test; 0.7 Gazebo platform spike (narrow, built).
- **1 Kinematics and end-effector control.** 1.1 model tests + tip site; 1.2 FK; 1.3 Jacobian;
  1.4 numerical IK; 1.5 EE-delta controller; 1.6 closed loop through PD actuators.
- **2 World, containers and the decision (no RL yet).** Build order: 2.1a minimal scene (table, can, config,
  container spec); 2.1b layout (bin, set-aside area, weighing pad, reach map, spawn sampler); 2.4 grasp physics
  (can); 2.5 scripted pick-and-place controller; 2.3 trajectory recorder + pass-fail/indicative tests; 2.2
  container registry, empty-weight database and enrolment logic; 2.6a pad sensor model and weight rule; 2.6b
  statics liquid wrench; 2.6c coded tilt routine and verdict; 2.7 safety supervisor with fault injection; 2.8
  sequencer + acceptance harness + metrics file.
- **3 Gymnasium environment for the skill.** 3.1 observations (with the simulated perception noise);
  3.2 action layer; 3.3 env core (goal B as input); 3.4 reward and termination; 3.5 wrappers and
  vectorization; 3.6 contract and baseline tests.
- **4 First learning.** 4.1 training scaffold + probe environments; 4.2 overfit one config;
  4.3 reach with randomization (SAC seeds and hyperparameter sensitivity); 4.4 grasp and lift;
  4.5 pick and place to B with the can. (The hidden destination flag, old 4.6, is removed.)
- **4B Bottle (after 4.5).** 4.7 bottle model, registry entry and grasp physics (a rigid simulated bottle;
  the real bottle flexes); 4.8 bottle in the scripted controller, the fullness entries and the cap check;
  4.9 the bottle skill (fine-tune, or one skill with the container type as an observation: decide at 4.7).
- **5 Robustness.** 5.1 randomization framework; 5.2 physics parameter set; 5.3 sensor and timing
  set; 5.4 sensitivity study; 5.5 train under randomization; 5.6 held-out perturbations.
- **6 TdRL comparison.** 6.1 read and choose the integration path; 6.2 map our tests to TdRL's;
  6.3 lexicographic comparator; 6.4 return and reward networks; 6.5 integrate with SAC; 6.6 the
  experiment; 6.7 write-up and decision.
- **7 Gazebo validation.** 7.1 platform; 7.2 policy export; 7.3 ROS policy node; 7.4 node against
  MuJoCo; 7.5 Gazebo world and model; 7.6 actuator matching; 7.7 Gazebo sensors; 7.8 sim-to-sim
  acceptance; 7.9 the gap loop.
- **8 Stretch.** Camera observations (end-to-end vision), optional.
- **9 Real arm (Tier 2, behind a soft gate).** 9.1 bring-up and calibration; 9.2 weighing pad, scale and
  database enrolment; 9.3 camera perception (detect, type, tab, cap; no tags on containers); 9.4 CoinFT
  bench and tilt on hardware (parallel track; the bench test is the gate); 9.5 the supervisor on
  hardware; 9.6 end-to-end trials; 9.7 results and write-up.

## Unverified or open

- Whether Gazebo cameras and lidar render in the container (software rendering; optional; not
  tried in 0.7). The base image is pinned by digest and every installed package version is
  recorded in `docker/gazebo/apt-versions.txt`, which a test compares against the image; the
  record is for arm64, so an amd64 machine needs its own, and a rebuild that pulls newer packages fails that test until the record is regenerated.
  Settled by 0.7 on this Mac: Jazzy's system Python is 3.12.3, and `ros-jazzy-ros-gz-sim`,
  `ros-jazzy-ros-gz-bridge` and `ros-jazzy-gz-ros2-control` install as arm64 binaries (Gazebo Sim 8.15.0).
- STS3215 servo parameters: the vendored MJCF uses kp 998.22, kv 2.731 and a +-3.35 N*m force
  limit (its class default says +-2.94, each actuator overrides it), while the upstream
  `joints_properties.xml` says kp 17.8 and kv 0. Their origin is a third-party README and is
  unverified; MuJoCo uses the MJCF values. Also unverified: real can dimensions (about 66 mm by
  122 mm), the joint ranges (two artifacts from one CAD export, not a measurement of a real arm),
  and the licenses of the tools behind the upstream model files.
- The 16.9 fl oz PET bottle: web figures only, none checked on a real bottle: empty weight about 21 g, or
  18.5 g in the 2024 lightweight redesign (not confirmed for this size), and a generic 203 mm by 63.5 mm.
  Weigh and measure the owner's own bottle. Whether an uncapped bottle spills at 30 mL when tilted is not
  measured (the spill study found about 50 mL at 90 degrees for glass).
- Unknown-container limits: the weight limit per material, and how reliably a camera recognises material,
  are unverified (relevant only once the scope goes beyond cans and capped bottles).
- Whether the cap check should treat a cap that is not confirmed as "set aside": the current rule, to be
  confirmed with the owner when the bottle work starts.
- Stable-Baselines3 default hyperparameters and CPU-vs-GPU guidance (recalled, not re-read).
- Steps per second on the development Mac (measured in 3.5).
- Whether `core/so101_chain.json` ships in a built wheel (hatchling): the editable install finds it, but no
  wheel has been built. Check before the ROS node (7.2, 7.3) installs the package.
- Verified 2026-09-24 against the official Claude Code docs: a private, untracked project context
  file is loaded alongside this one and should not be committed.
