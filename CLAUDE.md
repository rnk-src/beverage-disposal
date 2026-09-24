# beverage-disposal

Context file for anyone (human or AI assistant) working in this repository. It is committed,
public, and must stay accurate: see "Maintenance rules".

## Purpose

A simulated SO-101 arm learns, with reinforcement learning (RL), to pick a soda can off a
table, tell whether it is full or empty by feel (joint load), and dispose of it: an empty can
goes into a bin, a full can goes to a set-aside region on the table.

- **MuJoCo** simulates the physics and is where all training happens.
- **Gazebo + ROS 2** validates the trained policy on a *different* physics engine (a real
  sim-to-sim transfer test) through a ROS 2 policy node.
- This is a ground-up rebuild of an earlier scripted version, which faked the grasp by
  teleporting the can to the gripper. Here the grasp must be physical.
- The owner drives every decision and must understand every file. Understanding beats speed.

## Current status

- **Iteration 0 (foundations):** 0.1 to 0.5 (repo skeleton, uv environment and pytest tiers,
  ruff and pre-commit, living docs and staleness tests, vendored SO-101 model) are built
  test-first, with red output recorded before each piece of code, and merged on an integration
  branch awaiting owner review. The tests are then to be reorganized into subfolders. 0.6 starts
  after that. 0.7 (Gazebo rendering spike) is deferred until requested.
- Iterations 1 to 8 are planned in detail down to sub-iterations (see Roadmap) and not started.
- Last updated: 2026-09-24. Update this section in the same change as every sub-iteration.

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
- **Platform (validation):** Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic.
- **Tooling:** Python 3.12 (pinned `>=3.12,<3.13`), `uv` (ranges in `pyproject.toml`, exact
  versions in `uv.lock`, hatchling build backend, src layout), pytest with fast/medium/slow
  tiers, hypothesis, ruff (rules E, F, I, UP, B; 100 columns) and pre-commit.
  Versions locked 2026-09-24: mujoco 3.14.0, numpy 2.5.3, gymnasium 1.3.0, pytest 9.1.1,
  hypothesis 6.168.1, ruff 0.16.9, pre-commit 4.6.2.
- **Arm model:** official SO-101 MJCF/URDF (TheRobotStudio/SO-ARM100, "new" calibration),
  vendored untouched under `assets/so101/` with a provenance file (upstream commit
  `5f6d2b876a53a4872e405b991dd925556c9e38a4`, file checksums) and a NOTICE. Those files stay under
  their own Apache-2.0 license, so the repository is not purely public domain. Our own top-level
  MJCF includes them and adds the tip site and actuator settings. MJCF is primary for training; URDF is for Gazebo; a consistency
  test compares them.
- **Observations:** state-based first, designed so a camera can be added later. Measured joints,
  velocities, commanded joint targets, noisy joint load, tip position (our FK), can position
  relative to the tip, vector to the current destination, destination flag, previous action,
  time remaining, and a short frame-stacked history of load/gripper/previous action. Fixed
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
  finger pads on the grasp axis. A closed-form planar IK exists in tests only, as an oracle.
- **Actuators:** start from the servo data in the official files, verify by step-response tests
  (rise time, overshoot, settling, steady-state error), store the reference numbers so Gazebo can
  be matched to them.
- **Grasp:** physical only. Tune pads, friction and solver settings within plausible physics;
  no weld or teleport fallback.
- **Task and fullness:** the destination depends on a hidden variable (can mass). Staged: early
  curriculum stages give a destination flag, later stages hide it. Tests always know the truth.
  Set-aside destination is a marked region on the table. Can is a rigid cylinder (no sloshing);
  masses about 0.014 kg empty and 0.37 kg full.
- **Weight signal:** joint load (actuator force) through a sensor model (noise, quantization,
  per-episode offset), plus commanded targets so position sag is visible. A wrist force-torque
  sensor is a test oracle only. Sensor ranges come from an envelope test.
- **Reward:** sparse success bonus plus potential-based progress shaping, small regularizers,
  failure penalties, early termination on clear failures. Adversarial-policy tests (hover, wave,
  push the can off, drop it, do nothing) must all score below the scripted controller.
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
- **Curriculum:** stages S0 to S5, promoted automatically when the lower confidence bound of
  evaluation success passes a threshold (unit-tested scheduler).
- **Tracking:** dataclass + YAML configs and TensorBoard. Hydra and Weights & Biases are noted
  alternatives to revisit when sweeping.
- **Robustness (iteration 5, aimed at the Gazebo sim-to-sim test):** match what can be measured
  in Gazebo (for example actuator step responses) and randomize the residual uncertainty.
  Randomization ranges start narrow and widen automatically under the same confidence-bound
  promotion rule as the curriculum (stage S5). Mass is randomized as two classes (empty, full),
  each a continuous range, with a guaranteed gap so the correct destination is never ambiguous.
  Physics randomization covers friction, can mass and size, joint damping/friction/armature,
  actuator gains, gripper force limit and contact softness; sensor and timing randomization
  covers load noise/offset/quantization, encoder resolution and observation/action delay. Ranges
  come from a sensitivity study, and a "sim-to-sim gap budget" doc lists each MuJoCo/Gazebo
  difference and whether it is randomized, matched or accepted.
- **License:** The Unlicense. Vendored third-party files keep their own licenses.
- **Repo policy:** the owner makes every commit. Assistants must never run `git add` or
  `git commit`.

## Process rules

1. **Iterations and sub-iterations.** Each iteration delivers a milestone; each has several
   sub-iterations.
2. **Pause after every sub-iteration.** Stop, show the files, and teach what was written. Do not
   start the next sub-iteration until the owner says go.
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
6. **The owner commits.** Assistants summarize changes and may suggest a commit message.
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
- Sim-to-sim test protocol (none published; ours to define): iteration 7.
- Failure search and metamorphic testing: after iteration 5.
- Bridge choice (thin node over standard controllers vs a ros2_control controller plugin;
  `mujoco_ros2_control` only as the 7.4 integration test): decide at 7.3.
- `mink` library instead of our own IK: only if joint-limit or singularity problems appear.
- Hydra / Weights & Biases: when sweeping many runs.
- Scripted-controller demonstrations for imitation-assisted RL: only if the grasp stage stalls.

## Commands

Requires [`uv`](https://docs.astral.sh/uv/) and a Python 3.12 (uv will find or fetch one).

```bash
uv sync                      # create .venv from uv.lock (exact, reproducible versions)
uv run pytest                # default run: fast + medium tiers (slow is excluded)
uv run pytest -m fast        # quick loop, run on every change (about 45 s today)
uv run pytest -m slow        # slow tier only (for example training runs)
uv lock --check              # is uv.lock consistent with pyproject.toml?
uv run ruff check .          # lint (reports only; nothing is auto-fixed)
uv run ruff format .         # format
uv run pre-commit install    # once per clone: run ruff and the fast tier before every commit
```

Dependencies: ranges in `pyproject.toml`, exact versions in `uv.lock`. Runtime: `mujoco`,
`numpy`, `gymnasium`. Dev group: `pytest`, `hypothesis`, `packaging`, `ruff`, `pre-commit`. A test
fails if the installed environment drifts from `uv.lock`, and another fails if the "Versions
locked" sentence above disagrees with it, so update both together when a dependency changes. To add or upgrade a dependency, edit `pyproject.toml`, run
`uv lock` then `uv sync`, and commit both files.

Test tiers: an unmarked test is `fast`; a test may carry at most one of `fast`, `medium`,
`slow`; misspelled markers are errors. The pre-commit hooks run ruff check, ruff format and the
fast tier (an unmarked test counts as fast, so that is most of the suite). The format hook
rewrites files and stops the commit so you re-stage them.

## Layout

```
src/beverage_disposal/   Python package (placeholder name; ROS-free core arrives here)
tests/                   pytest tests (fast/medium/slow tiers from 0.2)
assets/                  vendored third-party files with provenance (the SO-101 model)
configs/                 YAML configs (task, training)
scripts/                 helper scripts (viewer, generators)
pyproject.toml  uv.lock  .python-version   uv project, exact lockfile, pinned Python
.pre-commit-config.yaml                    git hooks (ruff, fast tests)
README.md  CLAUDE.md  LICENSE  .gitignore
```

A test allows only known top-level entries to be published (`PROJECT_TOP_LEVEL` in
`tests/test_publishable_set.py`). When a legitimate new top-level file or folder appears, add it
there in the same change.

Ignored by the published ignore file: `.venv`, `runs/`, caches, editor and OS files. Owner-only
files (study notes, private context file, unpublished tests) are ignored through the clone's
local exclude file and are not listed here.

## Roadmap

Each sub-iteration follows: failing test, minimum code, green, refactor, pause and teach,
update docs.

- **0 Foundations.** 0.1 repo skeleton; 0.2 uv env + pytest tiers; 0.3 ruff + pre-commit;
  0.4 living docs and staleness tests; 0.5 vendored SO-101 model + provenance; 0.6 step and
  render smoke test; 0.7 Gazebo rendering spike (deferred).
- **1 Kinematics and end-effector control.** 1.1 model tests + tip site; 1.2 FK; 1.3 Jacobian;
  1.4 numerical IK; 1.5 EE-delta controller; 1.6 closed loop through PD actuators.
- **2 World and task tests.** 2.1 scene; 2.2 trajectory recorder + pass-fail/indicative tests;
  2.3 grasp physics; 2.4 scripted controller; 2.5 fullness signal; 2.6 acceptance harness.
- **3 Gymnasium environment.** 3.1 observations; 3.2 action layer; 3.3 env core; 3.4 reward;
  3.5 wrappers and vectorization; 3.6 contract and baseline tests.
- **4 First learning.** 4.1 training scaffold + probe environments; 4.2 overfit one config;
  4.3 reach with randomization (SAC seeds and hyperparameter sensitivity); 4.4 grasp and lift; 4.5 full task;
  4.6 hidden destination flag.
- **5 Robustness.** 5.1 randomization framework; 5.2 physics parameter set; 5.3 sensor and timing
  set; 5.4 sensitivity study; 5.5 train under randomization; 5.6 held-out perturbations.
- **6 TdRL comparison.** 6.1 read and choose the integration path; 6.2 map our tests to TdRL's;
  6.3 lexicographic comparator; 6.4 return and reward networks; 6.5 integrate with SAC; 6.6 the
  experiment; 6.7 write-up and decision.
- **7 Gazebo validation.** 7.1 platform; 7.2 policy export; 7.3 ROS policy node; 7.4 node against
  MuJoCo; 7.5 Gazebo world and model; 7.6 actuator matching; 7.7 Gazebo sensors; 7.8 sim-to-sim
  acceptance; 7.9 the gap loop.
- **8 Stretch.** Camera observations.

## Unverified or open

- Jazzy's system Python version (expected 3.12); arm64 apt packages for `ros_gz` and
  `gz_ros2_control` on Jazzy; whether Gazebo cameras and lidar render in a Mac VM.
- STS3215 servo parameters: the vendored MJCF uses kp 998.22, kv 2.731 and a +-3.35 N*m force
  limit (its class default says +-2.94, each actuator overrides it), while the upstream
  `joints_properties.xml` says kp 17.8 and kv 0. Their origin is a third-party README and is
  unverified; MuJoCo uses the MJCF values. Also unverified: real can dimensions (about 66 mm by
  122 mm), the joint ranges (two artifacts from one CAD export, not a measurement of a real arm),
  and the licenses of the tools behind the upstream model files.
- Stable-Baselines3 default hyperparameters and CPU-vs-GPU guidance (recalled, not re-read).
- Steps per second on the development Mac (measured in 3.5).
- Verified 2026-09-24 against the official Claude Code docs: a private, untracked project context
  file is loaded alongside this one and should not be committed.
