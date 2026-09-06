# Beverage Disposal Robot

A simulated robotic arm that picks up beverage cans and bottles from a
table, figures out whether they're full or empty, and disposes of them
accordingly: empty containers go in a bin, full ones are set aside. The
whole thing runs in simulation, built with ROS2 and Gazebo, and developed
iteratively with test-driven development — each capability (moving the
arm, gripping, picking something up, sensing fullness, deciding what to
do with it) was built and tested on its own before the next one was
added.

This is a portfolio and learning project, not a production system.

## Architecture

- **ROS2 Humble** — the middleware connecting all the pieces (motion
  control, perception, decision logic) as independent nodes that talk to
  each other over topics and actions.
- **Gazebo Fortress** — the physics simulator, not Gazebo Classic. Classic
  has no prebuilt packages for arm64 (Apple Silicon) and is end-of-life
  regardless of platform, so Fortress is the simulator throughout this
  project.
- **SO-101 arm** — from Hugging Face's LeRobot ecosystem, originally
  designed by TheRobotStudio. It's an affordable, widely-used arm in the
  current robotics learning community. Its ROS2 description is vendored
  into this repository and patched for compatibility with Gazebo Fortress
  (the arm's own upstream Gazebo package targets a newer ROS2/Gazebo
  combination than the one used here).

## Setup

Three setups are documented below. Only the VM-based one has actually been
built and tested end to end for this project — the other two follow
standard, well-documented ROS2 practice and should work, but haven't been
verified here.

All three share the same basic flow once the OS and ROS2 are in place:

```
git clone --recurse-submodules <this-repo-url>
cd beverage-disposal
git submodule update --init --recursive
scripts/apply_vendor_patches.sh
cd ros2_ws
colcon build
colcon test
```

`apply_vendor_patches.sh` applies a small set of local patches to the
vendored arm description so it works with Gazebo Fortress instead of the
newer Gazebo version its upstream package assumes. It only needs to be run
once, right after cloning.

### Native Ubuntu 22.04 (Linux)

The simplest path. Install ROS2 Humble and Gazebo Fortress following the
official ROS2 documentation for your platform's architecture, then follow
the setup flow above.

On x86_64, the Gazebo integration packages this project depends on
(`gz_ros2_control`, parts of `ros_gz`) have prebuilt binaries, so no source
build should be necessary. On arm64, some of these packages have no
prebuilt binaries for ROS2 Humble and need to be built from source inside
the workspace — `colcon build` will build them along with everything else,
it just takes longer the first time.

Native Linux with a real GPU should not need any of the rendering
workarounds mentioned below — those are specific to running inside a
virtual machine with a virtualized graphics adapter.

### Virtual machine (tested setup: Apple Silicon Mac, VMware Fusion, Ubuntu 22.04 arm64 guest)

This is the environment this project was actually developed and tested
on. If you're on a Mac (or otherwise need a VM), a similar setup should
work with other virtualization software too, though only VMware Fusion has
been verified here.

Notes specific to this setup:

- **arm64 packages.** Apple Silicon Macs run arm64 VMs, and as with native
  arm64 Linux above, `gz_ros2_control` and parts of `ros_gz` have no
  prebuilt binaries for Humble on arm64 — they get built from source as
  part of `colcon build`.
- **Rendering.** VMware's virtual GPU (SVGA3D) has a known compatibility
  issue with Gazebo's default Ogre2 rendering engine, causing a blank or
  glitchy viewport. The fix is to force the older Ogre1 engine when
  launching Gazebo with a GUI:
  ```
  ign gazebo --render-engine ogre <world file>
  ```
  (Fortress uses the `ign` command name; the newer `gz` alias isn't
  available on this version.)
  This is a VMware-specific workaround, not something native Linux or
  other virtualization software should need.
- **GUI over SSH.** If you're working over SSH into the VM (rather than at
  its own desktop), remember that GUI applications like Gazebo's viewer
  need an actual display to draw into — either run them from a terminal
  inside the VM's own graphical session, or set up X11/VNC forwarding
  separately.

### WSL2 (Windows) — untested

ROS2 Humble under WSL2 with an Ubuntu 22.04 distribution is a common,
well-documented setup, and should work for this project following the
same setup flow as native Linux above. It has not been tested as part of
this project, so if you hit something specific to WSL2 (GPU passthrough
for the Gazebo GUI is the most likely candidate), you're in genuinely
unverified territory — contributions documenting a working WSL2 setup are
welcome.

## Status

This project is under active, iterative development. Capabilities are
added and tested one at a time; see the commit history for the order they
were built in and what each one covers.

This README is a living document — it's updated whenever the project's
actual structure, setup steps, or component choices (arm, sensing
approach, etc.) change, so it shouldn't go stale relative to the code. If
something here doesn't match what you find in the repository, the code is
the source of truth.
