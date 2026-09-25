FROM ros:jazzy-ros-base@sha256:c3706ef0a0aa45413c07803cf433602f543b22e45b4855f6fca955c2d8ecc4e8

# ros-gz-sim pulls in Gazebo Harmonic as ROS vendor packages. The metapackage ros-jazzy-ros-gz is
# avoided because it also installs rviz2 and the demos, which a headless image does not need.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-gz-sim \
        ros-jazzy-ros-gz-bridge \
        ros-jazzy-gz-ros2-control \
    && rm -rf /var/lib/apt/lists/*

COPY worlds /worlds
COPY --chmod=755 step_world.py /usr/local/bin/step_world
