#!/usr/bin/env python3
"""Run a Gazebo world headless for an exact number of iterations and print the final state as JSON.

Usage: step_world WORLD.sdf STEPS

Everything goes through Gazebo's request/response services, so there is no topic subscription that
could connect too late and miss a message.
"""

import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.sdf_generator_config_pb2 import SdfGeneratorConfig
from gz.msgs10.serialized_map_pb2 import SerializedStepMap
from gz.msgs10.stringmsg_pb2 import StringMsg
from gz.msgs10.world_control_pb2 import WorldControl
from gz.transport13 import Node

TIMEOUT = 60.0
CALL_MS = 2000


def wait_for(attempt):
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        found = attempt()
        if found:
            return found
        time.sleep(0.1)
    raise TimeoutError("Gazebo did not reach the expected state in time")


def world_state(node, world):
    ok, reply = node.request(f"/world/{world}/state", Empty(), Empty, SerializedStepMap, CALL_MS)
    return reply if ok else None


def step(node, world, steps):
    request = WorldControl()
    # pause: true matters. A request with only multi_step also unpauses the world.
    request.pause = True
    request.multi_step = steps
    ok, reply = node.request(f"/world/{world}/control", request, WorldControl, Boolean, CALL_MS)
    assert ok and reply.data, "Gazebo refused the step request"


def model_poses(node, world):
    """Current position of every top-level model, static ones included."""
    ok, reply = node.request(
        f"/world/{world}/generate_world_sdf",
        SdfGeneratorConfig(),
        SdfGeneratorConfig,
        StringMsg,
        5000,
    )
    assert ok, "Gazebo could not generate the world SDF"
    poses = {}
    for model in ET.fromstring(reply.data).find("world").findall("model"):
        values = [float(v) for v in (model.findtext("pose") or "0 0 0 0 0 0").split()]
        poses[model.get("name")] = values[:3]
    return poses


def main(world_file, steps):
    world = ET.parse(world_file).getroot().find("world").get("name")
    server = subprocess.Popen(
        ["gz", "sim", "-s", world_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        node = Node()
        wait_for(lambda: world_state(node, world))
        if steps:
            step(node, world, steps)
        state = wait_for(
            lambda: (s := world_state(node, world)) and s.stats.iterations >= steps and s
        )
        stamp = state.stats.sim_time
        result = {
            "iterations": state.stats.iterations,
            "sim_time": stamp.sec + stamp.nsec * 1e-9,
            "poses": model_poses(node, world),
        }
    finally:
        server.terminate()
        server.wait()
    print(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
