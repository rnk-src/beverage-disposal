# beverage-disposal

A simulated [SO-101](https://github.com/TheRobotStudio/SO-ARM100) robot arm learns, with
reinforcement learning, to pick a soda can off a table, tell whether it is full or empty
by feel, and dispose of it: empty cans go in a bin, full cans go to a set-aside zone.

- **MuJoCo** simulates the physics and is where all training happens.
- **Gazebo** (with ROS 2) validates the trained policy on a different physics engine.
- Built test-first, one small milestone at a time. Every decision and its reasoning is
  written down.

> **Status: work in progress.** This is a ground-up rebuild of an earlier scripted
> version. Nothing is runnable yet beyond the repository checks; see `CLAUDE.md` for the
> current status, the decisions made so far, and the roadmap.

## Running the checks

```bash
uv sync         # reproducible environment from uv.lock (Python 3.12)
uv run pytest   # fast and medium tiers
```

`uv run pre-commit install` (once per clone) makes every commit run ruff and the fast tests.
`CLAUDE.md` lists all commands.

## License

[The Unlicense](LICENSE) (public domain). The vendored SO-101 model files under
`assets/so101/` keep their own license (Apache-2.0); their provenance is recorded there.
