# beverage-disposal

A simulated [SO-101](https://github.com/TheRobotStudio/SO-ARM100) robot arm learns, with
reinforcement learning, to pick a soda can off a table, tell whether it is full or empty
by feel, and dispose of it: empty cans go in a bin, full cans go to a set-aside zone.

- **MuJoCo** simulates the physics and is where all training happens.
- **Gazebo** (with ROS 2) validates the trained policy on a different physics engine.
- Built test-first, one small milestone at a time. Every decision and its reasoning is
  written down.

> **Status: work in progress.** This is a ground-up rebuild of an earlier scripted
> version. Nothing is runnable yet beyond the repository checks; see the repository for the
> current status, the decisions made so far, and the roadmap.

## Running the checks

```bash
python3 -m pytest tests/test_repo_hygiene.py -q
```

A reproducible Python environment (`uv`, Python 3.12) arrives in the next sub-iteration;
this section will then become `uv sync && uv run pytest`.

## License

[The Unlicense](LICENSE) (public domain). Third-party model files, when added, keep their
own licenses and are listed with their provenance.
