"""A controller under the supervisor's eye: any trip freezes the arm and ends the episode."""

import numpy as np

from beverage_disposal.core.interfaces import CONTROL_PERIOD, Controller, Observation
from beverage_disposal.core.supervisor import Event, Outcome, Supervisor


class SupervisedController:
    """Checks every observation and action; after the first trip it only holds still."""

    def __init__(self, controller: Controller, supervisor: Supervisor):
        self._controller = controller
        self._supervisor = supervisor
        self._steps = 0
        self.stop_event: Event | None = None

    @property
    def done(self) -> bool:
        return self.stop_event is not None or bool(getattr(self._controller, "done", False))

    def reset(self, observation: Observation) -> None:
        self._steps = 0
        self.stop_event = None
        self._controller.reset(observation)

    def act(self, observation: Observation) -> np.ndarray:
        if self.stop_event is not None:
            return self._hold(observation)
        self._steps += 1
        decision = self._supervisor.guard(lambda: self._first_trip(observation))
        if decision.outcome is not Outcome.PROCEED:
            self.stop_event = decision.event
            return self._hold(observation)
        action, decision = self._supervisor.filter_action(self._controller.act(observation))
        if decision.outcome is not Outcome.PROCEED:
            self.stop_event = decision.event
        return action

    def _first_trip(self, observation: Observation):
        decision = self._supervisor.check_observation(observation)
        if decision.outcome is not Outcome.PROCEED:
            return decision
        return self._supervisor.check_time(self._steps * CONTROL_PERIOD)

    @staticmethod
    def _hold(observation: Observation) -> np.ndarray:
        """Stand still with the gripper as last commanded, so a held container is not dropped."""
        gripper = observation.previous_action[3]
        return np.array([0.0, 0.0, 0.0, gripper if np.isfinite(gripper) else 0.0])
