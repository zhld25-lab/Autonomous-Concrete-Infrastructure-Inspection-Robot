"""Grid-world inspection environment for the RL planner prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ActionSpace:
    """Simple action-space container with readable action names."""

    names: Tuple[str, ...] = ("up", "down", "left", "right", "inspect")

    @property
    def n(self) -> int:
        return len(self.names)


class InspectionEnvironment:
    """Small grid-world environment for concrete surface inspection.

    The robot moves around a simulated concrete surface. The crack risk map is
    fixed during an episode, and the agent receives strong positive reward for
    inspecting high-risk cells.
    """

    def __init__(
        self,
        risk_map,
        high_risk_threshold: float = 0.65,
        start_position: Tuple[int, int] = (0, 0),
        max_steps: int | None = None,
    ):
        self.risk_map = np.asarray(risk_map, dtype=float)
        if self.risk_map.ndim != 2 or self.risk_map.shape[0] != self.risk_map.shape[1]:
            raise ValueError("risk_map must be a square 2D array.")

        self.grid_size = self.risk_map.shape[0]
        self.high_risk_threshold = high_risk_threshold
        self.start_position = tuple(start_position)
        self.max_steps = max_steps or self.grid_size * self.grid_size * 3
        self.action_space = ActionSpace()
        self.high_risk_mask = self.risk_map >= self.high_risk_threshold
        self.reset()

    def reset(self):
        """Reset the environment to the start of an episode."""

        self.position = self.start_position
        self.inspected = np.zeros_like(self.risk_map, dtype=bool)
        self.steps = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        """Return a hashable state with position, local risk, and inspected mask."""

        row, col = self.position
        current_risk_bin = int(self.risk_map[row, col] >= self.high_risk_threshold)
        inspected_mask = tuple(self.inspected.astype(int).flatten().tolist())
        return (row, col, current_risk_bin, inspected_mask)

    def _all_high_risk_cells_inspected(self) -> bool:
        """Check whether every high-risk cell has been inspected."""

        if not self.high_risk_mask.any():
            return True
        return bool(np.all(self.inspected[self.high_risk_mask]))

    def step(self, action: int):
        """Apply one action and return next_state, reward, done, info."""

        if self.done:
            return self._get_state(), 0.0, True, {"message": "Episode already finished."}

        row, col = self.position
        reward = 0.0
        info: Dict[str, object] = {"action_name": self.action_space.names[action]}

        if action == 0:
            next_position = (row - 1, col)
            reward = -1.0
        elif action == 1:
            next_position = (row + 1, col)
            reward = -1.0
        elif action == 2:
            next_position = (row, col - 1)
            reward = -1.0
        elif action == 3:
            next_position = (row, col + 1)
            reward = -1.0
        elif action == 4:
            next_position = (row, col)
            if self.inspected[row, col]:
                reward = -2.0
                info["inspection"] = "repeated"
            else:
                self.inspected[row, col] = True
                risk = self.risk_map[row, col]
                reward = 10.0 if risk >= self.high_risk_threshold else 0.0
                info["inspection"] = "new"
                info["risk"] = float(risk)
        else:
            raise ValueError(f"Unknown action: {action}")

        if action in {0, 1, 2, 3}:
            next_row, next_col = next_position
            if 0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size:
                self.position = next_position
            else:
                reward = -5.0
                info["movement"] = "invalid"

        if self._all_high_risk_cells_inspected():
            reward += 20.0
            self.done = True
            info["completion_bonus"] = True

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True
            info["time_limit"] = True

        return self._get_state(), reward, self.done, info
