"""Q-learning agent for the inspection route planner."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(
        self,
        num_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.02,
        random_seed: int = 42,
    ):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = np.random.default_rng(random_seed)
        self.q_table: Dict[object, np.ndarray] = defaultdict(lambda: np.zeros(self.num_actions, dtype=float))

    def select_action(self, state, training: bool = True) -> int:
        """Select an action using epsilon-greedy exploration during training."""

        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))

        q_values = self.q_table[state]
        best_actions = np.flatnonzero(np.isclose(q_values, q_values.max()))
        return int(self.rng.choice(best_actions))

    def update(self, state, action: int, reward: float, next_state, done: bool) -> None:
        """Apply the Q-learning update rule."""

        current_q = self.q_table[state][action]
        best_next_q = 0.0 if done else float(np.max(self.q_table[next_state]))
        target_q = reward + self.discount_factor * best_next_q
        self.q_table[state][action] = current_q + self.learning_rate * (target_q - current_q)

    def decay_exploration(self) -> None:
        """Slowly reduce random exploration after each episode."""

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, environment, num_episodes: int, progress_interval: int = 100) -> List[float]:
        """Train the agent in an environment and return episode rewards."""

        rewards = []

        for episode in range(1, num_episodes + 1):
            state = environment.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = self.select_action(state, training=True)
                next_state, reward, done, _ = environment.step(action)
                self.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            self.decay_exploration()
            rewards.append(total_reward)

            if progress_interval and episode % progress_interval == 0:
                print(
                    f"Episode {episode:4d} | "
                    f"reward={total_reward:7.2f} | "
                    f"epsilon={self.epsilon:.3f} | "
                    f"q_states={len(self.q_table)}"
                )

        return rewards

    def generate_route(self, environment) -> Tuple[List[Tuple[int, int]], List[str]]:
        """Run a greedy policy to generate an inspection route."""

        state = environment.reset()
        done = False
        route = [environment.position]
        actions = []

        while not done:
            action = self.select_action(state, training=False)
            next_state, _, done, info = environment.step(action)
            actions.append(info["action_name"])
            route.append(environment.position)
            state = next_state

        return route, actions
