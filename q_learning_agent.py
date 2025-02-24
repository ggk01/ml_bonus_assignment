import numpy as np
import random

class QLearningAgent:
    def __init__(self, num_nodes, num_colors, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.01, min_epsilon=0.0):
        self.num_nodes = num_nodes
        self.num_colors = num_colors
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration probability
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}  # State-action table

    def get_q_value(self, state, action):
        """Returns the Q-value for a given state-action pair."""
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        max_next_q = max([self.get_q_value(next_state, a) for a in range(self.num_colors)], default=0.0)
        current_q = self.get_q_value(state, action)
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_next_q)

        # ✅ Prevent overly negative Q-values
        self.q_table[(state, action)] = max(new_q, -5.0)

        # ✅ Print only when a good move is found
        if reward > 0:
            print(f"✅ Good move: Updated Q-value for (State={state}, Action={action}): {self.q_table[(state, action)]}")

    def select_action(self, state):
        """Selects an action using the ε-greedy policy, avoiding highly negative Q-values."""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_colors - 1)  # Explore
        else:
            # ✅ Ensure we do not select a very bad action
            valid_actions = [a for a in range(self.num_colors) if self.get_q_value(state, a) > -5]
            if valid_actions:
                return max(valid_actions, key=lambda a: self.get_q_value(state, a))  # Exploit best valid action
            return random.randint(0, self.num_colors - 1)  # If all bad, still explore

    def decay_epsilon(self):
        """Decays the exploration probability."""
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)
