import random

import numpy as np


class InformationSet:
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.cumulative_regrets = np.zeros(shape=self.num_actions)
        self.strategy_sum = np.zeros(shape=self.num_actions)

    def normalize(self, strategy: np.array) -> np.array:
        if sum(strategy) > 0:
            strategy /= sum(strategy)
        # use a uniform random strategy if there is no positive regrets
        else:
            strategy = np.array([1.0 / self.num_actions] * self.num_actions)
        return strategy

    def get_strategy(self, reach_probability: float, training=False) -> np.array:
        # Randomly select action while training
        if training and random.random() < 0.01:
            return np.array([1 / self.num_actions] * self.num_actions)

        # Return regret-matching strategy
        strategy = np.maximum(0, self.cumulative_regrets)
        strategy = self.normalize(strategy)
        self.strategy_sum += reach_probability * strategy
        return strategy

    def get_average_strategy(self) -> np.array:
        return self.normalize(self.strategy_sum.copy())

    def get_average_strategy_with_threshold(self, threshold: float = 0.01) -> np.array:
        avg_strat = self.get_average_strategy()
        avg_strat[avg_strat < threshold] = 0
        return self.normalize(avg_strat)
