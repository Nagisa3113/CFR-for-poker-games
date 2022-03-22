from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt

from abstracted.game_state import GameState
from evaluate.evaluate import compute_exploitability
from information_set import InformationSet


class CFRTrainer():
    def __init__(self):
        self.game_state = None
        self.infoset_map: Dict[str, InformationSet] = {}

    def reset(self):
        for n in self.infoset_map.values():
            n.strategy_sum = np.zeros(n.num_actions)

    def get_information_set(self, history: str, actions: List) -> InformationSet:
        if history not in self.infoset_map:
            self.infoset_map[history] = InformationSet(len(actions))
        return self.infoset_map[history]

    def cfr(self, game_state: GameState, reach_probabilities: np.array) -> np.array:
        if game_state.is_terminal():
            return game_state.get_payoffs()

        player_index = game_state.get_current_player_index()
        num_actions = len(game_state.get_actions())
        num_players = game_state.get_num_players()
        representation = game_state.get_representation()
        actions = game_state.get_actions()
        info_set = self.get_information_set(representation, actions)
        strategy = info_set.get_strategy(reach_probabilities[player_index], training=False)

        counterfactual_values = np.zeros([num_actions, num_players])

        for ix, action in enumerate(actions):
            action_probability = strategy[ix]
            # compute new reach probabilities after this action
            new_reach_probabilities = reach_probabilities.copy()
            new_reach_probabilities[player_index] *= action_probability
            next_state = game_state.handle_action(action)
            # recursively call cfr method
            counterfactual_values[ix] = self.cfr(next_state, new_reach_probabilities)

        # Value of the current game state is just counterfactual values weighted by action probabilities
        # node_values = counterfactual_values.dot(strategy)
        node_values = strategy.dot(counterfactual_values)

        # compute counterfactual reach probability
        cf_reach_prob = np.prod(reach_probabilities[:player_index]) * np.prod(reach_probabilities[player_index + 1:])

        for ix, action in enumerate(actions):
            regrets = counterfactual_values[ix][player_index] - node_values[player_index]
            info_set.cumulative_regrets[ix] += cf_reach_prob * regrets

        return node_values

    def train(self, game_state: GameState, num_iterations: int, eval_step: int):
        self.game_state = game_state
        num_players = len(self.game_state.get_players())
        exploitabilities = []

        utils = np.zeros(game_state.get_num_players())

        print(f"\n-----------------Running CFR for {num_iterations} steps-------------------\n")

        # print(f"Warm start for {num_iterations // 10} iterations")
        # for _ in range(num_iterations // 10):
        #     self.game_state.game_start()
        #     reach_probabilities = np.ones(num_players)
        #     payoffs = self.cfr(self.game_state, reach_probabilities)
        #     utils += payoffs
        # print("Resetting strategy sums")
        # self.reset()
        # utils = np.zeros(game_state.get_num_players())

        for _ in range(num_iterations):
            # use chance sampling at the root node of the game tree
            self.game_state.game_start()
            reach_probabilities = np.ones(num_players)
            payoffs = self.cfr(self.game_state, reach_probabilities)
            utils += payoffs

            if _ > 0 and _ % eval_step == 0:
                exploitabilities.append(compute_exploitability(game_state, self.infoset_map))
                print(f"Exploitability after {_} steps: {compute_exploitability(game_state, self.infoset_map):.6f}")

        print(f"\n-----------------Training end after {num_iterations} steps-------------------\n")

        print(f"Exploitability of final strategy is: {compute_exploitability(game_state, self.infoset_map):.6f}")

        for player in range(game_state.get_num_players()):
            print(f"average utility for player {player + 1}: {(utils[player] / num_iterations):.6f}")

        print(f"information sets:{len(self.infoset_map)}")
        for name, info_set in sorted(self.infoset_map.items(), key=lambda s: len(s[0])):
            print(f"{name:3}:    {info_set.get_average_strategy()}")

        x = [i * eval_step for i in range(len(exploitabilities))]
        plt.plot(x, exploitabilities)
        plt.savefig('figure.png')
        plt.show()
