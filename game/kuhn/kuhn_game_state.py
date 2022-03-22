import copy
import random

import numpy as np

from abstracted.game_state import GameState
from game.kuhn.kuhn_player import KuhnPlayer


class KuhnGameState(GameState):
    def __init__(self):
        self.deck = []
        self.history: str = ''
        self.num_players = 2
        self.players = []
        self.players = [KuhnPlayer(0), KuhnPlayer(1)]
        self.current_player_index = 0
        self.actions = ['B', 'C']
        self.cards = ['J', 'Q', 'K']

    def game_start(self):
        self.current_player_index = 0
        self.deck = random.sample(self.cards, self.num_players)
        self.history = ''

    def get_representation(self) -> str:
        player_index = self.current_player_index
        player_card_rank = self.deck[player_index][0]
        actions_as_string = "/".join([str(x) for x in self.history])
        return f'{player_card_rank}-{actions_as_string}'

    def get_actions(self):
        return self.actions

    def handle_action(self, action):
        next_state = copy.deepcopy(self)
        next_state.current_player_index = (next_state.current_player_index + 1) % next_state.num_players
        next_state.history += action
        return next_state

    def get_num_players(self) -> int:
        return self.num_players

    def get_players(self):
        return self.players

    def get_current_player(self):
        return self.players[self.current_player_index]

    def get_current_player_index(self) -> int:
        return self.current_player_index

    def is_terminal(self) -> bool:
        return self.history in ['BC', 'BB', 'CC', 'CBB', 'CBC']

    def get_payoffs(self) -> np.array:
        if self.history == 'BC':
            return np.array([1, -1])
        elif self.history == 'CBC':
            return np.array([-1, 1])
        else:  # CC or BB or CBB
            bet = 2 if 'B' in self.history else 1
            if self.deck[0] == 'K' or self.deck[1] == 'J':
                return np.array([1, -1]) * bet
            else:
                return np.array([-1, 1]) * bet
