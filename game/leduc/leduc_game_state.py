import copy
import random
from enum import Enum

import numpy as np

from abstracted.game_state import GameState
from abstracted.player import Player
from game.leduc.leduc_player import LeducPlayer


class Round(Enum):
    pre_flop = 0
    post_flop = 1


class LeducGameState(GameState):
    def __init__(self):
        self.deck = []
        self.history: str = ''
        self.round = Round.pre_flop
        self.num_players = 2
        self.current_player_index = 0
        self.players = [LeducPlayer(0), LeducPlayer(1)]
        self.actions = [
            'R',  # Raise
            'C',  # Check
            'F',  # Fold
        ]
        self.cards = ['J', 'J', 'Q', 'Q', 'K', 'K']

    def game_start(self):
        self.current_player_index = 0
        self.round = Round.pre_flop
        self.deck = random.sample(self.cards, self.num_players + 1)
        self.history = ''

    def get_representation(self) -> str:
        player_index = self.current_player_index
        player_card_rank = self.deck[player_index]
        actions_as_string = "/".join([str(x) for x in self.history])
        if self.round == Round.pre_flop:
            community_card_rank = ''
        else:
            community_card_rank = self.deck[-1]
        return f'{player_card_rank}/{community_card_rank}-{actions_as_string}'

    def get_actions(self):
        if self.history == '':
            return ['R', 'C']
        elif self.history == 'R':
            return ['R', 'C', 'F']
        elif self.history == 'C':
            return ['R', 'C']
        elif self.history == 'RR':
            return ['C', 'F']
        elif self.history == 'CR':
            return ['R', 'C', 'F']
        elif self.history == 'CRR':
            return ['C', 'F']

    def handle_action(self, action):
        next_state = copy.deepcopy(self)
        next_player = (next_state.current_player_index + 1) % next_state.num_players
        next_state.current_player_index = next_player
        next_state.history += action
        if len(next_state.history) >= next_state.num_players:
            next_state.round = Round.post_flop
        return next_state

    def get_num_players(self):
        return self.num_players

    def get_players(self):
        return self.players

    def get_current_player(self):
        return self.players[self.current_player_index]

    def get_current_player_index(self) -> int:
        return self.current_player_index

    def rank(self, hand: str) -> int:
        ranks = {
            'KK': 1,
            'QQ': 2,
            'JJ': 3,
            'KQ': 4, 'QK': 4,
            'KJ': 5, 'JK': 5,
            'QJ': 6, 'JQ': 6
        }

        cards = hand[0] + hand[1]
        return ranks[cards]

    def is_terminal(self) -> bool:
        return self.history in \
               ['RF', 'RC', 'CC',
                'RRF', 'RRC', 'CRF', 'CRC',
                'CRRF', 'CRRC']

    def get_payoffs(self) -> np.array:
        player_cards = [x[0] + self.deck[-1] for x in self.deck[0:2]]
        player_ranks = [self.rank(x) for x in player_cards]

        if player_ranks[0] < player_ranks[1]:
            utility = np.array([1, -1])
        elif player_ranks[0] > player_ranks[1]:
            utility = np.array([-1, 1])
        else:
            utility = np.array([0, 0])

        if self.history == 'RF':
            return np.array([1, -1])
        elif self.history == 'RC':
            return utility * 3
        elif self.history == 'CC':
            return utility
        elif self.history == 'RRF':
            return np.array([-3, 3])
        elif self.history == 'RRC':
            return utility * 5
        elif self.history == 'CRF':
            return np.array([-1, 1])
        elif self.history == 'CRC':
            return utility * 3
        elif self.history == 'CRRF':
            return np.array([3, -3])
        elif self.history == 'CRRC':
            return utility * 7
