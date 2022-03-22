import copy
import random

import numpy as np

from abstracted.game_state import GameState
from abstracted.player import Player
from game.kuhn.kuhn_player import KuhnPlayer

CARDNAMES = ['9', 'T', 'J', 'Q', 'K']


class MultiKuhnGameState(GameState):
    def __init__(self):
        self.deck = []
        self.history: str = ''
        self.num_players = 3
        self.players = []
        for i in range(self.num_players):
            self.players.append(KuhnPlayer(i))
        self.current_player_index = 0
        self.actions = ['B', 'C']
        self.cards = [x for x in range(self.num_players + 1)]

    def game_start(self):
        self.current_player_index = 0
        self.deck = random.sample(self.cards, self.num_players)
        self.history = ''

    def get_representation(self) -> str:
        player_index = self.current_player_index
        player_card_rank = CARDNAMES[-self.num_players - 1:][self.deck[player_index]]

        actions_as_string = "/".join([str(x) for x in self.history])
        return f'{player_card_rank}-{actions_as_string}'

    def get_actions(self):
        return self.actions

    def handle_action(self, action) -> GameState:
        next_state = copy.deepcopy(self)
        next_state.current_player_index = (next_state.current_player_index + 1) % next_state.num_players
        next_state.history += action
        return next_state

    def get_num_players(self) -> int:
        return len(self.players)

    def get_players(self):
        return self.players

    def get_current_player(self) -> Player:
        return self.players[self.current_player_index]

    def get_current_player_index(self) -> int:
        return self.current_player_index

    def is_terminal(self) -> bool:
        history = self.history
        num_players = self.num_players
        all_acted_after_raise = (history.find('B') > -1) and (len(history) - history.find('B') == num_players)
        all_fold = self.history == 'C' * self.num_players
        return all_acted_after_raise or all_fold

    def get_payoffs(self) -> np.array:
        cards = self.deck
        history = self.history
        num_players = self.num_players
        player = len(history) % num_players
        player_cards = cards[:num_players]
        num_opponents = num_players - 1

        all_fold = self.history == 'C' * self.num_players
        all_but_1_fold = len(self.history) >= self.num_players and \
                         self.history.endswith('C' * (self.num_players - 1))

        if all_fold:
            payouts = [-1] * num_players
            payouts[np.argmax(player_cards)] = num_opponents
            return payouts
        elif all_but_1_fold:
            payouts = [-1] * num_players
            payouts[player] = num_opponents
        else:
            payouts = [-1] * num_players
            active_cards = []
            active_indices = []
            for (ix, x) in enumerate(player_cards):
                if 'B' in history[ix::num_players]:
                    payouts[ix] = -2
                    active_cards.append(x)
                    active_indices.append(ix)
            payouts[active_indices[np.argmax(active_cards)]] = len(active_cards) - 1 + num_opponents
        return payouts
