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


class MultiLeducGameState(GameState):
    def __init__(self):
        self.deck = []
        self.history: str = ''
        self.round = Round.pre_flop
        self.num_players = 3
        self.current_player_index = 0
        self.have_raised = 0
        self.allow_raise_sum = 2
        self.have_acted = 0
        self.players = [LeducPlayer(0), LeducPlayer(1), LeducPlayer(2)]
        self.actions = [
            'R',  # Raise
            'C',  # Check
            'F',  # Fold
        ]
        self.cards = ['T', 'T', 'J', 'J', 'Q', 'Q', 'K', 'K']
        self.chips = [1] * self.num_players

    def game_start(self):
        self.current_player_index = 0
        self.round = Round.pre_flop
        self.have_acted = 0
        self.have_raised = 0
        for p in self.players:
            p.folded = False
        self.chips = [1] * self.num_players
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
        actions = ['R', 'C', 'F']

        # check if it has next players
        has_next_player = False
        next_player_index = self.current_player_index + 1
        while next_player_index < 2 * self.num_players and \
                next_player_index < self.current_player_index + self.num_players:
            if not self.players[next_player_index % self.num_players].folded:
                has_next_player = True
            next_player_index += 1

        # can not raise if other players have raised two times or don't have next players
        if self.have_raised >= self.allow_raise_sum or not has_next_player:
            actions.remove('R')

        # can not Fold if the player don't need to put any chip, instead Check
        if self.chips[self.current_player_index] >= max(self.chips):
            actions.remove('F')

        return actions

    def handle_action(self, action):
        next_state = copy.deepcopy(self)
        next_state.history += action

        # check: add the chips the same as the max of other players
        if action == 'C':
            next_state.chips[next_state.current_player_index] = max(next_state.chips)
            next_state.have_acted += 1
        # raise: add 2 more chips in pre_flop or 4 more chips in post_flop
        elif action == 'R':
            next_state.have_acted = 1
            next_state.have_raised += 1
            next_state.chips[next_state.current_player_index] = max(
                next_state.chips) + (2 if next_state.round == Round.pre_flop else 4)
        elif action == 'F':
            if not next_state.players[next_state.current_player_index].folded:
                next_state.players[next_state.current_player_index].folded = True
            next_state.have_acted += 1

        next_state.current_player_index = (next_state.current_player_index + 1) % next_state.num_players

        # skip if the next player has held
        while next_state.players[next_state.current_player_index].folded and len(
                next_state.history) < 2 * next_state.num_players:
            next_state.history += 'F'
            next_state.have_acted += 1
            next_state.current_player_index = (next_state.current_player_index + 1) % next_state.num_players

        if len(next_state.history) == next_state.num_players:
            next_state.round = Round.post_flop
            next_state.have_raised = 0

        return next_state

    def get_num_players(self):
        return self.num_players

    def get_players(self):
        return self.players

    def get_current_player(self) -> Player:
        return self.players[self.current_player_index]

    def get_current_player_index(self) -> int:
        return self.current_player_index

    def rank(self, hand) -> int:
        ranks = {
            'KK': 1,
            'QQ': 2,
            'JJ': 3,
            'TT': 4,
            'KQ': 5, 'QK': 5,
            'KJ': 6, 'JK': 6,
            'KT': 7, 'TK': 7,
            'QJ': 8, 'JQ': 8,
            'QT': 9, 'TQ': 9,
            'JT': 10, 'TJ': 10,
        }

        cards = hand[0][0] + hand[1][0]
        return ranks[cards]

    def is_terminal(self) -> bool:
        # game end if all other players act after the last raise or reach the max round
        reach_max_round = len(self.history) >= 2 * self.num_players
        all_players_acted = self.have_acted >= self.num_players

        # if all the players fold except one
        active_num = 0
        for player in self.players:
            if not player.folded:
                active_num += 1
        all_but_1_player_fold = active_num == 1

        return reach_max_round or all_players_acted or all_but_1_player_fold

    def get_payoffs(self) -> np.array:
        # list all active players
        active_players = []
        for player in self.players:
            if not player.folded:
                active_players.append(player)

        player_cards = [self.deck[player.index] + self.deck[-1] for player in active_players]
        player_ranks = [self.rank(x) for x in player_cards]
        highest_rank = min(player_ranks)
        winners = [0] * len(self.players)
        for index in range(len(winners)):
            if (self.rank(self.deck[index] + self.deck[-1])) == highest_rank:
                winners[index] = 1

        chips_sum = 0
        winners_sum = 0
        utilities = [0] * len(self.players)
        for index in range(len(winners)):
            if winners[index] == 0:
                chips_sum += self.chips[index]
                utilities[index] = -self.chips[index]
            if winners[index] == 1:
                winners_sum += 1
        for index in range(len(winners)):
            if winners[index] == 1:
                utilities[index] = chips_sum / winners_sum

        return utilities
