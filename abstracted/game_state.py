from abc import ABC, abstractmethod
from typing import List

import numpy as np

from abstracted.player import Player


class GameState(ABC):

    @abstractmethod
    def game_start(self):
        pass

    @abstractmethod
    def get_representation(self) -> str:
        pass

    @abstractmethod
    def get_actions(self):
        pass

    @abstractmethod
    def handle_action(self, action: str):
        pass

    @abstractmethod
    def get_num_players(self):
        pass

    @abstractmethod
    def get_players(self) -> List[Player]:
        pass

    @abstractmethod
    def get_current_player(self) -> Player:
        pass

    @abstractmethod
    def get_current_player_index(self) -> int:
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @abstractmethod
    def get_payoffs(self) -> np.array:
        pass
