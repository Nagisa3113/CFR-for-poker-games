from abstracted.player import Player


class KuhnPlayer(Player):
    def __init__(self, index):
        self.utils = 0
        self.index = index

    def get_index(self):
        return self.index
