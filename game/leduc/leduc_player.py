from abstracted.player import Player


class LeducPlayer(Player):
    def __init__(self, index):
        self.utils = 0
        self.index = index
        self.folded = False

    def get_index(self):
        return self.index
