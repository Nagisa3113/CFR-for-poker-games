import numpy as np

from cfr_trainer import CFRTrainer
from game.kuhn.kuhn_game_state import KuhnGameState
from game.kuhn.multi_kuhn_game_state import MultiKuhnGameState
from game.leduc.leduc_game_state import LeducGameState
from game.leduc.multi_leduc_game_state import MultiLeducGameState

if __name__ == "__main__":
    np.set_printoptions(precision=6, floatmode='fixed', suppress=True)
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    np.random.seed(42)

    num_iterations = 100000
    eval_step = 2000

    cfr_trainer = CFRTrainer()

    kuhn_poker = KuhnGameState()  # 12 info_sets
    multi_kuhn_poker = MultiKuhnGameState()  # 48 info_sets
    leduc_poker = LeducGameState()  # 36 info_sets
    multi_leduc_poker = MultiLeducGameState()  # 720 info_sets

    cfr_trainer.train(game_state=kuhn_poker, num_iterations=num_iterations, eval_step=eval_step)
