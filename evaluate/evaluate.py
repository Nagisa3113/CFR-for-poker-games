import itertools
import numpy as np
from abstracted.game_state import GameState


def calc_best_response(game_state: GameState, node_map, br_strat_map, br_player, prob):
    if game_state.is_terminal():
        return game_state.get_payoffs()

    key = game_state.get_representation()
    index = game_state.get_current_player_index()
    actions = game_state.get_actions()

    if index == br_player:
        vals = []
        for action in actions:
            next_state = game_state.handle_action(action)
            vals.append(calc_best_response(next_state, node_map, br_strat_map, br_player, prob))

        best_response_value = max(vals, key=lambda s: s[index])

        if key not in br_strat_map:
            br_strat_map[key] = np.zeros(len(actions))
        br_strat_map[key] = br_strat_map[key] + prob * np.array([x[index] for x in vals], dtype=np.float64)

        return best_response_value
    else:
        strategy = node_map[key].get_average_strategy()
        action_values = []
        for ix in range(len(actions)):
            next_state = game_state.handle_action(actions[ix])
            action_values.append(calc_best_response(next_state, node_map, br_strat_map, br_player, prob * strategy[ix]))
        return np.dot(strategy, action_values)


def calc_ev(game_state: GameState, index, strat_1, strat_2):
    if game_state.is_terminal():
        return game_state.get_payoffs()

    current_player = game_state.get_current_player_index()
    strat = strat_1 if current_player == index else strat_2
    strategy = strat[game_state.get_representation()]

    ev = []
    for action in game_state.get_actions():
        next_state = game_state.handle_action(action)
        ev.append(calc_ev(next_state, index, strat_1, strat_2))

    return np.dot(strategy, ev)


def compute_exploitability(game_state: GameState, infoset_map):
    exploitability = []
    br_strat_map = {}
    for cards in itertools.permutations(game_state.cards, game_state.get_num_players() + 1):
        game_state.deck = cards
        for player_index in range(game_state.get_num_players()):
            calc_best_response(game_state, infoset_map, br_strat_map, player_index, 1.0)

    for k, v in br_strat_map.items():
        v[:] = np.where(v == np.max(v), 1, 0)

    # get average strategy with threshold
    cfr_strategy = {k: v.get_average_strategy_with_threshold() for k, v in infoset_map.items()}

    for cards in itertools.permutations(game_state.cards, game_state.get_num_players() + 1):
        game_state.deck = cards
        utils = []
        for index in range(game_state.get_num_players()):
            extra_profit = calc_ev(game_state, index, br_strat_map, cfr_strategy)[index] - \
                    calc_ev(game_state, index, cfr_strategy, cfr_strategy)[index]
            utils.append(extra_profit)

        exploitability.append(sum(utils))

    return np.mean(exploitability)
