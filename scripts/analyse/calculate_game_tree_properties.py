import argparse
import json
import logging
import math
import sys

import jasscpp
import numpy as np
from jasscpp import GameSimCpp
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def get_number_of_information_sets(state):
    current_player = state.player
    current_player_hand = state.hands[current_player]
    num_unknown_cards = 36 - state.nr_played_cards - current_player_hand.sum()
    combinations = 1
    for other_player in range(4):
        if current_player == other_player:
            continue
        num_cards = state.hands[other_player].sum()
        combinations *= math.comb(num_unknown_cards, num_cards)
        num_unknown_cards -= num_cards

    return combinations


def get_neighbourhood(h):
    game_finished = h.nr_played_cards >= 36
    if game_finished:
        return [h.points]

    game = GameSimCpp()
    game.state = h
    valid_actions = np.flatnonzero(game.get_valid_cards())
    assert len(valid_actions) > 0, f"invalid nr of valid actions {valid_actions}, {game.state}"

    outcomes = []
    for a in valid_actions:
        game.perform_action_full(a)
        outcomes.extend(get_neighbourhood(game.state))

    return outcomes



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Calculate Game Tree Properties')
    parser.add_argument(f'--n_games', default=10_000, type=int)
    parser.add_argument(f'--game_type', default="schieber-jass", type=str)
    args = parser.parse_args()

    dfs = []
    lcs = []
    bs = []

    if args.game_type == "schieber-jass":
        utils = jasscpp.GameUtilsCpp()
        for _ in tqdm(range(args.n_games)):
            print('-')
            sim = GameSimCpp()
            dealer = np.random.choice(range(4))
            sim.init_from_cards(utils.deal_random_hand(), dealer=dealer)

            is_sizes = {p: [] for p in range(4)}

            leaf_neighbour_utility = []
            while not sim.is_done():
                player = sim.state.player
                is_sizes[player].append(get_number_of_information_sets(sim.state))

                if 27 < sim.state.nr_played_cards < 32:  # start of third last trick
                    current_leaf_neighbour_utility = get_neighbourhood(sim.state)

                    if len(current_leaf_neighbour_utility) > 1: # ensure only post terminal nodes are selected
                        leaf_neighbour_utility = current_leaf_neighbour_utility

                valid_actions = sim.get_valid_actions()
                action = np.random.choice(np.flatnonzero(valid_actions))
                sim.perform_action_full(action)

            is_sizes = [np.array(x) for x in is_sizes.values()]
            df = np.mean([1 - np.mean(x[1:] / x[:-1]) for x in is_sizes])
            dfs.append(df)

            leaf_neighbour_utility = np.array(leaf_neighbour_utility)
            lc = (1 - np.abs(leaf_neighbour_utility - sim.state.points) / 157).mean()
            lcs.append(lc)

            trump_team_won_most_points_neighbourhood = (
                        leaf_neighbour_utility[:, sim.state.declared_trump_player % 2] > 157 / 2)
            trump_team_won_most_points_local = (sim.state.points[sim.state.declared_trump_player % 2] > 157 / 2)
            b = (trump_team_won_most_points_neighbourhood == trump_team_won_most_points_local).mean()
            bs.append(b)
    else:  # kuhn-poker
        leaf_node_outcomes = np.array([
            [-1, -1, -2, 1, -2],
            [-1, -1, -2, 1, -2],
            [ 1, -1,  2, 1,  2],
            [-1, -1, -2, 1, -2],
            [ 1, -1,  2, 1,  2],
            [ 1, -1,  2, 1,  2],
        ])

        df_per_leaf_node = np.array([
            [0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000],
        ]) # always zero, cards are (if at all) only revealed at end of game

        leaf_node_neighbours = [[1, 2], [2], [1], [4], 3]

        for _ in tqdm(range(args.n_games)):
            information_set_idx = np.random.choice(range(6))
            leaf_idx = np.random.choice(range(5))

            dfs.append(df_per_leaf_node[information_set_idx, leaf_idx])

            neighbours = leaf_node_outcomes[information_set_idx][leaf_node_neighbours[leaf_idx]]
            sample_outcome = leaf_node_outcomes[information_set_idx, leaf_idx]
            lc = (1 - np.abs(neighbours - sample_outcome) / 4).mean()
            lcs.append(lc)

            p1_won_most_points_neighbourhood = (neighbours > 0)
            p1_won_most_points_local = (sample_outcome > 0)
            b = (p1_won_most_points_neighbourhood == p1_won_most_points_local).mean()
            bs.append(b)

    with open(f"{args.game_type}-game-tree.json", "w") as f:
        json.dump({
            "dfs": dfs,
            "lcs": lcs,
            "bs": bs
        }, f)