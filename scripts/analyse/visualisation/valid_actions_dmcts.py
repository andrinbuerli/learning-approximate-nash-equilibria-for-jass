import argparse
import logging
import os.path
import sys

import jasscpp
import jassmlcpp
import numpy as np
import matplotlib.pyplot as plt
from jass.arena.dealing_card_random_strategy import DealingCardRandomStrategy
from jass.game.const import team
from jasscpp import GameSimCpp, RuleSchieberCpp

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="Visualize MuZero Model for Jass")
    parser.add_argument(f'--run', default="1649334860")
    args = parser.parse_args()

    rule = RuleSchieberCpp()
    game = GameSimCpp()
    game.init_from_cards(dealer=0, hands=DealingCardRandomStrategy().deal_cards(
        game_nr=0,
        total_nr_games=1))
    #game.perform_action_trump(0)

    agent = lambda obs: jassmlcpp.mcts.DMCTSFullCpp.run(
        obs,
        jassmlcpp.mcts.RandomHandDistributionPolicyCpp(),
        jassmlcpp.mcts.UCTPolicyFullCpp(np.sqrt(2)),
        jassmlcpp.mcts.RandomRolloutPolicyFullCpp(),
        50,
        1000,
        -1)

    valid_actions = rule.get_full_valid_actions_from_state(game.state)
    valid_actions = valid_actions / valid_actions.sum()

    values = []
    i = 0

    if not os.path.exists("policy_dist"):
        os.mkdir("policy_dist")

    cum_rewards = [np.array([0, 0])]

    all_rewards = [np.array([0, 0])]
    all_reward_estimates = [np.array([0, 0])]

    rewards = [0, 0]
    hands = [game.state.hands[game.state.player]]
    players = [game.state.player]
    while not game.is_done():
        result = agent(jasscpp.observation_from_state(game.state, -1))
        valid_actions = rule.get_full_valid_actions_from_state(game.state)
        valid_actions = valid_actions / np.maximum(valid_actions.sum(), 1)

        value_prediction = (result.probability * result.reward).sum() * 157
        value = np.zeros(2)
        value[team[game.state.player]] = value_prediction
        value[1 - team[game.state.player]] = 157 - value_prediction

        values.append(value)

        plt.bar(range(valid_actions.shape[0]), result.probability, color="blue", label="predicted")
        plt.bar(range(valid_actions.shape[0]), valid_actions, color="red", label="valid", alpha=0.5)
        plt.title(f"player {game.state.player}")
        plt.legend()
        plt.savefig(f"policy_dist/{i}.png")
        plt.clf()

        i += 1

        action = np.flatnonzero(valid_actions)[0]
        prev_points = np.array(game.state.points)
        game.perform_action_full(action)
        rewards = np.array(game.state.points) - prev_points
        cum_rewards.append(cum_rewards[-1] + rewards)
        all_rewards.append(rewards)

    values = np.array(values)
    outcomes = values - np.array([
                np.sum([
                    x for i, x in enumerate(all_rewards[:k])
                ], axis=0) for k in range(1, np.array(all_rewards).shape[0])
            ])

    plt.plot(outcomes[:, 0], marker="x", label="pred 0", alpha=0.5, color="green")
    plt.plot(outcomes[:, 1], marker="x", label="pred 1", alpha=0.5, color="red")

    plt.plot(game.state.points[0] - np.array(cum_rewards)[:, 0], marker="o", label="value 0", alpha=0.5, color="green")
    plt.plot(game.state.points[1] - np.array(cum_rewards)[:, 1], marker="o", label="value 1", alpha=0.5, color="red")

    plt.legend()
    plt.xlabel("Moves")
    plt.ylabel("Cumulative Reward")
    plt.savefig("mdp-values-dmcts.png")
    plt.clf()

    plt.plot(values[:, 0], marker="x", label="pred 0", alpha=0.5, color="green")
    plt.plot(values[:, 1], marker="x", label="pred 1", alpha=0.5, color="red")

    plt.plot(np.repeat(game.state.points[0], values.shape[0]), marker="o", label="value 0", alpha=0.5, color="green")
    plt.plot(np.repeat(game.state.points[1], values.shape[0]), marker="o", label="value 1", alpha=0.5, color="red")

    plt.legend()
    plt.xlabel("Moves")
    plt.ylabel("Outcome")
    plt.savefig("outcome-values-dmcts.png")






