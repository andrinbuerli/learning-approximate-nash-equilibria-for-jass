import argparse
import logging
import os.path
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from jass.arena.dealing_card_random_strategy import DealingCardRandomStrategy
from jasscpp import GameSimCpp, RuleSchieberCpp

from lib.environment.networking.worker_config import WorkerConfig
from lib.factory import get_network, get_features
from lib.mu_zero.network.support_conversion import support_to_scalar

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__=="__main__":
    tf.config.experimental_run_functions_eagerly(True)
    parser = argparse.ArgumentParser(prog="Visualize MuZero Model for Jass")
    parser.add_argument(f'--run', default="1648651789")
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent.parent / "results" / args.run

    config = WorkerConfig()
    config.load_from_json(base_path / "worker_config.json")

    config.network.feature_extractor = get_features(config.network.feature_extractor)
    feature_extractor = config.network.feature_extractor
    network = get_network(config)

    network.load(base_path / "latest_network.pd", from_graph=True)

    rule = RuleSchieberCpp()
    game = GameSimCpp()
    game.init_from_cards(dealer=0, hands=DealingCardRandomStrategy().deal_cards(
        game_nr=0,
        total_nr_games=1))
    #game.perform_action_trump(0)

    features = feature_extractor.convert_to_features(game.state, rule)
    value, reward, policy, encoded_state = network.initial_inference(features[None])

    valid_actions = rule.get_full_valid_actions_from_state(game.state)
    valid_actions = valid_actions / valid_actions.sum()

    kl = (valid_actions * np.log(valid_actions / policy[0].numpy() + 1e-7)).sum()
    kls = []
    values = []
    i = 0

    if not os.path.exists("value_dist"):
        os.mkdir("value_dist")

    if not os.path.exists("reward_dist"):
        os.mkdir("reward_dist")

    if not os.path.exists("policy_dist"):
        os.mkdir("policy_dist")

    cum_rewards = [np.array([0, 0])]

    all_rewards = [np.array([0, 0])]
    all_reward_estimates = [np.array([0, 0])]

    rewards = [0, 0]
    while valid_actions.sum() > 0:
        valid_actions = rule.get_full_valid_actions_from_state(game.state)
        valid_actions = valid_actions / np.maximum(valid_actions.sum(), 1)

        kl = (valid_actions * np.log(valid_actions / policy[0].numpy() + 1e-7)).sum()
        kls.append(kl)
        values.append(support_to_scalar(value[0], min_value=0).numpy())

        plt.plot(value[0].numpy()[0], c="red")
        plt.plot(value[0].numpy()[1], c="blue")
        plt.savefig(f"value_dist/{i}.png")
        plt.clf()

        plt.plot(reward[0].numpy()[0], c="red")
        plt.plot(reward[0].numpy()[1], c="blue")
        plt.axvline(x=rewards[0], c="red")
        plt.axvline(x=rewards[1], c="blue")
        plt.savefig(f"reward_dist/{i}.png")
        plt.clf()

        plt.bar(range(valid_actions.shape[0]), policy[0].numpy(), color="blue", label="predicted")
        plt.bar(range(valid_actions.shape[0]), valid_actions, color="red", label="valid", alpha=0.5)
        plt.legend()
        plt.savefig(f"policy_dist/{i}.png")
        plt.clf()
        i += 1

        action = policy[0].numpy().argmax()
        prev_points = np.array(game.state.points)
        game.perform_action_full(action)
        rewards = np.array(game.state.points) - prev_points
        cum_rewards.append(cum_rewards[-1] + rewards)
        all_rewards.append(rewards)

        value, reward, policy, encoded_state = network.recurrent_inference(encoded_state, [[action]])
        all_reward_estimates.append(support_to_scalar(reward[0], min_value=0).numpy())

    plt.plot(kls, marker="o")
    plt.savefig("kls.png")
    plt.clf()

    plt.plot(np.array(values)[:, 0], marker="x", label="pred 0", alpha=0.5, color="green")
    plt.plot(np.array(values)[:, 1], marker="x", label="pred 1", alpha=0.5, color="red")

    plt.plot(game.state.points[0] - np.array(cum_rewards)[:, 0], marker="o", label="value 0", alpha=0.5, color="green")
    plt.plot(game.state.points[1] - np.array(cum_rewards)[:, 1], marker="o", label="value 1", alpha=0.5, color="red")

    plt.legend()
    plt.xlabel("Moves")
    plt.ylabel("Cumulative Reward")
    plt.savefig("values.png")
    plt.clf()

    legend = True
    for i, (r, r_estimate) in enumerate(zip(all_rewards, all_reward_estimates)):
        plt.scatter(i, r[0], marker="o", color="green", alpha=0.5, label="reward team 0" if legend else None)
        plt.scatter(i, r[1], marker="o", color="red", alpha=0.5, label="reward team 1" if legend else None)
        plt.scatter(i, r_estimate[0], marker="x", color="green", label="reward estimate team 0" if legend else None)
        plt.scatter(i, r_estimate[1], marker="x", color="red", label="reward estimate team 1" if legend else None)
        legend = False

    plt.xlabel("Moves")
    plt.ylabel("Reward")
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.savefig("rewards.png", bbox_inches='tight')







