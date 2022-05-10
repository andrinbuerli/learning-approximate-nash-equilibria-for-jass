import argparse
import logging
import os.path
import sys
from pathlib import Path

import jasscpp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from jass.arena.dealing_card_random_strategy import DealingCardRandomStrategy
from jasscpp import GameSimCpp, RuleSchieberCpp

from lib.environment.networking.worker_config import WorkerConfig
from lib.factory import get_network, get_features
from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
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
    parser.add_argument(f'--run', default="1649334850")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent.parent / "results" / args.run

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

    if isinstance(feature_extractor, FeaturesSetCppConv):
        features = feature_extractor.convert_to_features(jasscpp.observation_from_state(game.state, -1), rule)
    else:
        features = feature_extractor.convert_to_features(game.state, rule)
    value, reward, policy, player, hand, is_terminal, encoded_state = network.initial_inference(features[None], all_preds=True)

    valid_actions = rule.get_full_valid_actions_from_state(game.state)
    valid_actions = valid_actions / valid_actions.sum()

    policy_kl = (valid_actions * np.log(valid_actions / policy[0].numpy() + 1e-7)).sum()
    policy_kls = []
    values = []
    i = 0

    if not os.path.exists("value_dist"):
        os.mkdir("value_dist")

    if not os.path.exists("reward_dist"):
        os.mkdir("reward_dist")

    if not os.path.exists("policy_dist"):
        os.mkdir("policy_dist")

    if not os.path.exists("player_dist"):
        os.mkdir("player_dist")

    if not os.path.exists("hand_dist"):
        os.mkdir("hand_dist")

    cum_rewards = [np.array([0, 0])]

    all_rewards = [np.array([0, 0])]
    all_reward_estimates = [np.array([0, 0])]

    rewards = [0, 0]
    hands = [game.state.hands[game.state.player]]
    players = [game.state.player]
    player_kls = []
    hand_kls = []
    is_terminals = []
    while not game.is_done():
        valid_actions = rule.get_full_valid_actions_from_state(game.state)
        valid_actions = valid_actions / np.maximum(valid_actions.sum(), 1)

        policy_kl = (valid_actions * np.log(valid_actions / policy[0].numpy() + 1e-7)).sum()
        policy_kls.append(policy_kl)

        player_kl = np.log(1 / player[0][game.state.player].numpy() + 1e-7)
        player_kls.append(player_kl)
        true_hand_dist = hands[-1] / hands[-1].sum()
        pred_hand_dist = hand[0].numpy() / hand[0].numpy().sum()

        hand_kl = (true_hand_dist * np.log(true_hand_dist / pred_hand_dist + 1e-7)).sum()
        hand_kls.append(hand_kl)

        values.append(support_to_scalar(value[0], min_value=0).numpy())

        is_terminals.append(is_terminal[0])

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
        plt.title(f"player {game.state.player}")
        plt.legend()
        plt.savefig(f"policy_dist/{i}.png")
        plt.clf()

        plt.bar(range(player.shape[1]), player[0].numpy(), color="blue", label="predicted")
        plt.title(f"player {game.state.player}")
        plt.legend()
        plt.savefig(f"player_dist/{i}.png")
        plt.clf()

        plt.bar(range(hand.shape[1]), hand[0].numpy(), color="blue", label="predicted")
        plt.bar(range(game.state.hands[game.state.player].shape[0]), game.state.hands[game.state.player], color="red", label="valid", alpha=0.5)
        plt.title(f"player {game.state.player}")
        plt.legend()
        plt.savefig(f"hand_dist/{i}.png")
        plt.clf()
        i += 1

        action = np.flatnonzero(valid_actions)[0]
        prev_points = np.array(game.state.points)
        game.perform_action_full(action)
        players.append(game.state.player)
        hands.append(game.state.hands[game.state.player])
        hands.append(game.state.hands[game.state.player])
        rewards = np.array(game.state.points) - prev_points
        cum_rewards.append(cum_rewards[-1] + rewards)
        all_rewards.append(rewards)

        value, reward, policy, player, hand, is_terminal, encoded_state = network.recurrent_inference(encoded_state, [[action]], all_preds=True)
        all_reward_estimates.append(support_to_scalar(reward[0], min_value=0).numpy())

    for _ in range(5):
        all_rewards.append([0, 0])
        is_terminals.append(is_terminal[0])
        values.append(support_to_scalar(value[0], min_value=0).numpy())
        value, reward, policy, player, hand, is_terminal, encoded_state = network.recurrent_inference(encoded_state,
                                                                                                      [[action]],
                                                                                                      all_preds=True)
        all_reward_estimates.append(support_to_scalar(reward[0], min_value=0).numpy())

    plt.plot(policy_kls)
    colors = ["red", "blue", "green", "violet"]
    legend = [True, True, True, True]
    for i, (policy_kl, player) in enumerate(zip(policy_kls, players)):
        plt.scatter(i, policy_kl, c=colors[player], label=f"player {player}" if legend[player] else None)
        legend[player] = False
    plt.legend()
    plt.savefig("policy_kls.png")
    plt.clf()

    plt.plot(player_kls)
    for i, (player_kl, player) in enumerate(zip(player_kls, players)):
        plt.scatter(i, player_kl, c=colors[player], label=f"player {player}" if legend[player] else None)
        legend[player] = False
    plt.legend()
    plt.savefig("player_kls.png")
    plt.clf()

    plt.plot(hand_kls)
    for i, (hand_kl, player) in enumerate(zip(hand_kls, players)):
        plt.scatter(i, hand_kl, c=colors[player], label=f"player {player}" if legend[player] else None)
        legend[player] = False
    plt.legend()
    plt.savefig("hand_kls.png")
    plt.clf()

    plt.plot(is_terminals)
    plt.axvline(len(is_terminals)-5, c="r")
    plt.savefig("is_terminals.png")
    plt.clf()

    values = np.array(values)
    plt.plot(values[:, 0], marker="x", label="pred 0", alpha=0.5, color="green")
    plt.plot(values[:, 1], marker="x", label="pred 1", alpha=0.5, color="red")

    plt.plot(game.state.points[0] - np.array(cum_rewards)[:, 0], marker="o", label="value 0", alpha=0.5, color="green")
    plt.plot(game.state.points[1] - np.array(cum_rewards)[:, 1], marker="o", label="value 1", alpha=0.5, color="red")

    plt.legend()
    plt.xlabel("Moves")
    plt.ylabel("Cumulative Reward")
    plt.savefig("values.png")
    plt.clf()

    plt.plot(values[:, 0], marker="x", label="pred 0", alpha=0.5, color="green")
    plt.plot(values[:, 1], marker="x", label="pred 1", alpha=0.5, color="red")

    plt.plot(np.repeat(game.state.points[0], values.shape[0]), marker="o", label="value 0", alpha=0.5, color="green")
    plt.plot(np.repeat(game.state.points[1], values.shape[0]), marker="o", label="value 1", alpha=0.5, color="red")

    plt.legend()
    plt.xlabel("Moves")
    plt.ylabel("Outcome")
    plt.savefig("outcome-values.png")
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







