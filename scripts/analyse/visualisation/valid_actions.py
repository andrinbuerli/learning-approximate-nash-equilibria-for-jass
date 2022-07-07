import argparse
import logging
import os.path
import sys
from pathlib import Path
from sklearn.metrics import average_precision_score

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

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)

if __name__=="__main__":
    tf.config.experimental_run_functions_eagerly(True)
    parser = argparse.ArgumentParser(prog="Visualize MuZero Model for Jass")
    parser.add_argument(f'--run', default="1655477301")
    parser.add_argument(f'--no_legend', default=True)
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
    predicted_hands = [hand[0].numpy()]
    players = [game.state.player]
    player_kls = []
    hand_aps = []
    is_terminals = []
    while not game.is_done():
        print("Cards played", game.state.nr_played_cards)

        valid_actions_full = rule.get_full_valid_actions_from_state(game.state)
        valid_actions = valid_actions_full / np.maximum(valid_actions_full.sum(), 1)

        policy_kl = (valid_actions * np.log(valid_actions / policy[0].numpy() + 1e-7)).sum()
        policy_kls.append(policy_kl)

        player_kl = np.log(1 / player[0][game.state.player].numpy() + 1e-7)
        player_kls.append(player_kl)

        hand_ap = average_precision_score(hands[-1], hand[0].numpy())
        hand_aps.append(hand_ap)

        support_size = value.shape[-1]
        values.append(support_to_scalar(value[0], min_value=-support_size // 2).numpy())

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

        valid_policy = policy[0].numpy() * valid_actions_full
        valid_policy /= valid_policy.sum()
        action = np.random.choice(range(43), p=valid_policy)
        prev_points = np.array(game.state.points)
        game.perform_action_full(action)
        players.append(game.state.player)
        hands.append(np.copy(game.state.hands[game.state.player]))
        rewards = np.array(game.state.points) - prev_points
        cum_rewards.append(cum_rewards[-1] + rewards)
        all_rewards.append(rewards)

        value, reward, policy, player, hand, is_terminal, encoded_state = network.recurrent_inference(encoded_state, [[action]], all_preds=True)

        predicted_hands.append(hand[0].numpy())
        support_size = reward.shape[-1]
        all_reward_estimates.append(support_to_scalar(reward[0], min_value=-support_size//2).numpy())

    for _ in range(4):
        all_rewards.append([0, 0])
        is_terminals.append(is_terminal[0])
        support_size = value.shape[-1]
        values.append(support_to_scalar(value[0], min_value=-support_size//2).numpy())
        value, reward, policy, player, hand, is_terminal, encoded_state = network.recurrent_inference(encoded_state,
                                                                                                      [[action]],
                                                                                                      all_preds=True)
        support_size = reward.shape[-1]
        all_reward_estimates.append(support_to_scalar(reward[0], min_value=-support_size//2).numpy())

    figsize = (6, 4)
    plt.figure(figsize=figsize)

    plt.plot(policy_kls)
    colors = ["red", "blue", "green", "violet"]
    legend = [True, True, True, True]
    for i, (policy_kl, player) in enumerate(zip(policy_kls, players)):
        plt.scatter(i, policy_kl, c=colors[player], label=f"player {player}" if legend[player] else None)
        legend[player] = False
    if not args.no_legend:
        plt.legend(bbox_to_anchor=(1.1, 1))
    plt.savefig("policy_kls.png", bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=figsize)
    plt.plot(player_kls)
    legend = [True, True, True, True]
    for i, (player_kl, player) in enumerate(zip(player_kls, players)):
        plt.scatter(i, player_kl, c=colors[player], label=f"player {player}" if legend[player] else None)
        legend[player] = False
    if not args.no_legend:
        plt.legend(bbox_to_anchor=(1.1, 1))
    plt.savefig("player_kls.png", bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=figsize)
    plt.plot(hand_aps)
    legend = [True, True, True, True]
    for i, (hand_ap, player) in enumerate(zip(hand_aps, players)):
        plt.scatter(i, hand_ap, c=colors[player], label=f"player {player}" if legend[player] else None)
        legend[player] = False
    if not args.no_legend:
        plt.legend(bbox_to_anchor=(1.1, 1))
    plt.savefig("hand_aps.png", bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=figsize)
    print("hand average precisions per player",
          [
              average_precision_score(np.array(hands)[np.array(np.array(players) == i)].reshape(-1),
                                      np.array(predicted_hands)[np.array(np.array(players) == i)]
                                      .reshape(-1)) for i in range(4)
          ])

    plt.plot(is_terminals)
    plt.axvline(len(is_terminals)-5, c="r")
    plt.savefig("is_terminals.png", bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=figsize)
    plt.plot(game.state.points[0] - np.array(cum_rewards)[:, 0], alpha=0.5, color="green")
    plt.scatter(range(len(cum_rewards)), game.state.points[0] - np.array(cum_rewards)[:, 0], marker="o", label="$z_t$ team 0", alpha=0.5, color="green")
    plt.plot(game.state.points[1] - np.array(cum_rewards)[:, 1], alpha=0.5, color="red")
    plt.scatter(range(len(cum_rewards)), game.state.points[1] - np.array(cum_rewards)[:, 1], marker="o", label="$z_t$ team 1", alpha=0.5, color="red")

    values = np.array(values)
    plt.plot(values[:, 0], alpha=0.5, color="green")
    plt.scatter(range(values.shape[0]), values[:, 0], marker="x", label="$v_t$ team 0", alpha=0.5, color="green")
    plt.plot(values[:, 1], alpha=0.5, color="red")
    plt.scatter(range(values.shape[0]), values[:, 1], marker="x", label="$v_t$ team 1", alpha=0.5, color="red")


    if not args.no_legend:
        plt.legend(loc="upper right")
    plt.xlabel("Moves")
    plt.ylabel("Cumulative Reward")
    plt.savefig("values.png", bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=figsize)
    plt.plot(values[:, 0], marker="x", label="pred 0", alpha=0.5, color="green")
    plt.plot(values[:, 1], marker="x", label="pred 1", alpha=0.5, color="red")

    plt.plot(np.repeat(game.state.points[0], values.shape[0]), marker="o", label="value 0", alpha=0.5, color="green")
    plt.plot(np.repeat(game.state.points[1], values.shape[0]), marker="o", label="value 1", alpha=0.5, color="red")

    if not args.no_legend:
        plt.legend(bbox_to_anchor=(1.1, 1))
    plt.xlabel("Moves")
    plt.ylabel("Outcome")
    plt.savefig("outcome-values.png", bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=figsize)
    legend = True
    for i, (r, r_estimate) in enumerate(zip(all_rewards, all_reward_estimates)):
        plt.scatter(i, r[0], marker="o", color="green", alpha=0.5, label="$u_t$ team 0" if legend else None)
        plt.scatter(i, r[1], marker="o", color="red", alpha=0.5, label="$u_t$ team 1" if legend else None)
        plt.scatter(i, r_estimate[0], marker="x", color="green", label="$r_t$ team 0" if legend else None, s=60)
        plt.scatter(i, r_estimate[1], marker="x", color="red", label="$r_t$ team 1" if legend else None, s=60)
        legend = False

    plt.xlabel("Moves")
    plt.ylabel("Reward")
    if not args.no_legend:
        plt.legend(loc="upper left")
    plt.savefig("rewards.png", bbox_inches='tight')







