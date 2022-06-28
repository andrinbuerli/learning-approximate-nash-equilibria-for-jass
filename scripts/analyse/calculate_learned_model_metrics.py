import argparse
import logging
import sys
from pathlib import Path

import jasscpp
import numpy as np
from jass.arena.dealing_card_random_strategy import DealingCardRandomStrategy
from jasscpp import GameSimCpp, RuleSchieberCpp
from tqdm import tqdm
from sklearn.metrics import average_precision_score, top_k_accuracy_score

sys.path.append("../../")

from lib.environment.networking.worker_config import WorkerConfig
from lib.factory import get_network, get_features
from lib.jass.features.features_conv_cpp import FeaturesSetCppConv
from lib.mu_zero.network.support_conversion import support_to_scalar

from lib.util import set_allow_gpu_memory_growth

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__=="__main__":
    set_allow_gpu_memory_growth(True)
    parser = argparse.ArgumentParser(prog="Visualize MuZero Model for Jass")
    parser.add_argument(f'--run', default="1655477301")
    parser.add_argument(f'--n_games', default=10, type=int)
    args = parser.parse_args()

    print(f"Run: {args.run}")

    base_path = Path(__file__).resolve().parent.parent.parent / "results" / args.run

    config = WorkerConfig()
    config.load_from_json(base_path / "worker_config.json")

    config.network.feature_extractor = get_features(config.network.feature_extractor)
    feature_extractor = config.network.feature_extractor
    network = get_network(config)

    network.load(base_path / "latest_network.pd", from_graph=True)

    true_hand = []
    predicted_hand = []

    true_player = []
    predicted_player = []

    true_is_terminal = []
    predicted_is_terminal = []

    true_reward = []
    predicted_reward = []

    true_value = []
    predicted_value = []

    for j in tqdm(range(args.n_games)):
        print("-") # force flush
        rule = RuleSchieberCpp()
        game = GameSimCpp()
        game.init_from_cards(dealer=0, hands=DealingCardRandomStrategy().deal_cards(
            game_nr=0,
            total_nr_games=1))

        if isinstance(feature_extractor, FeaturesSetCppConv):
            features = feature_extractor.convert_to_features(jasscpp.observation_from_state(game.state, -1), rule)
        else:
            features = feature_extractor.convert_to_features(game.state, rule)
        value, reward, policy, player, hand, is_terminal, encoded_state = network.initial_inference(features[None],
                                                                                                    all_preds=True)
        true_hand.append(np.copy(game.state.hands[game.state.player]))
        predicted_hand.append(hand[0].numpy())

        true_player.append(np.copy(game.state.player))
        predicted_player.append(player[0].numpy())

        true_is_terminal.append(0)
        predicted_is_terminal.append(is_terminal[0].numpy())

        true_reward.append(np.array([0, 0]))
        predicted_reward.append(np.array([0, 0, 0, 0]))

        support_size = value.shape[-1]
        predicted_value.append(support_to_scalar(value[0], min_value=-support_size//2).numpy())

        current_rewards = [[0, 0]]
        while not game.is_done():
            valid_actions_full = rule.get_full_valid_actions_from_state(game.state)
            valid_actions = valid_actions_full / np.maximum(valid_actions_full.sum(), 1)

            valid_policy = policy[0].numpy() * valid_actions_full
            valid_policy /= valid_policy.sum()
            action = np.random.choice(range(43), p=valid_policy)
            prev_points = np.array(game.state.points)
            game.perform_action_full(action)
            rewards = np.array(game.state.points) - prev_points

            value, reward, policy, player, hand, is_terminal, encoded_state = network.recurrent_inference(encoded_state,
                                                                                                          [[action]],
                                                                                                          all_preds=True)
            true_hand.append(np.copy(game.state.hands[game.state.player]) if game.state.player >= 0 else None)
            predicted_hand.append(hand[0].numpy() if game.state.player >= 0 else None)

            true_player.append(np.copy(game.state.player) if game.state.player >= 0 else None)
            predicted_player.append(player[0].numpy() if game.state.player >= 0 else None)

            true_is_terminal.append(0)
            predicted_is_terminal.append(is_terminal[0].numpy())

            true_reward.append(rewards)
            current_rewards.append(rewards)
            support_size = reward.shape[-1]
            predicted_reward.append(support_to_scalar(reward[0], min_value=-support_size//2).numpy())

            support_size = value.shape[-1]
            predicted_value.append(support_to_scalar(value[0], min_value=-support_size//2).numpy())

        current_rewards = np.array(current_rewards)
        true_value.extend(np.cumsum(current_rewards[::-1], axis=0)[::-1])

        for _ in range(4):
            value, reward, policy, player, hand, is_terminal, encoded_state = network.recurrent_inference(encoded_state,
                                                                                                          [[action]],
                                                                                                          all_preds=True)
            true_hand.append(None)
            predicted_hand.append(None)

            true_player.append(None)
            predicted_player.append(None)

            true_is_terminal.append(1)
            predicted_is_terminal.append(is_terminal[0].numpy())

            true_reward.append(np.array([0, 0]))
            support_size = reward.shape[-1]
            predicted_reward.append(support_to_scalar(reward[0], min_value=-support_size//2).numpy())

            support_size = value.shape[-1]
            predicted_value.append(support_to_scalar(value[0], min_value=-support_size//2).numpy())
            true_value.append(np.array([0, 0]))

    true_hand = np.array(true_hand)
    predicted_hand = np.array(predicted_hand)

    true_player = np.array(true_player)
    predicted_player = np.array(predicted_player)

    true_is_terminal = np.array(true_is_terminal)
    predicted_is_terminal = np.array(predicted_is_terminal)

    true_reward = np.array(true_reward)
    predicted_reward = np.array(predicted_reward)

    true_value = np.array(true_value)
    predicted_value = np.array(predicted_value)

    print(40 * "#")
    print(f"Run: {args.run}")
    print(40*"#")

    player_preds = np.array(predicted_player[true_player != None].tolist())
    player_accuracy = top_k_accuracy_score(np.array(true_player[true_player != None].tolist()).reshape(-1), player_preds, k=1)
    print(f"Player Top-1 Accuracy: {player_accuracy}")

    is_terminal_ap = average_precision_score(np.array(true_is_terminal.tolist()), np.array(predicted_is_terminal.tolist())[:, 0])
    print(f"Is terminal AP: {is_terminal_ap}")

    hand_mAP = average_precision_score(np.array(true_hand[true_player != None].tolist()),
                                       np.array(predicted_hand[true_player != None].tolist()), average="macro")
    print(f"Overall Hands mAP: {hand_mAP}")

    reward_MAE = np.abs(np.array(true_reward.tolist()) - np.array(predicted_reward.tolist())[:, :2])
    print(f"Reward MAE: {reward_MAE.mean()}+-{reward_MAE.std()}")

    value_MAE = np.abs(np.array(true_value.tolist()) - np.array(predicted_value.tolist())[:, :2])
    print(f"Value MAE: {value_MAE.mean()}+-{value_MAE.std()}")

    for p in range(4):
        print(40*"-")
        print(f"Player {p}")
        print(40*"-")

        hand_mAP = average_precision_score(np.array(true_hand[true_player == p].tolist()),
                                        np.array(predicted_hand[true_player == p].tolist()), average="macro")
        print(f"Hand mAP: {hand_mAP}")





