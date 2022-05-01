from controller import Controller
from collections import deque
from common_utils import noop
from time import time
import numpy as np
import torch
import gc

NUM_EPISODES = 100000
T_MAX = 2000

SCORES_WINDOW_SIZE = 100
TARGET_SCORE = 0.5

def process_states(states):
    ret = states.reshape(2, 3, 8)

    assert all(
        (ret[i, j, 2], ret[i, j, 3]) == (ret[i, j, 6], ret[i, j, 7])
        for i in range(2)
        for j in range(3)
    ), "expected (3rd, 4th) == (7th, 8th)"

    # Get rid of redundant states.
    ret = ret[:, :, :6]
    ret = torch.from_numpy(ret).float()

    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            if ret[i, j].abs().sum() == 0:
                continue

            # Zero-center.
            ret[i, j] += torch.tensor([6, 0, 0, 2, 0, -1.5]).float()

            # Squish to [-1, 1].
            ret[i, j] /= torch.tensor([6, 2, 30, 9, 12, 5]).float()

    return ret

def process_rewards(rewards):
    ret = torch.tensor(rewards).float().unsqueeze(-1)
    ret[ret < 0] *= 10

    return ret

def train(
    env,
    brain_name,
    num_agents,
    num_episodes=NUM_EPISODES,
    debug=noop,
):
    gc.disable()

    scores_window = deque(maxlen=SCORES_WINDOW_SIZE)
    avg_scores_history = []
    scores_history = []

    controller = Controller((3, 6), 2, debug=debug)

    for i in range(1, num_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        controller.reset()

        states = process_states(env_info.vector_observations)
        scores = np.zeros(num_agents)

        start_time = time()
        num_steps = 0

        for j in range(1, T_MAX + 1):
            outputs = controller.act(states)
            actions = outputs["actions"].cpu().numpy()
            env_info = env.step(actions)[brain_name]

            next_states = process_states(env_info.vector_observations)
            rewards = env_info.rewards
            dones = env_info.local_done

            assert np.all(dones) or not np.any(dones), \
                "Expected all or none to be done."

            controller.step({
                "states": states,
                "rewards": process_rewards(rewards),
                "next_states": next_states,
                "dones": torch.tensor(dones).float().unsqueeze(-1),
                **outputs,
            })

            states = next_states

            # NOTE: Adding a list to a 1D NumPy
            # array does element-wise addition.
            scores += rewards
            num_steps = j

            if np.any(dones):
                break

        elapsed_time = time() - start_time

        # We take the max of the 2 agents' scores.
        ep_score = np.max(scores)

        scores_window.append(ep_score)
        scores_history.append(ep_score)

        avg_score = np.mean(scores_window)
        avg_scores_history.append(avg_score)

        buffer_len = controller.replay_buffer.get_len()

        gc.collect()

        print(f"\repisode {i}\tscore {ep_score:.2f}\tavg {avg_score:.2f}\tsteps {num_steps}\tbuffer_len {buffer_len:.2e}\ttime {elapsed_time:.2f}s\t", end="")

        if i % 100 == 0:
            print(f"\repisode {i}\tscore {ep_score:.2f}\tavg {avg_score:.2f}\tsteps {num_steps}\tbuffer_len {buffer_len:.2e}\ttime {elapsed_time:.2f}s\t")

        if i >= SCORES_WINDOW_SIZE and avg_score >= TARGET_SCORE:
            print(f"\nenv solved in {i} episodes! avg score: {avg_score:.2f}")

            controller.save(i)

            break

    gc.enable()

    return scores_history, avg_scores_history
