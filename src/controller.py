from policy import ActorNet, CriticNet
from collections import deque
from common_utils import noop
import torch.optim as optim
import random
import torch

###########################
# Begin hyper-parameters. #
###########################

ACTOR_LR = 0.001
CRITIC_LR = 0.001

REPLAY_BUFFER_SIZE = 100000
ROLLOUT_LEN = 1
BATCH_SIZE = 64

EPSILON = 0.1
GAMMA = 0.99
TAU = 0.001

GRADIENT_CLIP = 1
LEARN_EVERY = 1

#########################
# End hyper-parameters. #
#########################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer():
    def __init__(self, size, batch_size, debug=noop):
        self.replay_buffer = deque(maxlen=size)
        self.batch_size = batch_size

        self.debug = debug

    def get_len(self):
        return len(self.replay_buffer)

    def can_sample(self):
        return self.get_len() >= self.batch_size

    def add(self, trajectories):
        self.replay_buffer += trajectories

    def sample(self):
        return random.sample(self.replay_buffer, self.batch_size)

class Controller():
    def __init__(self, state_size, action_size, debug=noop, seed=1337):
        random.seed(seed)

        self.n_atoms = 51
        self.v_min = -0.1
        self.v_max = 0.1
        self.delta = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.v_lin = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(device)

        self.actor = ActorNet(state_size, action_size, seed).to(device)
        self.critic = CriticNet(state_size, action_size, self.n_atoms, seed).to(device)

        self.target_actor = ActorNet(state_size, action_size, seed).to(device)
        self.target_critic = CriticNet(state_size, action_size, self.n_atoms, seed).to(device)

        # Initialize target networks.
        self.soft_update_target_nets(1)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, BATCH_SIZE, debug=debug)

        self.trajectories = deque(maxlen=ROLLOUT_LEN)
        self.rollout_len = ROLLOUT_LEN
        self.epsilon = EPSILON

        self.action_size = action_size

        self.debug = debug
        self.num_steps = 0

    def reset(self):
        self.trajectories = deque(maxlen=self.rollout_len)

    def act(self, states, test=False):
        states = states.to(device)

        if self.contains_zero_state(states):
            return {
                "actions": torch.zeros(states.shape[0], self.action_size)
            }

        with self.actor.eval_no_grad():
            actions = self.actor(states)

        if not test:
            noise = torch.distributions.Normal(torch.zeros(actions.size()), 1)
            actions += self.epsilon * noise.sample().to(device)
            actions = actions.clamp(-1, 1)

        return {
            "actions": actions,
        }

    def step(self, transitions):
        if self.num_steps % LEARN_EVERY == 0 and self.replay_buffer.can_sample():
            self.learn()

        self.num_steps += 1

        if self.contains_zero_state(transitions["states"]):
            return

        exp_tuples = self.convert_to_exp_tuples(transitions)
        self.trajectories.append(exp_tuples)

        if len(self.trajectories) < self.rollout_len:
            return

        self.replay_buffer.add(list(zip(*self.trajectories)))

    def contains_zero_state(self, states):
        return (states.abs().sum(-1) == 0).sum().item() > 0

    def learn(self):
        samples = self.replay_buffer.sample()
        trajectories = [self.convert_to_transitions(exp_tuples) for exp_tuples in zip(*samples)]

        first_transitions = trajectories[0]
        first_states = first_transitions["states"]
        first_actions = first_transitions["actions"]

        local_q_dists = self.critic(first_states, first_actions)
        projected_target_q_dists = self.get_projected_target_q_dists(trajectories)
        total_critic_loss = -(torch.log(local_q_dists + 1e-10) * projected_target_q_dists).sum(dim=-1).mean()

        self.critic_opt.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), GRADIENT_CLIP)
        self.critic_opt.step()

        actions = self.actor(first_states)
        q_dists = self.critic(first_states, actions)
        total_actor_loss = -q_dists.matmul(self.v_lin).mean()

        self.actor_opt.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRADIENT_CLIP)
        self.actor_opt.step()

        self.soft_update_target_nets(TAU)

    def soft_update_target_nets(self, tau):
        self.soft_update_target_net(self.actor, self.target_actor, tau)
        self.soft_update_target_net(self.critic, self.target_critic, tau)

    def soft_update_target_net(self, local_net, target_net, tau):
        for target_params, local_params in zip(
            target_net.parameters(),
            local_net.parameters(),
        ):
            target_params.data.copy_(
                tau * local_params.data + \
                (1 - tau) * target_params.data
            )

    def convert_to_exp_tuples(self, transitions):
        return list(zip(
            transitions["states"].cpu(),
            transitions["actions"].cpu(),
            transitions["rewards"].cpu(),
            transitions["next_states"].cpu(),
            transitions["dones"].cpu(),
        ))

    def convert_to_transitions(self, exp_tuples):
        states, actions, rewards, next_states, dones = zip(*exp_tuples)

        return {
            "states": torch.stack(states).to(device),
            "actions": torch.stack(actions).to(device),
            "rewards": torch.stack(rewards).to(device),
            "next_states": torch.stack(next_states).to(device),
            "dones": torch.stack(dones).to(device),
        }

    def get_projected_target_q_dists(self, trajectories):
        N = len(trajectories)

        last_transitions = trajectories[-1]
        last_next_states = last_transitions["next_states"]

        discounted_rewards = torch.zeros(last_transitions["rewards"].size()).to(device)

        for transitions in reversed(trajectories):
            discounted_rewards = transitions["rewards"] + GAMMA * discounted_rewards

        discounted_rewards = discounted_rewards.squeeze()

        target_actions = self.target_actor(last_next_states)
        target_q_dists = self.target_critic(last_next_states, target_actions)

        projected_target_q_dists = torch.zeros(target_q_dists.size()).to(device)

        for j in range(self.n_atoms):
            Tz_j = torch.clamp(
                discounted_rewards + (GAMMA ** N) * (self.v_min + j * self.delta),
                min=self.v_min,
                max=self.v_max,
            )

            b_j = (Tz_j - self.v_min) / self.delta
            l = b_j.floor().long()
            u = b_j.ceil().long()

            eq_mask = l == u
            ne_mask = l != u

            projected_target_q_dists[eq_mask, l[eq_mask]] += target_q_dists[eq_mask, j]
            projected_target_q_dists[ne_mask, l[ne_mask]] += target_q_dists[ne_mask, j] * (u.float() - b_j)[ne_mask]
            projected_target_q_dists[ne_mask, u[ne_mask]] += target_q_dists[ne_mask, j] * (b_j - l.float())[ne_mask]

        return projected_target_q_dists.detach()

    def save(self, i):
        torch.save(self.actor.cpu().state_dict(), f"actor_{i}.pth")
        torch.save(self.critic.cpu().state_dict(), f"critic_{i}.pth")
