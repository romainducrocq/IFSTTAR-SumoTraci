from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQUENCY = 1000


class Network(nn.Module):
    def __init__(self, env_):
        super().__init__()

        input_layer_shape = int(np.prod(env_.observation_space.shape))
        output_layer_shape = env_.action_space.n

        self.net = nn.Sequential(
            nn.Linear(input_layer_shape, 64),
            nn.Tanh(),
            nn.Linear(64, output_layer_shape)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs_):
        obs_t = torch.as_tensor(obs_, dtype=torch.float32)
        q_values_ = self(obs_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values_, dim=1)[0]
        action_ = max_q_index.detach().item()
        return action_


env = gym.make("CartPole-v0")
replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0.0], maxlen=100)

episode_reward = 0.0

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

# Initialize Replay Buffer
obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    new_obs, rew, done, _ = env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    if done:
        obs = env.reset()

# Main training loop
obs = env.reset()
for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)

    new_obs, rew, done, _ = env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    episode_reward += rew

    if done:
        obs = env.reset()

        rew_buffer.append(episode_reward)
        episode_reward = 0.0

    # Display solution
    if len(rew_buffer) >= 100:
        if np.mean(rew_buffer) >= 195:
            while True:
                action = online_net.act(obs)
                obs, _, done, _ = env.step(action)
                env.render()
                if done:
                    env.reset()

    # Start Gradient step
    transitions = random.sample(replay_buffer, BATCH_SIZE)

    obses_t = torch.as_tensor(np.asarray([t[0] for t in transitions]), dtype=torch.float32)
    actions_t = torch.as_tensor(np.asarray([t[1] for t in transitions]), dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(np.asarray([t[2] for t in transitions]), dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(np.asarray([t[3] for t in transitions]), dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(np.asarray([t[4] for t in transitions]), dtype=torch.float32)

    # Compute Targets
    target_q_values = target_net(new_obses_t)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    targets = rews_t + (1-dones_t) * GAMMA * max_target_q_values

    # Compute Loss
    q_values = online_net(obses_t)
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    # Gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update Target Network
    if step % TARGET_UPDATE_FREQUENCY == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Logging
    if step % 1000 == 0:
        print()
        print('Step', step)
        print('Avg rew', np.mean(rew_buffer))

