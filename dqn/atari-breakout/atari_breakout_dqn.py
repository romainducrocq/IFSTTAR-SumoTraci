import os
from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random
import time

from torch.utils.tensorboard import SummaryWriter

from baselines_wrappers import DummyVecEnv, SubprocVecEnv, Monitor
from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames

import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()

DOUBLE = True
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = int(4e5)  # int(1e6) ## Hardware limitations
MIN_REPLAY_SIZE = 50000
EPSILON_START = 1.0
EPSILON_END = (0.01 if DOUBLE else 0.1)
EPSILON_DECAY = int(1e6)
NUM_ENVS = 4
TARGET_UPDATE_FREQUENCY = (30000 if DOUBLE else 10000) // NUM_ENVS
LR = 5e-5
ALGORITHM = ('double_' if DOUBLE else '') + 'dqn'
SAVE_PATH = './save/' + str(ALGORITHM) + '/lr' + str(LR) + '/model.pack'
SAVE_INTERVAL = 10000
LOG_DIR = './logs/' + str(ALGORITHM) + '/lr' + str(LR) + '/'
LOG_INTERVAL = 1000


def nature_cnn(observation_space, depths=(32, 64, 64), final_layer=512):
    n_input_channels = observation_space.shape[0]

    cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten()
    )

    with torch.no_grad():
        n_flatten = cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

    out = nn.Sequential(
        cnn,
        nn.Linear(n_flatten, final_layer),
        nn.ReLU()
    )

    return out


class Network(nn.Module):
    def __init__(self, _env, _device, double=True):
        super().__init__()

        self.num_actions = _env.action_space.n
        self.device = _device
        self.double = double

        conv_net = nature_cnn(_env.observation_space)
        self.net = nn.Sequential(
            conv_net,
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, _obses, _epsilon):
        obses_t = torch.as_tensor(_obses, dtype=torch.float32, device=self.device)
        q_values = self(obses_t)

        max_q_indices = torch.argmax(q_values, dim=1)
        _actions = max_q_indices.detach().tolist()

        for i in range(len(_actions)):
            if random.random() <= _epsilon:
                _actions[i] = random.randint(0, self.num_actions - 1)

        return _actions

    def compute_loss(self, _transitions, _target_net):
        obses_t = torch.as_tensor(np.stack([o.get_frames() for o in [t[0] for t in _transitions]]), dtype=torch.float32, device=self.device) \
            if isinstance(_transitions[0][0], PytorchLazyFrames) \
            else torch.as_tensor(np.asarray([t[0] for t in _transitions]), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(np.asarray([t[1] for t in _transitions]), dtype=torch.int64, device=self.device).unsqueeze(-1)
        rews_t = torch.as_tensor(np.asarray([t[2] for t in _transitions]), dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(np.asarray([t[3] for t in _transitions]), dtype=torch.float32, device=self.device).unsqueeze(-1)
        new_obses_t = torch.as_tensor(np.stack([o.get_frames() for o in [t[4] for t in _transitions]]), dtype=torch.float32, device=self.device) \
            if isinstance(_transitions[4][0], PytorchLazyFrames) \
            else torch.as_tensor(np.asarray([t[4] for t in _transitions]), dtype=torch.float32, device=self.device)

        # Compute Targets
        with torch.no_grad():
            if self.double:
                targets_online_q_values = self(new_obses_t)
                targets_online_best_q_indices = targets_online_q_values.argmax(dim=1, keepdim=True)

                targets_target_q_values = target_net(new_obses_t)
                targets_selected_q_values = torch.gather(input=targets_target_q_values, dim=1, index=targets_online_best_q_indices)

                targets = rews_t + (1-dones_t) * GAMMA * targets_selected_q_values
            else:
                target_q_values = _target_net(new_obses_t)
                max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

                targets = rews_t + (1-dones_t) * GAMMA * max_target_q_values

        # Compute Loss
        q_values = self(obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        _loss = nn.functional.smooth_l1_loss(action_q_values, targets)
        return _loss

    def save(self, save_path, _step, _episode_count, _rew_mean, _len_mean):
        params = {
            "model": {k: v.detach().cpu().numpy() for k, v, in self.state_dict().items()},
            "step": _step, "episode_count": _episode_count, "rew_mean": _rew_mean, "len_mean": _len_mean
        }
        params_data = msgpack.dumps(params)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(params_data)

    def load(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)

        with open(load_path, 'rb') as f:
            params_dict = msgpack.loads(f.read())

        params = {k: torch.as_tensor(v, device=self.device) for k, v in params_dict["model"].items()}
        self.load_state_dict(params)

        return params_dict["step"], params_dict["episode_count"], params_dict["rew_mean"], params_dict["len_mean"]


if __name__ == "__main__":
    start_time = time.time()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    make_env = lambda: Monitor(make_atari_deepmind('BreakoutNoFrameskip-v4', scale_values=True), allow_early_resets=True)
    # vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
    vec_env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])

    env = BatchedPytorchFrameStack(vec_env, k=NUM_ENVS)

    replay_buffer = deque(maxlen=BUFFER_SIZE)
    epinfos_buffer = deque([], maxlen=100)

    resume_step, episode_count, rew_mean, len_mean = 0, 0, 0, 0

    summary_writer = SummaryWriter(LOG_DIR)

    online_net = Network(env, device, double=DOUBLE)
    target_net = Network(env, device, double=DOUBLE)

    online_net = online_net.to(device)
    target_net = target_net.to(device)

    # Resume training from checkpoint
    if os.path.exists(SAVE_PATH):
        print()
        print("Resume training from " + SAVE_PATH + "...")
        resume_step, episode_count, rew_mean, len_mean = online_net.load(SAVE_PATH)
        [epinfos_buffer.append({'r': rew_mean, 'l': len_mean}) for _ in range(np.min([episode_count, epinfos_buffer.maxlen]))]
        print("Step: ", resume_step, ", Episodes: ", episode_count, ", Avg Rew: ", rew_mean, ", Avg Ep Len: ", len_mean)

    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

    print()
    print("Initialize Replay Buffer...")
    # Initialize Replay Buffer
    obses = env.reset()
    beginning_episodes = [True for _ in range(NUM_ENVS)]
    for t in range(MIN_REPLAY_SIZE):
        if t >= MIN_REPLAY_SIZE - resume_step:
            if isinstance(obses[0], PytorchLazyFrames):
                act_obses = np.stack([o.get_frames() for o in obses])
                actions = online_net.act(act_obses, 0.0)
            else:
                actions = online_net.act(obses, 0.0)
        else:
            actions = [env.action_space.sample() for _ in range(NUM_ENVS)]

        for e, beginning_episode in enumerate(beginning_episodes):
            if beginning_episode:
                beginning_episodes[e] = False
                if t >= MIN_REPLAY_SIZE - resume_step:
                    actions[e] = 1

        new_obses, rews, dones, _ = env.step(actions)
        for e, (obs, action, rew, done, new_obs) in enumerate(zip(obses, actions, rews, dones, new_obses)):
            transition = (obs, action, rew, done, new_obs)
            replay_buffer.append(transition)

            if done:
                beginning_episodes[e] = True

        obses = new_obses

        if (t+1) % 10000 == 0:
            print(str(t+1) + ' / ' + str(MIN_REPLAY_SIZE))
            print("--- %s seconds ---" % round((time.time() - start_time), 2))

    print()
    print("Start Training...")
    # Main training loop
    obses = env.reset()
    for step in itertools.count(start=resume_step):
        epsilon = np.interp(step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

        if isinstance(obses[0], PytorchLazyFrames):
            act_obses = np.stack([o.get_frames() for o in obses])
            actions = online_net.act(act_obses, epsilon)
        else:
            actions = online_net.act(obses, epsilon)

        new_obses, rews, dones, infos = env.step(actions)

        for obs, action, rew, done, new_obs, info in zip(obses, actions, rews, dones, new_obses, infos):
            transition = (obs, action, rew, done, new_obs)
            replay_buffer.append(transition)

            if done:
                epinfos_buffer.append({'r': info['episode']['r'], 'l': info['episode']['l']})
                episode_count += 1

        obses = new_obses

        # Start Gradient step
        transitions = random.sample(replay_buffer, BATCH_SIZE)
        loss = online_net.compute_loss(transitions, target_net)

        # Gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Target Network
        if step % TARGET_UPDATE_FREQUENCY == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Logging
        if step % LOG_INTERVAL == 0 and step > resume_step:
            rew_mean = np.mean([e['r'] for e in epinfos_buffer]) or 0
            len_mean = np.mean([e['l'] for e in epinfos_buffer]) or 0

            print()
            print('Step:', step)
            print('Avg Rew:', rew_mean)
            print('Avg Ep Len:', len_mean)
            print('Episodes:', episode_count)
            print("--- %s seconds ---" % round((time.time() - start_time), 2))

            summary_writer.add_scalar('AvgRew', rew_mean, global_step=step)
            summary_writer.add_scalar('AvgEpLen', len_mean, global_step=step)
            summary_writer.add_scalar('Episodes', episode_count, global_step=step)

        # Save
        if step % SAVE_INTERVAL == 0 and step > resume_step:
            print()
            print("Saving model...")
            online_net.save(SAVE_PATH, step, episode_count, rew_mean, len_mean)
            print("OK!")

