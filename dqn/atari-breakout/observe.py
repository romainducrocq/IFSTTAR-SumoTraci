import numpy as np
import torch
import itertools
from baselines_wrappers import DummyVecEnv
from atari_breakout_dqn import Network, SAVE_PATH, NUM_ENVS
from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames
import time

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    make_env = lambda: make_atari_deepmind('BreakoutNoFrameskip-v4', scale_values=True)
    vec_env = DummyVecEnv([make_env for _ in range(1)])
    env = BatchedPytorchFrameStack(vec_env, k=NUM_ENVS)

    net = Network(env, device)
    net = net.to(device)

    _, _, _, _ = net.load(SAVE_PATH)

    obs = env.reset()
    beginning_episode = True
    for t in itertools.count():
        if isinstance(obs[0], PytorchLazyFrames):
            act_obs = np.stack([o.get_frames() for o in obs])
            action = net.act(act_obs, 0.0)
        else:
            action = net.act(obs, 0.0)

        if beginning_episode:
            action = [1]
            beginning_episode = False

        obs, rew, done, _ = env.step(action)
        env.render()
        time.sleep(1/30)

        if done[0]:
            obs = env.reset()
            beginning_episode = True
