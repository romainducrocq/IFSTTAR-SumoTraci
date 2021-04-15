import gym
from gym.utils.play import play

if __name__ == "__main__":

    play(gym.make('BreakoutNoFrameskip-v4'), zoom=3)
