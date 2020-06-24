import os
import argparse
import numpy as np
import torch
import gym
from gym import wrappers

from model import GaussianPolicy
from utils import grad_false

import warnings
warnings.filterwarnings('ignore')


def testing(num_episode=10):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    policy = GaussianPolicy(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        hidden_units=[256, 256]).to(device)

    policy.load(os.path.join('models', args.env_name, 'policy.pth'))
    grad_false(policy)

    def exploit(state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, action = policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    e_rewrads = []
    for _ in range(num_episode):
        state = env.reset()
        episode_reward = 0.
        done = False
        while not done:
            # env.render()
            action = exploit(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        e_rewrads.append(episode_reward)
    print("Average reward of " + args.env_name + "is %.1f"%(np.mean(e_rewrads)))
    print("Average std of " + args.env_name + "is %.1f"%(np.std(e_rewrads)))


if __name__ == '__main__':
    testing()
