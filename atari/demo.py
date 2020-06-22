import math
import random
import time
import numpy as np
import os
from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import model as m
from atari_wrappers import wrap_deepmind, make_atari

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if gpu is to be used


def demo(num_episode=1):
    eps = 0.01
    env_raw = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env_raw)
    c, h, w = m.fp(env.reset()).shape
    n_actions = env.action_space.n
    policy_net = m.DQN(h, w, n_actions, device).to(device)
    policy_net.load_state_dict(torch.load("models/best4_375.pth"))
    policy_net.eval()
    sa = m.ActionSelector(eps, eps, policy_net, 100, n_actions, device)
    q = deque(maxlen=5)
    e_rewards = []
    for _ in range(num_episode):
        env.reset()
        e_reward = 0
        for _ in range(5): # no-op
            n_frame, _, done, _ = env.step(0)
            n_frame = m.fp(n_frame)
            q.append(n_frame)
        
        while not done:
            # if num_episode <= 3:
            #     env.render()
            #     # time.sleep(0.03)
            state = torch.cat(list(q))[1:].unsqueeze(0)
            action, eps = sa.select_action(state, False)
            n_frame, reward, done, _ = env.step(action)
            n_frame = m.fp(n_frame)
            q.append(n_frame)
            e_reward += reward
            # print(e_reward)
            # print(action)

        e_rewards.append(e_reward)
    avg_reward = float(sum(e_rewards))/float(num_episode)
    env.close()
    print(avg_reward)


if __name__ == "__main__":
    demo(100)