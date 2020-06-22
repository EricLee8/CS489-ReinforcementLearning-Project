from matplotlib import pyplot as plt
import numpy as np
import os

names = ["Boxing", "Breakout", "Pong"]

def get_rewards_from_file(name):
    rewards = []
    stds = []
    steps = []
    i = 0
    with open("atari/"+name+".csv", "r") as f:
        for line in f:
            i += 1
            info = line.replace(' ','').split(',')
            r = float(info[0])
            # std = float(info[1]) # 暂时没有加上这个信息
            std = 5
            rewards.append(r)
            stds.append(std)
            steps.append(i)
    return rewards, stds, steps


def draw_one_env(name, interval=5):
    rewards, stds, steps = get_rewards_from_file(name)
    r1 = list(map(lambda x: x[0]-x[1], zip(rewards, stds)))
    r2 = list(map(lambda x: x[0]+x[1], zip(rewards, stds)))
    plt.fill_between(steps, r1, r2, alpha=0.35)
    plt.plot(steps, rewards, label=name+"NoFrameskip-v4")
    plt.grid()
    plt.xlabel("Number of training steps (*50000)")
    plt.ylabel("Testing reward")
    plt.legend()
    plt.savefig("atari_pics/"+name+".png", dpi=250)


if __name__ == "__main__":
    if not os.path.exists("atari_pics"):
        os.makedirs("atari_pics")
    for i in range(3):
        draw_one_env(names[i])
        plt.clf()