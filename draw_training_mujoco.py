from matplotlib import pyplot as plt
import numpy as np
import json
import os

names = ["ant", "half", "hopper", "human"]
real_names = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Humanoid-v2"]


def compute_r_std(f):
    line = f.readline()
    line = f.readline()
    rs = []
    while not '------' in line and line:
        line = f.readline()
        if '------' in line or not line:
            break
        r = float(line[line.find("reward:"):].replace("reward: ", ""))
        rs.append(r)
    rs = np.array(rs)
    if len(rs) == 0:
        print(rs)
    return rs.mean(), rs.std()


def get_train_data(name):
    rewards = []
    steps = []
    stds = []
    with open("mujoco/results/" + name + ".log") as f:
        line = f.readline()
        while line:
            line = f.readline()
            if "Num steps" in line or "warnings" in line:
                if "Num steps" in line and len(rewards)>=1:
                    test_r = float(line[line.find("reward:"):].replace("reward: ", ""))
                    rewards[-1] = max(rewards[-1], test_r)
                r, std = compute_r_std(f)
                step_num = int(line[:line.find("reward:")].strip().replace("Num steps: ", "")) if "Num steps" in line else 0
                steps.append(step_num/10000)
                rewards.append(r)
                stds.append(std)
    rewards.remove(rewards[-1])
    stds.remove(stds[-1])
    steps.remove(steps[0])
    return rewards, stds, steps


def draw_env(name, i):
    rewards, stds, steps = get_train_data(name)
    r1 = list(map(lambda x: x[0]-x[1], zip(rewards, stds)))
    r2 = list(map(lambda x: x[0]+x[1], zip(rewards, stds)))
    plt.fill_between(steps, r1, r2, alpha=0.35)
    plt.plot(steps, rewards, label=real_names[i]+"-SAC")
    print(name + " Mean std: %.1f" %(np.std(stds)))

    f = open("mujoco/results/"+name+"_ppo.json", "r")
    loadDict = json.load(f)
    rewards, stds = loadDict["rewards"], loadDict["stds"]
    r1 = list(map(lambda x: x[0]-x[1], zip(rewards, stds)))
    r2 = list(map(lambda x: x[0]+x[1], zip(rewards, stds)))
    plt.fill_between(steps, r1, r2, alpha=0.35)
    plt.plot(steps, rewards, label=real_names[i]+"-PPO")
    plt.grid()
    plt.xlabel("Number of training steps (*10000)")
    plt.ylabel("Testing reward")
    plt.legend()
    plt.savefig("mujoco_pics/"+name+".png", dpi=250)


def draw_env_all():
    for i in range(4):
        rewards, stds, steps = get_train_data(names[i])
        r1 = list(map(lambda x: x[0]-x[1], zip(rewards, stds)))
        r2 = list(map(lambda x: x[0]+x[1], zip(rewards, stds)))
        plt.fill_between(steps, r1, r2, alpha=0.25)
        plt.plot(steps, rewards, label=real_names[i])
    plt.grid()
    plt.xlabel("Number of training steps (*10000)")
    plt.ylabel("Testing reward")
    plt.legend()
    plt.savefig("mujoco_pics/all.png", dpi=250)


if __name__ == "__main__":
    if not os.path.exists("mujoco_pics"):
        os.makedirs("mujoco_pics")
    for i in range(4):
        draw_env(names[i], i)
        plt.clf()