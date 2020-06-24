import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4')
args = parser.parse_args()

assert args.env_name in ["BreakoutNoFrameskip-v4", "PongNoFrameskip-v4", \
    "BoxingNoFrameskip-v4", "Hopper-v2", "Humanoid-v2", "HalfCheetah-v2", "Ant-v2"],\
        "Environment " + args.env_name + " is not supported!!!"

if args.env_name in ["BreakoutNoFrameskip-v4", "PongNoFrameskip-v4", "BoxingNoFrameskip-v4"]:
    os.chdir("atari")
    os.system("python demo.py --env_name " + args.env_name)

else:
    os.chdir("mujoco")
    os.system("python demo.py --env_name " + args.env_name)
