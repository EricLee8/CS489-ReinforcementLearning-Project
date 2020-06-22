# CS489 Reinforcement Learning: Project

## Project Introduction

This is the code for the project of CS489: Reinforce Learning.  
This project requires us to implement two kinds of model-free RL methods, which are value-based RL and policy-based RL. We should choose RL algorithms to solve two benchmark environments: Atari Game and Mujoco Robots, which are discrete space control and continuous space control, respectively.  

## Methods

For Atari Games, I chose DQN with some optimization, such as Losing-life-stopping (especially works for Breakout) and Skip-frame. For Mujoco Robots, I firstly tried A3C and PPO (PPO2) but got bad results. Finally I used SAC (Soft Actor-Critic) and got better results.

## Results

The results are presented as follows. Note that due to the shortage of time and computing resource, some environment should have reached better results. For example, after 3M steps, Humanoid-v2 model can still improve its performance if we continue to train it. But time is not enough for me to do that, so I stopped it at the point of 3M steps.

### Atari Games

|    Environment Name    | Best Testing Score |
| :--------------------: | :----------------: |
| BreakoutNoFrameskip-v4 |       398.0        |
|   PongNoFrameskip-v4   |        20.0        |
|  BoxingNoFrameskip-v4  |        92.3        |

### Mujoco Robots

| Evironment Name | Best Testing Score |
| :-------------: | :----------------: |
|    Hopper-v2    |       3777.3       |
|   Humanoid-v2   |       6422.1       |
| HalfCheetah-v2  |      15875.6       |
|     Ant-v2      |       6978.2       |

## Training Reward Pictures

