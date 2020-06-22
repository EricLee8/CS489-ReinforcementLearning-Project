# CS489 Reinforcement Learning: Project

## Project Introduction

This is the code for the project of CS489: Reinforce Learning.  
This project requires us to implement two kinds of model-free RL methods, which are value-based RL and policy-based RL. We should choose RL algorithms to solve two benchmark environments: Atari Game and Mujoco Robots, which are discrete space control and continuous space control, respectively.  

## How to Run

To install the dependencies, run:  

`$ pip install -r requirements.txt` 

Note that gym[atari], tb-nightly future and mujoco_py can not be install by running the above code, for the first two, run:  

`$ pip install gym[atari]`

`$ pip install tb-nightly future`

For mujoco_py, you need to get a license.  

After installing all dependencies, you can run my code as:  

`$ python run.py --env_name BreakoutNoFrameskip-v4`

Note that my code only support the following 7 environments!

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
### Atari Games
<table>
    <tr>
        <td ><center><img src="https://github.com/EricLee8/CS489-ReinforcementLearning-Project/blob/master/atari_pics/Breakout.png"></center></td>
        <td ><center><img src="https://github.com/EricLee8/CS489-ReinforcementLearning-Project/blob/master/atari_pics/Boxing.png"></center></td>
      <td ><center><img src="https://github.com/EricLee8/CS489-ReinforcementLearning-Project/blob/master/atari_pics/Pong.png"></center></td>
    </tr>
</table>  

### Mujoco Robots
<table>
    <tr>
        <td ><center><img src="https://github.com/EricLee8/CS489-ReinforcementLearning-Project/blob/master/mujoco_pics/ant.png"></center></td>
        <td ><center><img src="https://github.com/EricLee8/CS489-ReinforcementLearning-Project/blob/master/mujoco_pics/half.png"></center></td>
    </tr>
</table>  
<table>
    <tr>
      <td ><center><img src="https://github.com/EricLee8/CS489-ReinforcementLearning-Project/blob/master/mujoco_pics/human.png"></center></td>
      <td ><center><img src="https://github.com/EricLee8/CS489-ReinforcementLearning-Project/blob/master/mujoco_pics/hopper.png"></center></td>
    </tr>
</table>
