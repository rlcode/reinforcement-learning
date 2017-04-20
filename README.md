<p align="center"><img width="90%" src="images/Reinforcement-Learning.png"></p>

--------------------------------------------------------------------------------

> Minimal and clean examples of reinforcement learning algorithms presented by [RLCode](https://rlcode.github.io) team. [[한국어]](./README-kr.md)
>
> Maintainers - [Woongwon](https://github.com/dnddnjs), [Youngmoo](https://github.com/zzing0907), [Hyeokreal](https://github.com/Hyeokreal), [Uiryeong](https://github.com/wooridle), [Keon](https://github.com/keon)

From the most basic algorithms to the more recent ones categorized as 'deep reinforcement learning', the examples are easy to read with comments.
Please feel free to create a [Pull Request](https://github.com/rlcode/reinforcement-learning/pulls), or open an [issue](https://github.com/rlcode/reinforcement-learning/issues)!

## Dependencies
1. Python 3.5
2. Tensorflow 1.0.0
3. Keras 
4. numpy
5. pandas
6. matplot

### Install Requirements
```
pip install -r requirements.txt
```

## Table of Contents

**Code 1** - Mastering the basics of reinforcement learning in the simplified world called "Grid World"

- [Policy Iteration](./Code%201.%20Grid%20World/1.%20Policy%20Iteration)
- [Value Iteration](./Code%201.%20Grid%20World/2.%20Value%20Iteration)
<<<<<<< Updated upstream
- [Monte Carlo](./Code%201.%20Grid%20World/3.%20Monte-Carlo)
- [SARSA](./Code%201.%20Grid%20World/4.%20SARSA)
- [Q-Learning](./Code%201.%20Grid%20World/5.%20Q%20Learning)
=======
- [SARSA](Code 1. Grid World/4. SARSA)
- [Q-Learning](Code 1. Grid World/5. Q Learning)
- [Monte Carlo](Code 1. Grid World/3. Monte-Carlo)
>>>>>>> Stashed changes
- [Deep Q Network](./Code%201.%20Grid%20World/6.%20DQN)
- [Policy Gradient](./Code%201.%20Grid%20World/7.%20Policy%20Gradient)

**Code 2** - Applying deep reinforcement learning on basic Cartpole game.

- [Deep Q Network](./Code%202.%20Cartpole/1.%20DQN)
- [Double Deep Q Network](./Code%202.%20Cartpole/2.%20Double%20DQN)
- [Dueling Deep Q Network](./Code%202.%20Cartpole/3.%20Dueling%20DQN)
- [Policy Gradient](./Code%202.%20Cartpole/4.%20Policy%20Gradient)
- [Actor Critic](./Code%202.%20Cartpole/5.%20Actor-Critic)
- Asynchronous Advantage Actor Critic (A3C) - WIP

**Code 3** - Mastering Atari games with Deep Reinforcement Learning

- [Breakout](./Code%203.%20Atari%20Game/1.%20Breakout) - DQN, PG, A3C
- [Pong](./Code%203.%20Atari%20Game/2.%20Pong) - DQN, PG, A3C
