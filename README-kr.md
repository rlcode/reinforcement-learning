<p align="center"><img width="90%" src="/images/Reinforcement-Learning.png" /></p>

--------------------------------------------------------------------------------

> [RLCode](https://rlcode.github.io)팀이 직접 만든 강화학습 예제들을 모아놓은 Repo 입니다. [영문 (English)](./README.md)
>
> Maintainers - [이웅원](https://github.com/dnddnjs), [이영무](https://github.com/zzing0907), [양혁렬](https://github.com/Hyeokreal), [이의령](https://github.com/wooridle), [김건우](https://github.com/keon)

[Pull Request](https://github.com/rlcode/reinforcement-learning/pulls)는 언제든 환영입니다.
문제나 버그, 혹은 궁금한 사항이 있으면 [이슈](https://github.com/rlcode/reinforcement-learning/issues)에 글을 남겨주세요.


## 필요한 라이브러리들 (Dependencies)
1. Python 3.5
2. Tensorflow 1.0.0
3. Keras 
4. numpy
5. pandas
6. matplot

## 목차 (Table of Contents)

**Code 1** - 비교적 단순한 환경인 그리드월드에서 강화학습의 기초를 쌓기

- [정책 이터레이션 (Policy Iteration)](./Code%201.%20Grid%20World/1.%20Policy%20Iteration)
- [가치 이터레이션 (Value Iteration)](./Code%201.%20Grid%20World/2.%20Value%20Iteration)
- [살사 (SARSA)](./Code%201.%20Grid%20World/3.%20SARSA)
- [큐러닝 (Q-Learning)](./Code%201.%20Grid%20World/4.%20Q%20Learning)
- [몬테카를로 (Monte Carlo)](./Code%201.%20Grid%20World/5.%20Monte-Carlo)
- [Deep Q Network](./Code%201.%20Grid%20World/6.%20DQN)
- [Policy Gradient](./Code%201.%20Grid%20World/7.%20Policy%20Gradient)

**Code 2** - 카트폴 예제를 이용하여 여러가지 딥러닝을 강화학습에 응용한 알고리즘들을 적용해보기

- [Deep Q Network](./Code%202.%20Cartpole/1.%20DQN)
- [Double Deep Q Network](./Code%202.%20Cartpole/2.%20Double%20DQN)
- [Dueling Deep Q Network](./Code%202.%20Cartpole/3.%20Dueling%20DQN)
- [Policy Gradient](./Code%202.%20Cartpole/4.%20Policy%20Gradient)
- [Actor Critic](./Code%202.%20Cartpole/5.%20Actor-Critic)

**Code 3** - 딥러닝을 응용하여 좀더 복잡한 Atari게임을 마스터하는 에이전트 만들기

- [벽돌깨기 (Breakout)](./Code%203.%20Atari%20Game/1.%20Breakout)
- [퐁 (Pong)](./Code%203.%20Atari%20Game/2.%20Pong)
