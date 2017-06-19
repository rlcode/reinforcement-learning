<p align="center"><img width="90%" src="images/Reinforcement-Learning.png" /></p>

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
6. pillow
7. matplot
8. Skimage
9. h5py

### 설치 방법 (Install Requirements)
```
pip install -r requirements.txt
```


## 목차 (Table of Contents)

**Grid World** - 비교적 단순한 환경인 그리드월드에서 강화학습의 기초를 쌓기

- [정책 이터레이션 (Policy Iteration)](./1-grid-world/1-policy-iteration)
- [가치 이터레이션 (Value Iteration)](./1-grid-world/2-value-iteration)
- [몬테카를로 (Monte Carlo)](./1-grid-world/3-monte-carlo)
- [살사 (SARSA)](./1-grid-world/4-sarsa)
- [큐러닝 (Q-Learning)](./1-grid-world/5-q-learning)
- [Deep SARSA](./1-grid-world/6-deep-sarsa)
- [REINFORCE](./1-grid-world/7-reinforce)

**CartPole** - 카트폴 예제를 이용하여 여러가지 딥러닝을 강화학습에 응용한 알고리즘들을 적용해보기

- [Deep Q Network](./2-cartpole/1-dqn)
- [Double Deep Q Network](./2-cartpole/2-double-dqn)
- [Policy Gradient](./2-cartpole/3-reinforce)
- [Actor Critic (A2C)](./2-cartpole/4-actor-critic)
- [Asynchronous Advantage Actor Critic (A3C)](./2-cartpole/5-a3c)

**Atari** - 딥러닝을 응용하여 좀더 복잡한 Atari게임을 마스터하는 에이전트 만들기

- **벽돌깨기(Breakout)** - [DQN](./3-atari/1-breakout/breakout_dqn.py), [DDQN](./3-atari/1-breakout/breakout_ddqn.py) [Dueling DDQN](./3-atari/1-breakout/breakout_ddqn.py) [A3C](./3-atari/1-breakout/breakout_a3c.py)
- **퐁(Pong)** - [Policy Gradient](./3-atari/2-pong/pong_reinforce.py), [A3C](./3-atari/2-pong/pong-a3c.py)

**OpenAI GYM** - [WIP]

- Mountain Car - [DQN](./4-gym/1-mountaincar)

