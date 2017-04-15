import argparse
import gym
from gym.spaces import Box, Discrete
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np


def createLayers():
    x = Input(shape=env.observation_space.shape)
    if args.batch_norm:
        h = BatchNormalization()(x)
    else:
        h = x
    for i in range(args.layers):
        h = Dense(args.hidden_size, activation=args.activation)(h)
        if args.batch_norm and i != args.layers - 1:
            h = BatchNormalization()(h)
    y = Dense(env.action_space.n + 1)(h)
    if args.advantage == 'avg':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                   output_shape=(env.action_space.n,))(y)
    elif args.advantage == 'max':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                   output_shape=(env.action_space.n,))(y)
    elif args.advantage == 'naive':
        z = Lambda(lambda a: K.expand_dims(a[:, 0], dim=-1) + a[:, 1:], output_shape=(env.action_space.n,))(y)
    else:
        assert False

    return x, z


parser = argparse.ArgumentParser()
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--batch_norm', action="store_true", default=False)
parser.add_argument('--no_batch_norm', action="store_false", dest='batch_norm')
parser.add_argument('--replay_start_size', type=int, default=50000)
parser.add_argument('--train_repeat', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--episodes', type=int, default=1000)
parser.add_argument('--max_timesteps', type=int, default=200)
parser.add_argument('--activation', choices=['tanh', 'relu'], default='tanh')
parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam')
# parser.add_argument('--optimizer_lr', type=float, default=0.001)
parser.add_argument('--exploration', type=float, default=0.1)
parser.add_argument('--advantage', choices=['naive', 'max', 'avg'], default='naive')
parser.add_argument('--display', action='store_true', default=True)
parser.add_argument('--no_display', dest='display', action='store_false')
parser.add_argument('--gym_record')
parser.add_argument('--update_frequency', type=int, default=4)
parser.add_argument('--target_net_update_frequency', type=int, default=32)
parser.add_argument('--replay_memory_size', type=int, default=1000000)

parser.add_argument('environment')
args = parser.parse_args()

env = gym.make(args.environment)
assert isinstance(env.observation_space, Box)
assert isinstance(env.action_space, Discrete)

if args.gym_record:
    env.monitor.start(args.gym_record, force=True)

x, z = createLayers()
model = Model(input=x, output=z)
model.summary()
model.compile(optimizer='adam', loss='mse')

x, z = createLayers()
target_model = Model(input=x, output=z)
target_model.set_weights(model.get_weights())

prestates = []
actions = []
rewards = []
poststates = []
terminals = []

total_reward = 0
timestep = 0

for i_episode in range(args.episodes):
    observation = env.reset()
    episode_reward = 0
    for t in range(args.max_timesteps):
        if args.display:
            env.render()

        if timestep < args.replay_start_size or np.random.random() < args.exploration:
            action = env.action_space.sample()
            if args.verbose > 0:
                print("e:", i_episode, "e.t:", t, "action:", action, "random")
        else:
            s = np.array([observation])
            q = model.predict(s, batch_size=1)
            action = np.argmax(q[0])
            if args.verbose > 0:
                print("e:", i_episode, "e.t:", t, "action:", action, "q:", q)

        if len(prestates) >= args.replay_memory_size:
            delidx = np.random.randint(0, len(prestates) - 1 - args.batch_size)
            del prestates[delidx]
            del actions[delidx]
            del rewards[delidx]
            del poststates[delidx]
            del terminals[delidx]

        prestates.append(observation)
        actions.append(action)

        observation, reward, done, info = env.step(action)
        episode_reward += reward
        if args.verbose > 1:
            print("reward:", reward)

        rewards.append(reward)
        poststates.append(observation)
        terminals.append(done)

        timestep += 1

        if timestep > args.replay_start_size:
            if timestep % args.update_frequency == 0:
                for k in xrange(args.train_repeat):
                    if len(prestates) > args.batch_size:
                        # indexes = range(args.batch_size)
                        # indexes = np.random.choice(len(prestates), size=args.batch_size)
                        indexes = np.random.randint(len(prestates), size=args.batch_size)
                    else:
                        indexes = range(len(prestates))

                    pre_sample = np.array([prestates[i] for i in indexes])
                    post_sample = np.array([poststates[i] for i in indexes])
                    qpre = model.predict(pre_sample)
                    qpost = target_model.predict(post_sample)
                    for i in xrange(len(indexes)):
                        if terminals[indexes[i]]:
                            qpre[i, actions[indexes[i]]] = rewards[indexes[i]]
                        else:
                            qpre[i, actions[indexes[i]]] = rewards[indexes[i]] + args.gamma * np.amax(qpost[i])
                    model.train_on_batch(pre_sample, qpre)

            if timestep % args.target_net_update_frequency == 0:
                if args.verbose > 0:
                    print('timestep:', timestep, 'DDQN: Updating weights')
                weights = model.get_weights()
                target_model.set_weights(weights)
                    # weights = model.get_weights()
                    # target_weights = target_model.get_weights()
                    # for i in xrange(len(weights)):
                    #     weights[i] *= args.tau
                    #     target_weights[i] *= (1 - args.tau)
                    #     target_weights[i] += weights[i]
                    # target_model.set_weights(target_weights)

        if done:
            break

    print("Episode {} finished after {} timesteps, episode reward {}".format(i_episode + 1, t + 1, episode_reward))
    total_reward += episode_reward

print("Average reward per episode {}".format(total_reward / args.episodes))

if args.gym_record:
    env.monitor.close()
