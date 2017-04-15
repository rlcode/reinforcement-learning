
'''
This code is based on:
https://github.com/hunkim/DeepRL-Agents
CF https://github.com/golbin/TensorFlow-Tutorials
https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/dqn.py
'''
import numpy as np
import tensorflow as tf
import random
from collections import deque
import dqn

import gym
from gym import wrappers

env = gym.make('CartPole-v0')

# Constants defining our neural network
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis = 0.9
REPLAY_MEMORY = 50000


def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        # terminal?
        if done:
            Q[0, action] = reward
        else:
            # get target from target DQN (Q')
            Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack)


def ddqn_replay_train(mainDQN, targetDQN, train_batch):
    '''
    Double DQN implementation
    :param mainDQN: main DQN
    :param targetDQN: target DQN
    :param train_batch: minibatch for train
    :return: loss
    '''
    x_stack = np.empty(0).reshape(0, mainDQN.input_size)
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)

    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        # terminal?
        if done:
            Q[0, action] = reward
        else:
            # Double DQN: y = r + gamma * targetDQN(s')[a] where
            # a = argmax(mainDQN(s'))
            Q[0, action] = reward + dis * \
                targetDQN.predict(next_state)[
                0, np.argmax(mainDQN.predict(next_state))]

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack)


def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):

    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def bot_play(mainDQN, env=env):
    # See our trained network in action
    state = env.reset()
    reward_sum = 0
    while True:
        env.render()
        action = np.argmax(mainDQN.predict(state))
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break


def main():
    max_episodes = 5000
    # store the previous observations in replay memory
    replay_buffer = deque()

    last_100_game_reward = deque()

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
        targetDQN = dqn.DQN(sess, input_size, output_size, name="target")
        tf.global_variables_initializer().run()

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(max_episodes):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0
            state = env.reset()

            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)
                if done:  # Penalty
                    reward = -100

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1

            print("Episode: {}  steps: {}".format(episode, step_count))

            if episode % 10 == 1:  # train every 10 episode
                # Get a random batch of experiences.
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = ddqn_replay_train(mainDQN, targetDQN, minibatch)

                print("Loss: ", loss)
                # copy q_net -> target_net
                sess.run(copy_ops)

            last_100_game_reward.append(step_count)

            if len(last_100_game_reward) > 100:
                last_100_game_reward.popleft()

                avg_reward = np.mean(last_100_game_reward)

                if avg_reward > 199:
                    print("Game Cleared in {episode} episodes with avg reward {avg_reward}")
                    break

        # See our trained bot in action
        env2 = wrappers.Monitor(env, 'gym-results', force=True)

        for i in range(200):
            bot_play(mainDQN, env=env2)

        env2.close()
        gym.upload("gym-results", api_key="sk_VT2wPcSSOylnlPORltmQ")


if __name__ == "__main__":
    main()