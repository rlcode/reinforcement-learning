import os
from waterworld import WaterWorld
from ple import PLE
from pygame.constants import K_w, K_a, K_s, K_d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

matplotlib.use("Agg")

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"


tf.set_random_seed(0)


def process_state(state):
    return np.array([state.values()])


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 28])

fc_W1 = weight_variable([28, 50])
fc_b1 = bias_variable([50])

fc_W2 = weight_variable([50, 50])
fc_b2 = bias_variable([50])

fc_W3 = weight_variable([50, 5])
fc_b3 = bias_variable([5])


v_fc_W1 = weight_variable([28, 50])
v_fc_b1 = bias_variable([50])

v_fc_W2 = weight_variable([50, 50])
v_fc_b2 = bias_variable([50])

v_fc_W3 = weight_variable([50, 1])
v_fc_b3 = bias_variable([1])


h1 = tf.nn.tanh(tf.matmul(x, fc_W1) + fc_b1)
h2 = tf.nn.tanh(tf.matmul(h1, fc_W2) + fc_b2)
out = tf.nn.softmax(tf.matmul(h2, fc_W3) + fc_b3)

v_h1 = tf.nn.tanh(tf.matmul(x, v_fc_W1) + v_fc_b1)
v_h2 = tf.nn.tanh(tf.matmul(v_h1, v_fc_W2) + v_fc_b2)
v_out = tf.matmul(v_h2, v_fc_W3) + v_fc_b3

act = tf.placeholder(tf.float32, [None, 5])
Q = tf.placeholder(tf.float32, [None, ])


v_loss = tf.reduce_sum(tf.square(v_out - Q))
v_train = tf.train.RMSPropOptimizer(0.0001).minimize(v_loss)

good_prob = tf.reduce_sum(tf.multiply(out, act), reduction_indices=[1])
eligibility = tf.log(good_prob) * tf.subtract(Q, v_out)
loss = -tf.reduce_sum(eligibility)
train = tf.train.AdamOptimizer(0.0001).minimize(loss)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

def dic_to_list(state):
    temp_state = []

    for key, value in state.items():
        if key == 'creeps':
            for input in value:
                temp_state.append(input)
        else:
            temp_state.append(value)
    return temp_state


def get_action(obs):
    action = out.eval(feed_dict={x: obs})
    act_prob.append(np.amax(action[0]))
    rand_num = np.random.random()
    if action[0][0] > rand_num:
        return K_w
    elif sum(action[0][:2]) > rand_num:
        return K_a
    elif sum(action[0][:3]) > rand_num:
        return K_s
    elif sum(action[0][:4]) > rand_num:
        return K_d
    else:
        return None

def discounted_reward(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * 0.99 + r[t]
        discounted_r[t] = running_add

    return discounted_r


game = WaterWorld(width=256, height=256, num_creeps=8)
p = PLE(game, display_screen=True, state_preprocessor=process_state)

p.init()
reward, totalReward, step, epi = 0, 0, 0, 0
actions, rewards, states, r = [], [], [], []
act_prob, probs = [], []
#plt.ion()
plt.figure(1)
plt.subplot(211)
plt.xlabel("episodes")
plt.ylabel("rewards")
plt.subplot(212)
plt.xlabel("episodes")
plt.ylabel("probability")
scores, time = [], []
step = 0
while True:
    step+=1
    if p.game_over() or step % 3000 == 0:
        epi += 1
        print("episode : ", epi, "reward : ", totalReward, "step : ", step)

        discountReward = discounted_reward(rewards)

        # print (discountReward)

        r = np.concatenate((r, discountReward), axis=0)

        if epi % 5 == 0:
            _ = sess.run(v_train, feed_dict={x: states,
                                             Q: r})

            _ = sess.run(train, feed_dict={x: states,
                                           Q: r,
                                           act: actions})

            r, actions, states = [], [], []

        scores.append(totalReward)
        probs.append(sum(act_prob)/step)
        time.append(epi)

        if epi % 10 == 0:
            plt.subplot(211)
            plt.plot(time, scores, 'b-')
            plt.subplot(212)
            plt.plot(time, probs, 'b-')
            plt.savefig("./save/waterworld_PG.png")


            print (step_loss)
            if epi % 100 == 0:
                pass
                saver.save(sess, "./save/waterworld_PG")


        rewards, act_prob = [], []
        totalReward = 0
        step = 0
        p.reset_game()

    state = game.getGameState()
    state = dic_to_list(state)
    states.append(state)
    action = get_action([state])
    if action == K_w:
        actions.append([1, 0, 0, 0, 0])
    elif action == K_a:
        actions.append([0, 1, 0, 0, 0])
    elif action == K_s:
        actions.append([0, 0, 1, 0, 0])
    elif action == K_d:
        actions.append([0, 0, 0, 1, 0])
    else:
        actions.append([0, 0, 0, 0, 1])


    reward = p.act(action)
    totalReward += reward
    rewards.append(reward)
