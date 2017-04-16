import tensorflow as tf
import numpy as np
import gym


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev= 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 4])

fc_W1 = weight_variable([4,10])
fc_b1 = bias_variable([10])

fc_W2 = weight_variable([10,10])
fc_b2 = bias_variable([10])

fc_W3 = weight_variable([10,2])
fc_b3 = bias_variable([2])


h1 = tf.nn.tanh(tf.matmul(x, fc_W1) + fc_b1)
h2 = tf.nn.tanh(tf.matmul(h1, fc_W2) + fc_b2)
out = tf.nn.softmax(tf.matmul(h2, fc_W3) + fc_b3)

act = tf.placeholder(tf.float32, [None, 2])
rwd = tf.placeholder(tf.float32, [None, ])

good_prob = tf.reduce_sum(tf.mul(out,act), reduction_indices=[1])
egibility = tf.log(good_prob)*rwd
loss = -tf.reduce_sum(egibility)
train = tf.train.RMSPropOptimizer(1e-2).minimize(loss)

def get_action(obs):
    action = out.eval(feed_dict={x: obs})
    if action[0][0] > np.random.random():
        return 0
    else:
        return 1

def discounted_reward(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * 0.99 + r[t]
        discounted_r[t] = running_add

    return discounted_r

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

env = gym.make('CartPole-v0')
obs = env.reset()
_epi, reward_sum = 0,0
reward_avg = 0
s, a, r = [],[],[]
is_train = True

for i in range(10000):
    while True:
        if not is_train:
            env.render()
        s.append(obs)
        action = get_action(np.array([obs]))
        action_array = np.zeros(2)

        if action == 0:
            action_array[0] = 1
        else:
            action_array[1] = 1

        a.append(action_array)
        obs, reward, done, _ = env.step(action)
        r.append(reward)
        reward_sum += reward

        if done or reward_sum >= 1000:
            reward_avg = reward_avg*0.9 + reward_sum*0.1
            print ("Episode %i finished. Reward is %i. Average Reward is %f" \
                   %(_epi, reward_sum, reward_avg))

            obs = env.reset()
            reward_sum=0
            _epi += 1

            discount_r = discounted_reward(r)
            discount_r -= np.mean(discount_r)
            discount_r /= np.std(discount_r)

            if is_train:
                _, step_result = sess.run([train, out], feed_dict = \
                                          {x: s, act: a, rwd: discount_r})
            s,a,r = [],[],[]

            if reward_avg > 800:
                saver.save(sess, "../openai_upload/cartpole.ckpt")
                is_train = False
            break
