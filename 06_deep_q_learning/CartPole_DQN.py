import gym
import numpy as np
import tensorflow as tf
import random

in_size = 4
out_size = 2
experience_size = 2000

class DQNAgent:
    def __init__(self):
        self.gamma = 0.99
        self.decay_rate = 0.99
        self.eps_decay = 0.95
        self.epsilon = 1
        self.learning_rate = 1e-4
        self.learning = True
        self.exp_batch_size = 256
        self.hidden1 = 160
        self.hidden2 = 160
        self.hidden3 = 160

        np.random.seed(0)
        tf.set_random_seed(0)

        self.W1 = self.weight_variable([in_size, self.hidden1])
        self.b1 = self.bias_variable([self.hidden1])
        self.W2 = self.weight_variable([self.hidden1, self.hidden2])
        self.b2 = self.bias_variable([self.hidden2])
        self.W3 = self.weight_variable([self.hidden2, self.hidden3])
        self.b3 = self.bias_variable([self.hidden3])
        self.W4 = self.weight_variable([self.hidden3, out_size])
        self.b4 = self.bias_variable([out_size])

        self.W1_ = tf.Variable(self.W1.initialized_value(), trainable=False)
        self.b1_ = tf.Variable(self.b1.initialized_value(), trainable=False)
        self.W2_ = tf.Variable(self.W2.initialized_value(), trainable=False)
        self.b2_ = tf.Variable(self.b2.initialized_value(), trainable=False)
        self.W3_ = tf.Variable(self.W3.initialized_value(), trainable=False)
        self.b3_ = tf.Variable(self.b3.initialized_value(), trainable=False)
        self.W4_ = tf.Variable(self.W4.initialized_value(), trainable=False)
        self.b4_ = tf.Variable(self.b4.initialized_value(), trainable=False)


    def weight_variable(self, shape):
        return tf.Variable(self.xavier_initializer(shape))


    def bias_variable(self, shape):
        return tf.Variable(self.xavier_initializer(shape))

    def xavier_initializer(self, shape):
        bound = 1 / np.sqrt(np.sum(shape))
        return tf.random_uniform(shape, minval=-bound, maxval=bound)


    def get_Q_val(self):
        obs = tf.placeholder(tf.float32, [None, in_size])
        h1 = tf.nn.tanh(tf.matmul(obs, self.W1) + self.b1)
        h2 = tf.nn.tanh(tf.matmul(h1, self.W2) + self.b2)
        h3 = tf.nn.tanh(tf.matmul(h2, self.W3) + self.b3)
        Q = tf.matmul(h3, self.W4) + self.b4
        return obs, Q

    def target_Q_val(self):
        obs = tf.placeholder(tf.float32, [None, in_size])
        h1 = tf.nn.tanh(tf.matmul(obs, self.W1_) + self.b1_)
        h2 = tf.nn.tanh(tf.matmul(h1, self.W2_) + self.b2_)
        h3 = tf.nn.tanh(tf.matmul(h2, self.W3_) + self.b3_)
        prev_Q = tf.matmul(h3, self.W4_) + self.b4_
        return obs, prev_Q

    def prepro(self, x):
        return x[1:]

    def get_action(self, Q, feed):
        act_values = Q.eval(feed_dict = feed)

        if np.random.uniform() <= self.epsilon:
            act = random.randrange(out_size)
        else:
            act = np.argmax(act_values)

        action = np.zeros_like(act_values[0])
        action[act] = 1
        return action


def learning(env):
    agent = DQNAgent()
    sess = tf.InteractiveSession()

    obs, Q1 = agent.get_Q_val()
    act = tf.placeholder(tf.float32, [None, out_size])
    rwd = tf.placeholder(tf.float32, [None,])
    next_obs, Q2 = agent.target_Q_val()

    y = tf.reduce_sum(tf.mul(Q1, act), reduction_indices=1)
    t = rwd + agent.gamma*tf.reduce_max(Q2, reduction_indices=1)
    loss = tf.reduce_mean(tf.square(tf.sub(y,t)))
    train = tf.train.RMSPropOptimizer(agent.learning_rate).minimize(loss)

    sess.run(tf.initialize_all_variables())

    feed = {}
    global_step, exp_pointer = 0,0

    xs = np.empty([experience_size, in_size])
    acts = np.empty([experience_size, out_size])
    rwds = np.empty([experience_size])
    next_xs = np.empty([experience_size, in_size])

    score_queue = []


    x = env.reset()
    score, loss_value, episode_num = 0, 0, 0
    while True:
        #x = agent.prepro(x)
        xs[exp_pointer] = x

        action = agent.get_action(Q1, {obs : np.reshape(x, (1,-1))})
        acts[exp_pointer] = action

        x, reward, done, _ = env.step(np.argmax(action))

        score += reward
        if done and score < 100:
            reward = -300

        rwds[exp_pointer] = score
        next_xs[exp_pointer] = x


        exp_pointer += 1
        if exp_pointer >= experience_size:
            exp_pointer = 0

        global_step += 1
        if global_step > experience_size and not done:
            rand_indexs = np.random.choice(experience_size, agent.exp_batch_size)
            feed.update({obs: xs[rand_indexs]})
            feed.update({act: acts[rand_indexs]})
            feed.update({rwd: rwds[rand_indexs]})
            feed.update({next_obs: next_xs[rand_indexs]})

            if agent.learning:
                step_loss,_ = sess.run([loss, train], feed_dict = feed)
            else:
                step_loss = sess.run(loss, feed_dict = feed)

            loss_value += step_loss

        if global_step % 400 == 0 and global_step >= experience_size:
            agent.epsilon *= agent.eps_decay
            if agent.epsilon < 1e-4:
                agent.epsilon = 1e-4

        if done:
            print ("pisode : {}, score : {}, avg_loss = {}".format(episode_num, score, loss_value/score))
            print ("Epsilon : {}".format(agent.epsilon))
            if not learning:
                print ("test!!")
            if episode_num >= 200:
                score_queue.pop(0)
                score_queue.append(score)
            else:
                score_queue.append(score)

            if global_step % 200 == 0:
                agent.W1_.assign(agent.W1)
                agent.b1_.assign(agent.b1)
                agent.W2_.assign(agent.W2)
                agent.b2_.assign(agent.b2)
                agent.W3_.assign(agent.W3)
                agent.b3_.assign(agent.b3)
                agent.W4_.assign(agent.W4)
                agent.b4_.assign(agent.b4)


            if np.mean(score) > 195:
                agent.learning = False
            score, loss_value = 0, 0
            episode_num += 1
            x = env.reset()

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    learning(env)
