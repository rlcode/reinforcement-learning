import tensorflow as tf
import numpy as np
import threading
import gym
import os
from scipy.misc import imresize
import environment_a3c_load_weights
from environment_a3c_load_weights import Env
from renderenv_load_weights import EnvRender

total_episodes = 0

def copy_src_to_dst(from_scope, to_scope):
    """Creates a copy variable weights operation
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def discount_reward(rewards, gamma=0.99):
    """Returns discounted rewards
    """
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        if rewards[t] != 0:
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add

    return discounted_r


class A3CNetwork(object):

    def __init__(self, name, input_shape, output_dim, logdir=None):
        """Network structure is defined here
        """
        with tf.variable_scope(name):
            self.states = tf.placeholder(tf.float32, shape=[None, input_shape], name="states")
            self.actions = tf.placeholder(tf.uint8, shape=[None], name="actions")
            self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
            self.advantage = tf.placeholder(tf.float32, shape=[None], name="advantage")

            action_onehot = tf.one_hot(self.actions, output_dim, name="action_onehot")
            net = self.states

            with tf.variable_scope("layer1") :
                net = tf.layers.dense(net,60,name = "dense")
                net = tf.nn.relu(net,name = 'relu')
            
            with tf.variable_scope("layer2") :
                net = tf.layers.dense(net,60,name = "dense")
                net = tf.nn.relu(net,name = 'relu')

            # actor network
            actions = tf.layers.dense(net, output_dim, name="final_fc")
            self.action_prob = tf.nn.softmax(actions, name="action_prob")
            single_action_prob = tf.reduce_sum(self.action_prob * action_onehot, axis=1)

            entropy = - self.action_prob * tf.log(self.action_prob + 1e-7)
            entropy = tf.reduce_sum(entropy, axis=1)

            log_action_prob = tf.log(single_action_prob + 1e-7)
            maximize_objective = log_action_prob * self.advantage + entropy * 0.01
            self.actor_loss = - tf.reduce_mean(maximize_objective)

            # value network
            self.values = tf.squeeze(tf.layers.dense(net, 1, name="values"))
            self.value_loss = tf.losses.mean_squared_error(labels=self.rewards,
                                                           predictions=self.values)

            self.total_loss = self.actor_loss + self.value_loss * .5
            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=.99)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=2e-4)
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.gradients = self.optimizer.compute_gradients(self.total_loss, var_list)
        self.gradients_placeholders = []

        for grad, var in self.gradients:
            self.gradients_placeholders.append((tf.placeholder(var.dtype, shape=var.get_shape()), var))
        self.apply_gradients = self.optimizer.apply_gradients(self.gradients_placeholders)

        if logdir:
            loss_summary = tf.summary.scalar("total_loss", self.total_loss)
            value_summary = tf.summary.histogram("values", self.values)

            self.summary_op = tf.summary.merge([loss_summary, value_summary])
            self.summary_writer = tf.summary.FileWriter(logdir)


class Agent(threading.Thread):

    def __init__(self, session, env, coord, name, global_network, input_shape, output_dim, saver,logdir=None):
        """Agent worker thread
        """
        super(Agent, self).__init__()
        self.local = A3CNetwork(name, input_shape, output_dim, logdir)
        self.global_to_local = copy_src_to_dst("global", name)
        self.global_network = global_network

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.env = env
        self.sess = session
        self.coord = coord
        self.name = name
        self.logdir = logdir
        self.saver = saver


    def play_episode(self, env_render):
        self.sess.run(self.global_to_local)
        global total_episodes
        if total_episodes == 0 :
            self.coord.request_stop()
        states = []
        actions = []
        rewards = []
        
        s = self.env.reset()
        env_render.reset(self.env)

        done = False
        total_reward = 0
        time_step = 0
        global_step = 0
        while not done:
            env_render.render()
            a = self.choose_action(s)
            s2, r, done, next_coords, mod_rewards = self.env.step(a)
            env_render.move(next_coords, mod_rewards)
            total_reward += r

            states.append(s)
            actions.append(a)

            s = s2

            time_step += 1
            global_step += 1

            if time_step >= 40:
                if r == 1:
                    r *= np.power(0.99, (time_step/2))
                elif r == -1:
                    r *= np.power(1.01, (time_step/2))
            
            rewards.append(r)

            if time_step >= 80 or done:
                states, actions, rewards = [], [], []
                time_step = 0
                print("episode no. " + str(total_episodes) + " global episode " + str(global_step) +" total score :" + str(total_reward))
                total_episodes +=1
                break
        

    def run(self):
        gx = environment_a3c_load_weights.goal_x
        gy = environment_a3c_load_weights.goal_y
        Hx = environment_a3c_load_weights.HEIGHT
        Hy = environment_a3c_load_weights.WIDTH
        ob_list = environment_a3c_load_weights.obs_list
        env_render = EnvRender(gx, gy, Hx, Hy, ob_list)
        while not self.coord.should_stop():
            self.play_episode(env_render)
            
    def choose_action(self, states):
        """
        """
        states = np.reshape(states, [-1, self.input_shape])
        feed = {
            self.local.states: states
        }

        action = self.sess.run(self.local.action_prob, feed)
        action = np.squeeze(action)

        return np.argmax(action)

    def train(self, states, actions, rewards):
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        feed = {
            self.local.states: states
        }

        values = self.sess.run(self.local.values, feed)

        rewards = discount_reward(rewards, gamma=0.99)

        advantage = rewards - values

        feed = {
            self.local.states: states,
            self.local.actions: actions,
            self.local.rewards: rewards,
            self.local.advantage: advantage
        }

        gradients = self.sess.run(self.local.gradients, feed)

        feed = []
        for (grad, _), (placeholder, _) in zip(gradients, self.global_network.gradients_placeholders):
            feed.append((placeholder, grad))

        feed = dict(feed)
        self.sess.run(self.global_network.apply_gradients, feed)


def main():
    # try:
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    coord = tf.train.Coordinator()

    save_path = "models/model.ckpt"
    n_threads = 1
    
    input_shape = 23
    output_dim = 4  # {0,1, 2, 3}
    global_network = A3CNetwork(name="global",
                                input_shape=input_shape,
                                output_dim=output_dim)
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
    saver = tf.train.Saver(var_list=var_list)

    thread_list = []
    env_list = []
    
    global total_episodes

    for id in range(n_threads):
        env = Env()

        single_agent = Agent(env=env,
                             session=sess,
                             coord=coord,
                             name="thread_{}".format(id),
                             global_network=global_network,
                             input_shape=input_shape,
                             output_dim=output_dim,
                             saver=saver)
        thread_list.append(single_agent)
        env_list.append(env)

    
    saver = tf.train.Saver()
    saver.restore(sess, 'models/modelmr1.ckpt')
    

    for t in thread_list:
        t.start()

    print("Ctrl + C to close")

    coord.wait_for_stop()
    
          
    if coord.wait_for_stop() :
        print 'stopped'
        
    

if __name__ == '__main__':
    main()