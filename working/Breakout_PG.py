import gym
import random
import tensorflow as tf
import numpy as np

DIM = 105*80*2
gamma = 0.99
batch_size = 10

def weight_variable(shape):
    #bound = 1 / np.sqrt(np.sum(shape))
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)
    #return tf.Variable(tf.random_uniform(shape, minval=0, maxval=bound))

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def prepro(state):
    # set images to gray scale
    gray_state = np.zeros([210,160])
    
    for i in range(210):
        for j in range(160):
            gray_state[i][j] = np.mean(state[i][j])
    
    # get rid of noises
    gray_state[gray_state == 142] = 0
    gray_state = gray_state[::2,::2]
    return gray_state

def discountRewards(rewards):
    discounted_r = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r

class AGENT():
    def __init__(self, learning_rate = 1e-4):
        self.learning_rate = learning_rate
         
        self.conv_W1 = weight_variable([5,5,2,32])
        self.conv_b1 = bias_variable([32])
        self.conv_W2 = weight_variable([4,4,32,64])
        self.conv_b2 = bias_variable([64])
        self.conv_W3 = weight_variable([3,3,64,64])
        self.conv_b3 = bias_variable([64])
        
        self.fc_W1 = weight_variable([13*10*64, 512])
        self.fc_b1 = bias_variable([512])
        self.fc_W2 = weight_variable([512,2])
        self.fc_b2 = bias_variable([2])
        
        self.v_conv_W1 = weight_variable([5,5,2,16])
        self.v_conv_b1 = bias_variable([16])
        self.v_conv_W2 = weight_variable([4,4,16,32])
        self.v_conv_b2 = bias_variable([32])
        self.v_conv_W3 = weight_variable([3,3,32,32])
        self.v_conv_b3 = bias_variable([32])
        
        self.v_fc_W1 = weight_variable([13*10*32, 512])
        self.v_fc_b1 = bias_variable([512])
        self.v_fc_W2 = weight_variable([512,1])
        self.v_fc_b2 = bias_variable([1])
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())
        
        
        self.state, self.prob, self.conv_drop, self.fc_drop = self.getPolicy()
        self.act, self.adv, self.train = self.policyOptimizer()
        self.v_state, self.value = self.getValue()
        self.v_n_state, self.n_value = self.getValue()
        self.rwd, self.v_train = self.valueOptimizer()
        
        
    def getAction(self, get_action_state):
        action = 4 if random.random() < .5 else 5
        return action   
        
        
    def policyOptimizer(self):
        act = tf.placeholder(tf.float32, [None, 2])
        adv = tf.placeholder(tf.float32, [None, 1])
        
        good_probabilities = tf.reduce_sum(tf.mul(self.prob, act),
                                      reduction_indices = [1])
        log_probabilities = tf.log(tf.clip_by_value(good_probabilities, 1e-10, 1e+8)) * adv
        loss = -tf.reduce_sum(log_probabilities)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)
        
        return act, adv, optimizer
    
    def valueOptimizer(self):
        rwd = tf.placeholder(tf.float32, [None, 1])
        
        value1 = self.value
        value2 = rwd + self.n_value*gamma
        v_loss = tf.reduce_mean(tf.square(value2 - value1), reduction_indices=[1])
        v_optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(v_loss)
        
        return rwd, v_optimizer
        
    def getPolicy(self):
        state = tf.placeholder(tf.float32, [None, 105, 80, 2]) 
        conv_drop = tf.placeholder(tf.float32)
        fc_drop = tf.placeholder(tf.float32)
        state_image = tf.reshape(state, [-1,105,80,2])
        
        conv_h1_out = tf.nn.conv2d(state_image, self.conv_W1, strides = [1,2,2,1], padding = "SAME")
        conv_h1 = tf.nn.relu(conv_h1_out + self.conv_b1)
        conv_h1_drop = tf.nn.dropout(conv_h1, conv_drop)
        conv_h2_out = tf.nn.conv2d(conv_h1_drop, self.conv_W2, strides = [1,2,2,1], padding = "SAME")
        conv_h2 = tf.nn.relu(conv_h2_out + self.conv_b2)
        conv_h2_drop = tf.nn.dropout(conv_h2, conv_drop)
        conv_h3_out = tf.nn.conv2d(conv_h2_drop, self.conv_W3, strides = [1,2,2,1], padding = "SAME")
        conv_h3 = tf.nn.relu(conv_h3_out + self.conv_b3)
        
        conv_h3_flat = tf.reshape(conv_h3, [-1, 13*10*64])
        
        fc_h1 = tf.nn.relu(tf.matmul(conv_h3_flat, self.fc_W1) + self.fc_b1)
        fc_h1_drop = tf.nn.dropout(fc_h1, fc_drop)
        prob = tf.nn.softmax(tf.matmul(fc_h1_drop, self.fc_W2) + self.fc_b2)
        
        return state, prob, conv_drop, fc_drop
    
    def getValue(self):
        v_state = tf.placeholder(tf.float32, [None, 105, 80, 2])
        v_state_image = tf.reshape(v_state, [-1,105,80,2])
        
        v_conv_h1_out = tf.nn.conv2d(v_state_image, self.v_conv_W1, strides=[1,2,2,1], padding="SAME")
        v_conv_h1 = tf.nn.relu(v_conv_h1_out+self.v_conv_b1)
        v_conv_h1_drop = tf.nn.dropout(v_conv_h1, self.conv_drop)
        v_conv_h2_out = tf.nn.conv2d(v_conv_h1_drop, self.v_conv_W2, strides=[1,2,2,1], padding="SAME")
        v_conv_h2 = tf.nn.relu(v_conv_h2_out+self.v_conv_b2)
        v_conv_h2_drop = tf.nn.dropout(v_conv_h2, self.conv_drop)
        v_conv_h3_out = tf.nn.conv2d(v_conv_h2_drop, self.v_conv_W3, strides=[1,2,2,1], padding="SAME")
        v_conv_h3 = tf.nn.relu(v_conv_h3_out+self.v_conv_b3)
        
        v_conv_h3_flat = tf.reshape(v_conv_h3, [-1, 13*10*32])
        
        v_fc_h1 = tf.nn.relu(tf.matmul(v_conv_h3_flat, self.v_fc_W1) + self.v_fc_b1)
        v_fc_h1_drop = tf.nn.dropout(v_fc_h1, self.fc_drop)
        value = tf.matmul(v_fc_h1_drop, self.v_fc_W2) + self.v_fc_b2
        
        return v_state, value
        
    
    def getActionProb(self, get_action_prob_state):
        action_prob = self.sess.run(self.prob, feed_dict = {self.state: get_action_prob_state, self.conv_drop: 1.0, self.fc_drop: 1.0})
        return action_prob              
                         
                         
env = gym.make("Breakout-v0")
agent = AGENT()
obs = env.reset()
prev_x = None
states, rewards, actions, next_states = [],[],[],[]
running_reward = None
reward_sum, step = 0, 0
episode_number = 0

while True:
    env.render()
    cur_x = obs
    x = [cur_x, prev_x] if prev_x is not None else np.zeros(DIM)
    x = np.reshape(x, [-1 ,105, 80, 2])
    states.append(x)
    prev_x = cur_x
    if step != 0:
        next_states.append(x) 
    
    
    action = agent.getAction(x)
    actions.append([0,1] if action == 5 else [1,0])
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    reward_sum += reward
    step += 1

    
    if done:
        next_states.append(np.reshape(np.zeros(DIM), [-1, 105, 80, 2]))
        episode_number += 1
        
        discounted_epi_reward = discountRewards(rewards)
        
        #discounted_epi_reward -= np.mean(discounted_epi_reward)
        #discounted_epi_reward /= np.std(discounted_epi_reward) 
        
        epi_state = np.vstack(states)
        epi_reward = np.vstack(discounted_epi_reward)
        epi_action = np.vstack(actions)
        epi_n_state = np.vstack(next_states)
        
        states,rewards,actions,next_states = [],[],[],[]
        
        if episode_number % batch_size == 0:
            agent.sess.run(agent.v_train, feed_dict={agent.v_state: epi_state, agent.rwd: epi_reward,
                                                     agent.v_n_state: epi_n_state, agent.conv_drop: 0.4, agent.fc_drop: 0.5})
            
            epi_advantage = epi_reward - agent.sess.run(agent.value, feed_dict={agent.v_state: epi_state})
            
            agent.sess.run(agent.train, feed_dict={agent.state: epi_state, agent.act: epi_action, 
                                                   agent.adv: epi_reward, agent.conv_drop: 0.4, agent.fc_drop: 0.5}) 
        reward_sum, step = 0,0
        obs = env.reset() # reset env
        prev_x = None
