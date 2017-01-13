import numpy as np
import pickle
import gym


render = False
resume = False
hidden = 20
batch_size = 10

def sigmoid(x):
    return 1. / (1. + np.exp(-x)) 


class RL:
    def __init__(self, in_size, out_size):
        if resume: self.model = pickle.load(open('Acrobot.p', 'rb'))
        else:
            self.model = {}
            self.model['W1'] = np.random.randn(hidden, in_size) /  np.sqrt(in_size)
            self.model['W2'] = np.random.randn(hidden) / np.sqrt(hidden)
        self.gamma = 0.99
        self.decay_rate = 0.99
        self.learning_rate = 0.0002
        self.grad_buffer = []
        for i in range(10):    
            self.grad_buffer.append({k: np.zeros_like(v) for k,v in self.model.iteritems()})
        self.rmsprop_cache = {k: np.zeros_like(v) for k,v in self.model.iteritems()}
        
    
    def discounted_rewards(self, r):
        discounted_r = np.zeros_like(r)
        running_add = 1
        for t in reversed(range(0, r.size)):
            running_add = running_add*0.99 + r[t]
            discounted_r[t] = running_add
        
        return discounted_r
    
    
    def policy_forward(self, x):
        h = np.dot(self.model['W1'], x)
        h = sigmoid(h)
        logp = np.dot(self.model['W2'], h)
        p = sigmoid(logp)
        return p, h
    
    
    def policy_backward(self, eph, epdlogp, epx, ep_num):
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.model['W2'])
        eph_dot = eph*(1-eph)
        dW1 = dh*eph_dot
        dW1 = np.dot(dW1.T, epx )
        
        for k in self.model: self.grad_buffer[ep_num%10][k] = {'W1':dW1, 'W2':dW2}[k]
    
    def learning(self):
        tmp = self.grad_buffer[0]
        for i in range(1,10):
            for k,v in self.model.iteritems():
                tmp[k] += self.grad_buffer[i][k]
             
        for k,v in self.model.iteritems():
            g = tmp[k]
            self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1-self.decay_rate)*g**2
            self.model[k] += self.learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
            
            

env = gym.make('Acrobot-v1')
rl = RL(4,1)
observation = env.reset()
prev_x = None
reward_sum, episode_num, step_size = 0,0,0
xs,hs,dlogps,drs = [],[],[],[]
while True:
    if render : env.render()
    
    x = observation
    
    act_prob, h = rl.policy_forward(x)

    action = 2 if np.random.uniform() < act_prob else 0
     
    
    xs.append(x)
    hs.append(h)
    y = 1 if action == 2 else 0
    dlogps.append(y - act_prob)
    
    
    observation, reward, done, info = env.step(action)
    reward_sum += reward 
    

    drs.append(reward)
    
    
    step_size += 1    
    if done:
        episode_num+=1
        print ("Episode : " + str(episode_num) + ", Reward : " + str(reward_sum) + "   " + str(act_prob))
        
        
        
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [],[],[],[]
        
        discounted_epr = rl.discounted_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        
        epdlogp *= discounted_epr
        rl.policy_backward(eph, epdlogp, epx, episode_num)
        
        
        rl.learning()
                    
        if episode_num % 1000 == 0: 
            pickle.dump(rl.model, open('Acrobot.p', 'wb'))
            
        reward_sum = 0
        observation = env.reset()
        step_size = 0