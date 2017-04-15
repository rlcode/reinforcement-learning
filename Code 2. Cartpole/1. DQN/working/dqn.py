""" DQN Class
This code is based on:
https://github.com/hunkim/DeepRL-Agents
CF https://github.com/golbin/TensorFlow-Tutorials
https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/dqn.py
Notes
----------
When modifying this code,
write test codes in `tests/test_DQN.py` as well.
"""
import numpy as np
import tensorflow as tf

# Reusable DQN


class DQN:

    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size=16, l_rate=0.01):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            net = self._X

            net = tf.layers.dense(net, h_size, activation=tf.nn.relu)
            net = tf.layers.dense(net, self.output_size)
            self._Qpred = net

        # We need to define the parts of the network needed for learning a
        # policy
        self._Y = tf.placeholder(
            shape=[None, self.output_size], dtype=tf.float32)

        # Loss function
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        # Learning
        self._train = tf.train.AdamOptimizer(
            learning_rate=l_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [-1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={
            self._X: x_stack, self._Y: y_stack})