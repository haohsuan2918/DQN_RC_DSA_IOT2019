"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# DQN + MLP with one hidden layer
class MLP1:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.epsilon = e_greedy

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self.nInternalUnits = 64
        self._build_net(self.nInternalUnits)

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        self.sess.run(self.replace_target_op)

        self.cost_his = []

    def _build_net(self, nInternalUnits):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features])  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions])  # for calculating loss

        self.w1_eval = tf.Variable(tf.random_normal([self.n_features, nInternalUnits], mean=0.0, stddev=0.3))
        self.b1_eval = tf.Variable(tf.random_normal([1, nInternalUnits], mean=0.0, stddev=0.3))
        self.l1_eval = tf.nn.tanh(tf.matmul(self.s, self.w1_eval) + self.b1_eval)

        self.w2_eval = tf.Variable(tf.random_normal([nInternalUnits, self.n_actions], mean=0.0, stddev=0.3))
        self.b2_eval = tf.Variable(tf.random_normal([1, self.n_actions], mean=0.0, stddev=0.3))
        self.q_eval = tf.matmul(self.l1_eval, self.w2_eval) + self.b2_eval

        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        self._train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)


        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features])    # input

        self.w1_target = tf.Variable(tf.random_normal([self.n_features, nInternalUnits], mean=0.0, stddev=0.3))
        self.b1_target = tf.Variable(tf.random_normal([1, nInternalUnits], mean=0.0, stddev=0.3))
        self.l1_target = tf.nn.tanh(tf.matmul(self.s_, self.w1_target) + self.b1_target)

        self.w2_target = tf.Variable(tf.random_normal([nInternalUnits, self.n_actions], mean=0.0, stddev=0.3))
        self.b2_target = tf.Variable(tf.random_normal([1, self.n_actions], mean=0.0, stddev=0.3))
        self.q_next = tf.matmul(self.l1_target, self.w2_target) + self.b2_target

        self.replace_target_op = [tf.assign(self.w1_target, self.w1_eval), tf.assign(self.b1_target, self.b1_eval),
                                  tf.assign(self.w2_target, self.w2_eval), tf.assign(self.b2_target, self.b2_eval)]


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):

        batch_memory = self.memory

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        next_q_value = self.gamma * np.max(q_next, axis=1)

        for index in range(len(eval_act_index)):
            q_target[index, eval_act_index[index]] = reward[index] + next_q_value[index]


        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features], self.q_target: q_target})

        self.cost_his.append(self.cost)

        # replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        self.learn_step_counter += 1

# DQN + MLP with two hidden layer
class MLP2:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.epsilon = e_greedy

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self.nInternalUnits = 64
        self._build_net(self.nInternalUnits)

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        self.sess.run(self.replace_target_op)

        self.cost_his = []

    def _build_net(self, nInternalUnits):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features])  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions])  # for calculating loss

        self.w1_eval = tf.Variable(tf.random_normal([self.n_features, nInternalUnits], mean=0.0, stddev=0.3))
        self.b1_eval = tf.Variable(tf.random_normal([1, nInternalUnits], mean=0.0, stddev=0.3))
        self.l1_eval = tf.nn.tanh(tf.matmul(self.s, self.w1_eval)+self.b1_eval)

        self.w2_eval = tf.Variable(tf.random_normal([nInternalUnits, nInternalUnits], mean=0.0, stddev=0.3))
        self.b2_eval = tf.Variable(tf.random_normal([1, nInternalUnits], mean=0.0, stddev=0.3))
        self.l2_eval = tf.nn.tanh(tf.matmul(self.l1_eval, self.w2_eval)+self.b2_eval)

        self.w3_eval = tf.Variable(tf.random_normal([nInternalUnits, self.n_actions], mean=0.0, stddev=0.3))
        self.b3_eval = tf.Variable(tf.random_normal([1, self.n_actions], mean=0.0, stddev=0.3))
        self.q_eval = tf.matmul(self.l2_eval, self.w3_eval) + self.b3_eval

        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        self._train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)


        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features])    # input

        self.w1_target = tf.Variable(tf.random_normal([self.n_features, nInternalUnits], mean=0.0, stddev=0.3))
        self.b1_target = tf.Variable(tf.random_normal([1, nInternalUnits], mean=0.0, stddev=0.3))
        self.l1_target = tf.nn.tanh(tf.matmul(self.s, self.w1_target)+self.b1_target)

        self.w2_target = tf.Variable(tf.random_normal([nInternalUnits, nInternalUnits], mean=0.0, stddev=0.3))
        self.b2_target = tf.Variable(tf.random_normal([1, nInternalUnits], mean=0.0, stddev=0.3))
        self.l2_target = tf.nn.tanh(tf.matmul(self.l1_target, self.w2_target)+self.b2_target)

        self.w3_target = tf.Variable(tf.random_normal([nInternalUnits, self.n_actions], mean=0.0, stddev=0.3))
        self.b3_target = tf.Variable(tf.random_normal([1, self.n_actions], mean=0.0, stddev=0.3))
        self.q_next = tf.matmul(self.l2_target, self.w3_target)+self.b3_target

        self.replace_target_op = [tf.assign(self.w1_target, self.w1_eval), tf.assign(self.b1_target, self.b1_eval),
                                  tf.assign(self.w2_target, self.w2_eval), tf.assign(self.b2_target, self.b2_eval),
                                  tf.assign(self.w3_target, self.w3_eval), tf.assign(self.b3_target, self.b3_eval)]


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):

        batch_memory = self.memory

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        next_q_value = self.gamma * np.max(q_next, axis=1)

        for index in range(len(eval_act_index)):
            q_target[index, eval_act_index[index]] = reward[index] + next_q_value[index]


        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
            feed_dict={self.s: batch_memory[:, :self.n_features], self.q_target: q_target})

        self.cost_his.append(self.cost)

        # replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        self.learn_step_counter += 1



