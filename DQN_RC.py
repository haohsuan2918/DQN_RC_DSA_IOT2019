import numpy as np
from pyESN_online import ESN
import copy

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=10,
            memory_size=300,
            lr = 0.01
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = memory_size
        self.epsilon = e_greedy

        # total learning step
        self.learn_step_counter = 0
        
        # initialize learning rate
        self.lr = lr 

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features*2+2))

        # build net
        self._build_net()

        self.cost_his = []

    def _build_net(self):
        # ------------------ ESN parameters ------------------
        nInternalUnits = 64
        spectralRadius = 0.80
        inputScaling = 2 * np.ones(self.n_features)
        inputShift = -1 * np.ones(self.n_features)
        teacherScaling = 1 * np.ones(self.n_actions)
        teacherShift = 0 * np.ones(self.n_actions)
        self.nForgetPoints = 50

        # ------------------ build evaluate_net ------------------
        self.eval_net = ESN(n_inputs=self.n_features, n_outputs=self.n_actions, n_reservoir=nInternalUnits,
                  spectral_radius=spectralRadius, sparsity=1 - min(0.2 * nInternalUnits, 1), noise=0, lr=self.lr,
                  input_shift=inputShift, input_scaling=inputScaling,
                  teacher_scaling=teacherScaling, teacher_shift=teacherShift)

        # ------------------ build target_net ------------------
        self.target_net = ESN(n_inputs=self.n_features, n_outputs=self.n_actions, n_reservoir=nInternalUnits,
                  spectral_radius=spectralRadius, sparsity=1 - min(0.2 * nInternalUnits, 1), noise=0, lr=self.lr,
                  input_shift=inputShift, input_scaling=inputScaling,
                  teacher_scaling=teacherScaling, teacher_shift=teacherShift)

        self.target_net = copy.deepcopy(self.eval_net)

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
            actions_value = self.eval_net.predict(observation, 0, continuation=True)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):

        batch_memory = self.memory

        eval_net_input = batch_memory[:, :self.n_features]
        target_net_input = batch_memory[:, -self.n_features:]

        q_eval = self.eval_net.predict(eval_net_input, 0, continuation=False)
        q_next = self.target_net.predict(target_net_input, 0, continuation=False)

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]



        next_q_value = self.gamma * np.max(q_next, axis=1)
        for index in range(len(eval_act_index)):
            q_target[index, eval_act_index[index]] = reward[index] + next_q_value[index]


        # train eval network
        pred_train = self.eval_net.fit(eval_net_input, q_target, self.nForgetPoints)
        self.cost = np.linalg.norm(pred_train-q_target)
        self.cost_his.append(self.cost)

        # prepare the same reservoir state for next training when calculating q_eval, q_next
        self.eval_net.startstate = copy.deepcopy(self.eval_net.laststate)
        self.eval_net.startinput = copy.deepcopy(self.eval_net.lastinput)
        self.eval_net.startoutput = copy.deepcopy(self.eval_net.lastoutput)

        self.target_net.startstate = copy.deepcopy(self.target_net.laststate)
        self.target_net.startinput = copy.deepcopy(self.target_net.lastinput)
        self.target_net.startoutput = copy.deepcopy(self.target_net.lastoutput)


        # replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net = copy.deepcopy(self.eval_net)
            #print('\ntarget_params_replaced\n')

        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
