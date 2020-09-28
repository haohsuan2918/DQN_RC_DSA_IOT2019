import numpy as np
import matplotlib.pyplot as plt
import copy

class DSA_Markov():
    def __init__(
            self,
            n_channel,
            n_su,
            sense_error_prob_max = 0.2,
            punish_interfer_PU = -2
    ):
        self.n_channel = n_channel # The number of the channels
        self.n_su = n_su # The number of the SUs

        # Initialize the Markov channels
        self._build_Markov_channel()

        # Initialize the locations of SUs and PUs
        self._build_location()

        # Set the noise (mW)
        self.Noise = 1 * np.float_power(10, -8)
        # Set the carrier frequency (5 GHz)
        self.fc = 5
        # Set the K in channel gain
        self.K = 8
        # Set the power of PU and SU (mW)
        self.SU_power = 20
        self.PU_power = 40

        # Initialize SINR (no consider interference of SUs)
        self.render_SINR()

        self.n_actions = n_channel + 1 # The action space size
        self.n_features = n_channel # The sensing result space

        self.sense_error_prob_max = sense_error_prob_max
        self.sense_error_prob = np.random.uniform(0, self.sense_error_prob_max, size=(self.n_su, self.n_channel))

        # The punishment for interfering PUs
        self.punish_interfer_PU = punish_interfer_PU

    def _build_Markov_channel(self):
        # Initialize channel state (uniform distribution)
        # 1: Inactive state (primary user is not using)
        # 0: Active state (primary user is using)

        self.channel_state = np.random.choice(2, self.n_channel)

        # Initialize the transition probability of independent channels
        self.stayGood_prob = np.random.uniform(0.7, 1, self.n_channel)
        self.stayBad_prob = np.random.uniform(0, 0.3, self.n_channel)
        self.goodToBad_prob = 1 - self.stayGood_prob
        self.badToGood_prob = 1 - self.stayBad_prob

    def _build_location(self):

        # Initialize the location of PUs
        self.PU_TX_x = np.random.uniform(0, 150, self.n_channel)
        self.PU_TX_y = np.random.uniform(0, 150, self.n_channel)
        self.PU_RX_x = np.random.uniform(0, 150, self.n_channel)
        self.PU_RX_y = np.random.uniform(0, 150, self.n_channel)

        # Initialize the location of SUs transmitters
        self.SU_TX_x = np.random.uniform(0+40, 150-40, self.n_su)
        self.SU_TX_y = np.random.uniform(0+40, 150-40, self.n_su)

        # Initialize the distance between SUs' transmitter and receiver
        self.SU_d = np.random.uniform(20, 40, self.n_su)

        # Initialize the location of SUs receivers
        SU_theda = 2 * np.pi * np.random.uniform(0, 1, self.n_su)
        SU_dx = self.SU_d * np.cos(SU_theda)
        SU_dy = self.SU_d * np.sin(SU_theda)
        self.SU_RX_x = self.SU_TX_x + SU_dx
        self.SU_RX_y = self.SU_TX_y + SU_dy

        # Compute the distance between PU_TX and SU_RX
        self.SU_RX_PU_TX_d = np.zeros((self.n_su, self.n_channel))
        for k in range(self.n_su):
            for l in range(self.n_channel):
                self.SU_RX_PU_TX_d[k][l] = np.sqrt(
                    np.float_power(self.SU_RX_x[k] - self.PU_TX_x[l], 2) + np.float_power(
                        self.SU_RX_y[k] - self.PU_TX_y[l], 2))

        # Compute the distance between PU_TX and SU_RX
        self.SU_RX_SU_TX_d = np.zeros((self.n_su, self.n_su))
        for k1 in range(self.n_su):
            for k2 in range(self.n_su):
                self.SU_RX_SU_TX_d[k1][k2] = np.sqrt(
                    np.float_power(self.SU_RX_x[k1] - self.SU_TX_x[k2], 2) + np.float_power(
                        self.SU_RX_y[k1] - self.SU_TX_y[k2], 2))

        # Plot the locations
        plt.plot(self.PU_TX_x, self.PU_TX_y, 'ro', label='PU_TX')
        plt.plot(self.PU_RX_x, self.PU_RX_y, 'rx', label='PU_RX')
        plt.plot(self.SU_TX_x, self.SU_TX_y, 'bs', label='SU_TX')
        plt.plot(self.SU_RX_x, self.SU_RX_y, 'b^', label='SU_RX')
        plt.legend(loc='lower right')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.show()

    def store_action(self, action):
        self.action = action

    def sense(self):
        tmp_dice = np.random.uniform(0, 1, size=(self.n_su, self.n_channel))  # roll the dice between 0 and 1
        error_index = tmp_dice < self.sense_error_prob # True: sensing error happens, False: sensing is correct

        # Get the sensing result
        self.sensing_result = self.channel_state*(1-error_index) + (1-self.channel_state)*(error_index)

        return self.sensing_result

    def access(self, action):
        # action = [0, n_channel-1]: access the selected channel
        # action = n_channel: do not access the channel
        self.success = 0
        self.fail_PU = 0
        self.fail_collision = 0

        self.reward = np.zeros(self.n_su)

        # Calculate the interference of SUs
        Interferecne_SU = 0
        SU_sigma2 = np.float_power(10, -((41 + 22.7 * np.log10(self.SU_RX_SU_TX_d) + 20 * np.log10(self.fc / 5)) / 10))
        for k in range(self.n_su):
            SU_sigma2[k][k] = 0

        for k in range(self.n_su):
            if (action[k] == self.n_channel): # action is not choosing any channel
                self.reward[k] = 0
            else: # action is choosing one of channels
                for q in range(self.n_su):
                    if (action[q] == action[k]):
                        Interferecne_SU = Interferecne_SU + SU_sigma2[k][q]*self.SU_power
                SINR = self.H2[k, action[k]] * self.SU_power / (Interferecne_SU + self.Interferecne_PU[k, action[k]] + self.Noise)
                self.reward[k] = np.log2(1 + SINR)


                if (self.channel_state[action[k]] == 1):
                    if (len(np.where(action == action[k])[0]) == 1):
                        # successful transmission
                        self.success = self.success + 1
                    else:
                        # collision with SU
                        self.fail_collision = self.fail_collision + 1
                else:
                    # collision with PU
                    self.fail_PU = self.fail_PU + 1
                    self.reward[k] = self.punish_interfer_PU
                    if (len(np.where(action == action[k])[0]) > 1):
                        # collision with SU
                        self.fail_collision = self.fail_collision + 1

        return self.reward

    def render(self):
        # The probability of staying in current state in next time slot
        stay_prob = self.channel_state*self.stayGood_prob + (1-self.channel_state)*self.stayBad_prob
        tmp_dice = np.random.uniform(0, 1, self.n_channel) # roll the dice between 0 and 1
        stay_index = tmp_dice < stay_prob # 1: stay in current state, 0: change state

        # Update the channel state
        self.channel_state = self.channel_state*stay_index + (1-self.channel_state)*(1-stay_index)

    def render_SINR(self):
        # Update the SINR

        # Calculate the channel gain
        SU_d = copy.deepcopy(np.reshape(self.SU_d, (-1, 1)))
        for n in range(self.n_channel-1):
            SU_d = np.hstack( (SU_d, np.reshape(self.SU_d, (-1, 1))) )

        SU_sigma2 = np.float_power(10, -((41+22.7*np.log10(SU_d)+20*np.log10(self.fc/5))/10))
        CN_real = np.random.normal(0, 1, size=(self.n_su, self.n_channel))
        CN_imag = np.random.normal(0, 1, size=(self.n_su, self.n_channel))
        theda = np.random.uniform(0, 1, size=(self.n_su, self.n_channel))
        H = np.sqrt(self.K/(self.K+1)*SU_sigma2)*np.exp(1j*2*np.pi*theda) + np.sqrt(1/(self.K+1)*SU_sigma2/2)*(CN_real + 1j*CN_imag)
        self.H2 = np.float_power(np.absolute(H), 2)

        # Calculate the interference of PUs
        PU_sigma2 = np.float_power(10, -((41 + 22.7 * np.log10(self.SU_RX_PU_TX_d) + 20 * np.log10(self.fc / 5)) / 10))
        channel_state = np.array([self.channel_state for k in range(self.n_su)])
        self.Interferecne_PU = self.PU_power * PU_sigma2 * (1 - channel_state)

        self.SINR = self.H2*self.SU_power/(self.Interferecne_PU + self.Noise)


class DSA_Period():
    def __init__(
            self,
            n_channel,
            n_su,
            sense_error_prob_max=0.0,
            punish_interfer_PU=-4,
            punish_idle = 0
    ):
        self.n_channel = n_channel
        self.n_su = n_su
        self._build_periodic_channel()
        self._build_location()

        # Set the noise (6*10^(-8) mW)
        self.Noise = 1 * np.float_power(10, -8)
        # Set the carrier frequency (5 GHz)
        self.fc = 5
        # Set the K in channel gain
        self.K = 8
        # Set the power
        self.SU_power = 20
        self.PU_power = 40
        # Initialize SINR (no consider interference of SUs)
        self.render_SINR()

        self.n_actions = n_channel+1 # select at most one channel
        self.n_features = n_channel

        self.sense_error_prob_max = sense_error_prob_max
        self.sense_error_prob = np.random.uniform(0, self.sense_error_prob_max, size=(self.n_su, self.n_channel))

        self.punish_interfer_PU = punish_interfer_PU
        self.punish_idle = punish_idle

        self.channel_state_his = np.zeros(self.n_channel)

    def _build_periodic_channel(self):

        self.channel_state = np.ones(self.n_channel)

        # Initialize period
        self.period = np.ones(self.n_channel, dtype=int) * 3

        self.count = np.ones(self.n_channel)

    def _build_location(self):

        # Initialize the location of PUs
        self.PU_TX_x = np.random.uniform(0, 150, self.n_channel)
        self.PU_TX_y = np.random.uniform(0, 150, self.n_channel)
        self.PU_RX_x = np.random.uniform(0, 150, self.n_channel)
        self.PU_RX_y = np.random.uniform(0, 150, self.n_channel)

        # Initialize the location of SUs transmitters
        self.SU_TX_x = np.random.uniform(0+40, 150-40, self.n_su)
        self.SU_TX_y = np.random.uniform(0+40, 150-40, self.n_su)

        # Initialize the location of SUs receivers
        self.SU_d = np.array([30])
        # self.SU_d = np.random.uniform(20, 40, self.n_su)

        SU_theda = 2 * np.pi * np.random.uniform(0, 1, self.n_su)
        SU_dx = self.SU_d * np.cos(SU_theda)
        SU_dy = self.SU_d * np.sin(SU_theda)
        self.SU_RX_x = self.SU_TX_x + SU_dx
        self.SU_RX_y = self.SU_TX_y + SU_dy

        # Compute the distance between PU_TX and SU_RX
        self.SU_RX_PU_TX_d = np.zeros((self.n_su, self.n_channel))
        for k in range(self.n_su):
            for l in range(self.n_channel):
                self.SU_RX_PU_TX_d[k][l] = np.sqrt(np.float_power(self.SU_RX_x[k]-self.PU_TX_x[l], 2) + np.float_power(self.SU_RX_y[k] - self.PU_TX_y[l],2))

        # Compute the distance between PU_TX and SU_RX
        self.SU_RX_SU_TX_d = np.zeros((self.n_su, self.n_su))
        for k1 in range(self.n_su):
            for k2 in range(self.n_su):
                self.SU_RX_SU_TX_d[k1][k2] = np.sqrt(np.float_power(self.SU_RX_x[k1]-self.SU_TX_x[k2],2) + np.float_power(self.SU_RX_y[k1]-self.SU_TX_y[k2],2) )

        # Plot the locations
        plt.plot(self.PU_TX_x, self.PU_TX_y, 'ro', label='PU_TX')
        plt.plot(self.PU_RX_x, self.PU_RX_y, 'rx', label='PU_RX')
        plt.plot(self.SU_TX_x, self.SU_TX_y, 'bo', label='SU_TX')
        plt.plot(self.SU_RX_x, self.SU_RX_y, 'bx', label='SU_RX')
        plt.legend(loc='lower right')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.show()

    def store_action(self, action):
        self.action = action

    def sense(self):
        tmp_dice = np.random.uniform(0, 1, size=(self.n_su, self.n_channel))  # roll the dice between 0 and 1
        error_index = tmp_dice < self.sense_error_prob # True: sensing error happens, False: sensing is correct

        # Get the sensing result
        self.sensing_result = self.channel_state*(1-error_index) + (1-self.channel_state)*(error_index)

        return self.sensing_result

    def access(self, action):
        # action = [0, n_channel-1]: access the selected channel
        # action = n_channel: do not access the channel
        self.success = 0
        self.fail_PU = 0
        self.fail_collision = 0

        self.ACK = np.zeros((self.n_su, self.n_channel))
        self.reward = np.zeros(self.n_su)

        # Calculate the interference of PUs
        Interferecne_SU = 0
        SU_sigma2 = np.float_power(10, -((41 + 22.7 * np.log10(self.SU_RX_SU_TX_d) + 20 * np.log10(self.fc / 5)) / 10))
        for k in range(self.n_su):
            SU_sigma2[k][k] = 0

        for k in range(self.n_su):
            if (action[k] == self.n_channel):
                self.reward[k] = self.punish_idle
            else:
                for q in range(self.n_su):
                    if (action[q] == action[k]):
                        Interferecne_SU = Interferecne_SU + SU_sigma2[k][q]*self.SU_power
                SINR = self.H2[k, action[k]] * self.SU_power / (Interferecne_SU + self.Interferecne_PU[k, action[k]] + self.Noise)
                self.reward[k] = np.log2(1 + SINR)

                if (self.channel_state[action[k]] == 1):
                    if (len(np.where(action == action[k])[0]) == 1):
                        self.success = self.success + 1
                    else:
                        self.fail_collision = self.fail_collision + 1
                else:
                    self.fail_PU = self.fail_PU + 1
                    self.reward[k] = self.punish_interfer_PU
                    if (len(np.where(action == action[k])[0]) > 1):
                        self.fail_collision = self.fail_collision + 1

        return self.reward


    def render(self):

        self.count = self.count + 1
        for k in range(self.n_channel):
            self.channel_state[k] = 1
            if (self.count[k] % self.period[k] == 0):
                self.channel_state[k] = 0


    def render_SINR(self):

        # Calculate the channel gain
        SU_d = copy.deepcopy(np.reshape(self.SU_d, (-1, 1)))
        for n in range(self.n_channel-1):
            SU_d = np.hstack( (SU_d, np.reshape(self.SU_d, (-1, 1))) )

        SU_sigma2 = np.float_power(10, -((41+22.7*np.log10(SU_d)+20*np.log10(self.fc/5))/10))
        CN_real = np.random.normal(0, 1, size=(self.n_su, self.n_channel))
        CN_imag = np.random.normal(0, 1, size=(self.n_su, self.n_channel))
        theda = np.random.uniform(0, 1, size=(self.n_su, self.n_channel))
        H = np.sqrt(self.K/(self.K+1)*SU_sigma2)*np.exp(1j*2*np.pi*theda) + np.sqrt(1/(self.K+1)*SU_sigma2/2)*(CN_real + 1j*CN_imag)
        self.H2 = np.float_power(np.absolute(H), 2)

        # Calculate the interference of PUs
        PU_sigma2 = np.float_power(10, -((41 + 22.7 * np.log10(self.SU_RX_PU_TX_d) + 20 * np.log10(self.fc / 5)) / 10))
        channel_state = np.array([self.channel_state for k in range(self.n_su)])
        self.Interferecne_PU = self.PU_power * PU_sigma2 * (1 - channel_state)

        self.SINR = self.H2*self.SU_power/(self.Interferecne_PU + self.Noise)