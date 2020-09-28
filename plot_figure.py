import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":

    n_channel = 6
    n_su = 2
    file_folder = ".\\result\\channel_%d_su_%d_punish_-2" % (n_channel, n_su)

    if n_channel == 22:
        batch_size = 4000
        n_curve = 3
    elif n_channel == 6:
        batch_size = 2000
        n_curve = 5

    success_hist = []
    fail_PU_hist = []
    fail_collision_hist = []
    overall_reward = []

    label_list = ['DQN + RC', 'Myopic', 'QLearning', 'DQN + MLP1', 'DQN + MLP2']
    marker_list = ['r-s', 'b-^', 'g-o', 'c-d', 'm-x']

    for n in range(n_curve):
        success_hist.append(np.load(file_folder + '\\success_hist_%d.npy' % (n+1)))
        fail_PU_hist.append(np.load(file_folder + '\\fail_PU_hist_%d.npy' % (n+1)))
        fail_collision_hist.append(np.load(file_folder + '\\fail_collision_hist_%d.npy' % (n+1)))
        overall_reward.append(np.load(file_folder + '\\overall_reward_%d.npy' % (n+1)))

    # Calculate the running mean
    if n_channel == 6:
        mean_average_time = 4
    elif n_channel == 22:
        mean_average_time = 3

    record_number = int(success_hist[0].size / mean_average_time)

    success_hist_mean = []
    fail_PU_hist_mean = []
    fail_collision_hist_mean = []
    overall_reward_mean = []
    for n in range(n_curve):
        success_hist_mean.append(np.zeros(record_number))
        fail_PU_hist_mean.append(np.zeros(record_number))
        fail_collision_hist_mean.append(np.zeros(record_number))
        overall_reward_mean.append(np.zeros(record_number))
        for k in range(record_number):
            index = np.arange(k * mean_average_time, (k + 1) * mean_average_time)
            success_hist_mean[n][k] = np.mean(success_hist[n][index])
            fail_PU_hist_mean[n][k] = np.mean(fail_PU_hist[n][index])
            fail_collision_hist_mean[n][k] = np.mean(fail_collision_hist[n][index])
            overall_reward_mean[n][k] = np.mean(overall_reward[n][index])

    total_record_number = (np.arange(record_number) + 1) * batch_size * mean_average_time

    plt.figure()
    for n in range(n_curve):
        plt.plot(total_record_number, success_hist_mean[n]/batch_size, marker_list[n], label=label_list[n])
    plt.legend(loc='lower right')
    if n_channel == 6:
        plt.ylim(0.3, 0.9)
    elif n_channel == 22:
        plt.ylim(0.7, 1)
    plt.ylabel('Average success rate')
    plt.xlabel('Training steps')

    plt.figure()
    for n in range(n_curve):
        plt.plot(total_record_number, fail_PU_hist_mean[n]/batch_size, marker_list[n], label=label_list[n])
    plt.legend(loc='upper right')
    if n_channel == 6:
        plt.ylim(0.1, 0.3)
    elif n_channel == 22:
        plt.ylim(0, 0.3)
    plt.ylabel('Average collision (with PU) rate')
    plt.xlabel('Training steps')

    plt.figure()
    for n in range(n_curve):
        plt.plot(total_record_number, fail_collision_hist_mean[n] / batch_size, marker_list[n], label=label_list[n])
    plt.legend(loc='upper right')
    if n_channel == 6:
        plt.ylim(0, 0.5)
    plt.ylabel('Average collision (with SU) rate')
    plt.xlabel('Training steps')


    plt.figure()
    for n in range(n_curve):
        plt.plot(total_record_number, overall_reward_mean[n], marker_list[n], label=label_list[n])
    plt.legend(loc='lower right')
    if n_channel == 6:
        plt.ylim(2, 5)
        plt.ylim(2, 6)
    elif n_channel == 22:
        plt.ylim(3, 6)
    plt.ylabel('Average reward')
    plt.xlabel('Training steps')

    plt.show()

    PU_TX_x = np.load(file_folder + '\PU_TX_x.npy')
    PU_TX_y = np.load(file_folder + '\PU_TX_y.npy')
    PU_RX_x = np.load(file_folder + '\PU_RX_x.npy')
    PU_RX_y = np.load(file_folder + '\PU_RX_y.npy')

    SU_TX_x = np.load(file_folder + '\SU_TX_x.npy')
    SU_TX_y = np.load(file_folder + '\SU_TX_y.npy')
    SU_RX_x = np.load(file_folder + '\SU_RX_x.npy')
    SU_RX_y = np.load(file_folder + '\SU_RX_y.npy')

    # Plot the locations
    plt.figure()
    plt.plot(PU_TX_x, PU_TX_y, 'ro', label='PU_TX')
    plt.plot(PU_RX_x, PU_RX_y, 'rx', label='PU_RX')
    plt.plot(SU_TX_x, SU_TX_y, 'bs', label='SU_TX')
    plt.plot(SU_RX_x, SU_RX_y, 'b^', label='SU_RX')
    plt.legend(loc='lower right')
    plt.ylabel('y')
    plt.xlabel('x')