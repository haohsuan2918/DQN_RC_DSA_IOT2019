from DSA_env import DSA_Period
from DQN_RC import DeepQNetwork
from DQN_MLP import MLP1
import numpy as np
import copy



if __name__ == "__main__":

    random_seed = 10
    np.random.seed(random_seed)

    # Initialize the environment
    n_channel = 1
    n_su = 1
    env = DSA_Period(n_channel, n_su)
    env_copy = copy.deepcopy(env)

    # training parameters
    batch_size = 3000
    replace_target_iter = 1
    total_episode = batch_size * replace_target_iter * 100
    epsilon_update_period = batch_size * replace_target_iter * 20
    e_greedy = [0.6, 0.9, 1.0]
    coherence_time = 1
    learning_rate = 0.01


    '''
    Initialize the DQN_RC
    '''
    DQN_RC_list = []
    epsilon_index = np.zeros(n_su, dtype=int)
    for k in range(n_su):
        DQN_tmp = DeepQNetwork(env.n_actions, env.n_features,
                          reward_decay=0.9,
                          e_greedy=e_greedy[0],
                          replace_target_iter=replace_target_iter,
                          memory_size=batch_size,
                          lr=learning_rate
                          )
        DQN_RC_list.append(copy.deepcopy(DQN_tmp))

    '''
    SUs sense the environment and get the sensing result (contains sensing errors)
    '''
    observation = env.sense()

    # Initialize some record values
    reward_sum = np.zeros(n_su)
    overall_reward_1 = []
    success_hist_1 = []
    fail_PU_hist_1 = []
    fail_collision_hist_1 = []
    success_sum = 0
    fail_PU_sum = 0
    fail_collision_sum = 0

    action = np.zeros(n_su).astype(np.int32)
    for step in range(total_episode):

        # SU choose action based on observation
        for k in range(n_su):
            action[k] = DQN_RC_list[k].choose_action(observation[k,:])

        # update the environment
        env.render()
        if ((step+1)% coherence_time == 0):
            env.render_SINR()

        # SU take action and get the reward
        reward = env.access(action)

        # Record reward, the number of success / interference / collision
        reward_sum = reward_sum + reward
        #reward_batch_sum = reward_batch_sum + reward
        success_sum = success_sum + env.success
        fail_PU_sum = fail_PU_sum + env.fail_PU
        fail_collision_sum = fail_collision_sum + env.fail_collision


        # SU sense the environment and get the sensing result (contains sensing errors)
        observation_ = env.sense()

        # Store one episode (s, a, r, s')
        for k in range(n_su):
            state = observation[k, :]
            state_ = observation_[k, :]
            DQN_RC_list[k].store_transition(state, action[k], reward[k], state_)

        # Each SU learns their DQN model
        if ((step+1) % batch_size == 0):
            for k in range(n_su):
                DQN_RC_list[k].learn()

            # Record reward, the number of success / interference / collision
            overall_reward_1.append(np.sum(reward_sum)/batch_size/n_su)
            success_hist_1.append(success_sum/n_su)
            fail_PU_hist_1.append(fail_PU_sum/n_su)
            fail_collision_hist_1.append(fail_collision_sum/n_su)

            # After one batch, refresh the record
            reward_sum = np.zeros(n_su)
            success_sum = 0
            fail_PU_sum = 0
            fail_collision_sum = 0

        # Update epsilon
        if ((step + 1) % epsilon_update_period == 0):
            for k in range(n_su):
                epsilon_index[k] = min(len(e_greedy) - 1, epsilon_index[k] + 1)
                DQN_RC_list[k].epsilon = e_greedy[epsilon_index[k]]
            print('epsilon update to %.1f' % (DQN_RC_list[k].epsilon))

        # Print the record after replace target net
        if ((step + 1) % (batch_size * replace_target_iter) == 0):
            print('Training time = %d;  success = %d;  fail_PU = %d;  fail_collision = %d' %
                    ((step + 1), success_hist_1[-1], fail_PU_hist_1[-1], fail_collision_hist_1[-1]))
            print('overall_reward_1 = %.4f' % overall_reward_1[-1])

        # swap observation
        observation = observation_


    '''
    Initialize the DQN_MLP1 (one hidden layer)
    '''
    DQN_MLP1_list = []
    for k in range(n_su):
        DQN_tmp = MLP1(env.n_actions, env.n_features,
                               learning_rate = learning_rate,
                               reward_decay=0.9,
                               e_greedy=e_greedy[0],
                               replace_target_iter=replace_target_iter,
                               memory_size=batch_size
                               )
        DQN_MLP1_list.append(DQN_tmp)


    # SUs sense the environment and get the sensing result (contains sensing errors)
    observation = env.sense()

    # Initialize some record values
    reward_sum = np.zeros(n_su)
    overall_reward_2 = []
    success_hist_2 = []
    fail_PU_hist_2 = []
    fail_collision_hist_2 = []
    success_sum = 0
    fail_PU_sum = 0
    fail_collision_sum = 0

    action = np.zeros(n_su).astype(np.int32)
    for step in range(total_episode):

        # SU choose action based on observation
        for k in range(n_su):
            action[k] = DQN_MLP1_list[k].choose_action(observation[k, :])

        # update the environment
        env.render()
        if ((step + 1) % coherence_time == 0):
            env.render_SINR()

        # SU take action and get the reward
        reward = env.access(action)

        # Record reward, the number of success / interference / collision
        reward_sum = reward_sum + reward
        # reward_batch_sum = reward_batch_sum + reward
        success_sum = success_sum + env.success
        fail_PU_sum = fail_PU_sum + env.fail_PU
        fail_collision_sum = fail_collision_sum + env.fail_collision


        # SU sense the environment and get the sensing result (contains sensing errors)
        observation_ = env.sense()

        # Store one episode (s, a, r, s')
        for k in range(n_su):
            state = observation[k, :]
            state_ = observation_[k, :]
            DQN_MLP1_list[k].store_transition(state, action[k], reward[k], state_)

        # Each SU learns their DQN model
        if ((step + 1) % batch_size == 0):
            for k in range(n_su):
                DQN_MLP1_list[k].learn()

            # Record reward, the number of success / interference / collision
            overall_reward_2.append(np.sum(reward_sum) / batch_size / n_su)
            success_hist_2.append(success_sum / n_su)
            fail_PU_hist_2.append(fail_PU_sum / n_su)
            fail_collision_hist_2.append(fail_collision_sum / n_su)

            # After one batch, refresh the record
            reward_sum = np.zeros(n_su)
            success_sum = 0
            fail_PU_sum = 0
            fail_collision_sum = 0

        # Update epsilon
        if ((step + 1) % epsilon_update_period == 0):
            for k in range(n_su):
                epsilon_index[k] = min(len(e_greedy) - 1, epsilon_index[k] + 1)
                DQN_MLP1_list[k].epsilon = e_greedy[epsilon_index[k]]
            print('epsilon update to %.1f' % (DQN_MLP1_list[k].epsilon))

        # Print the record after replace target net
        if ((step + 1) % (batch_size * replace_target_iter) == 0):
            print('Training time = %d;  success = %d;  fail_PU = %d;  fail_collision = %d' %
                  ((step + 1), success_hist_2[-1], fail_PU_hist_2[-1], fail_collision_hist_2[-1]))
            print('overall_reward_2 = %.4f' % overall_reward_2[-1])

        # swap observation
        observation = observation_


    file_folder = '.\\result\\channel_1_su_1_temporal'
    np.save(file_folder + '\\PU_TX_x', env.PU_TX_x)
    np.save(file_folder + '\\PU_TX_y', env.PU_TX_y)
    np.save(file_folder + '\\PU_RX_x', env.PU_RX_x)
    np.save(file_folder + '\\PU_RX_y', env.PU_RX_y)
    np.save(file_folder + '\\SU_TX_x', env.SU_TX_x)
    np.save(file_folder + '\\SU_TX_y', env.SU_TX_y)
    np.save(file_folder + '\\SU_RX_x', env.SU_RX_x)
    np.save(file_folder + '\\SU_RX_y', env.SU_RX_y)
    np.save(file_folder + '\\success_hist_1', success_hist_1)
    np.save(file_folder + '\\success_hist_2', success_hist_2)
    np.save(file_folder + '\\fail_PU_hist_1', fail_PU_hist_1)
    np.save(file_folder + '\\fail_PU_hist_2', fail_PU_hist_2)
    np.save(file_folder + '\\fail_collision_hist_1', fail_collision_hist_1)
    np.save(file_folder + '\\fail_collision_hist_2', fail_collision_hist_2)
    np.save(file_folder + '\\overall_reward_1', overall_reward_1)
    np.save(file_folder + '\\overall_reward_2', overall_reward_2)



