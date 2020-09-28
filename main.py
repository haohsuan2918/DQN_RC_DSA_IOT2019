from DSA_env import DSA_Markov
from DQN_RC import DeepQNetwork
from DQN_MLP import MLP1
from DQN_MLP import MLP2
from QLearning import QLearningTable
import matplotlib.pyplot as plt
import numpy as np
import copy



if __name__ == "__main__":

    random_seed = 3
    np.random.seed(random_seed)

    # Initialize the environment
    n_channel = 6
    n_su = 2

    env = DSA_Markov(n_channel, n_su)
    env_copy = copy.deepcopy(env)

    # training parameters
    if n_channel == 6:
        batch_size = 2000
    elif n_channel == 22:
        batch_size = 4000
    replace_target_iter = 1
    total_episode = batch_size * replace_target_iter * 140
    epsilon_update_period = batch_size * replace_target_iter * 20
    e_greedy = [0.3, 0.9, 1]
    learning_rate = 0.01

    flag_DQN_RC = True
    flag_Myopic = True
    flag_QLearning = False
    flag_DQN_MLP1 = False
    flag_DQN_MLP2 = False


    '''
    Our proposed DQN+RC
    '''
    if flag_DQN_RC:
        # Initialize the DQN_RC
        DQN_RC_list = []
        epsilon_index = np.zeros(n_su, dtype=int)
        for k in range(n_su):
            DQN_tmp = DeepQNetwork(env.n_actions, env.n_features,
                              reward_decay=0.9,
                              e_greedy= e_greedy[0],
                              replace_target_iter = replace_target_iter,
                              memory_size=batch_size,
                              lr = learning_rate
                              )
            DQN_RC_list.append(DQN_tmp)


        # SUs sense the environment and get the sensing result (contains sensing errors)
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

            # update the channel states
            env.render()
            env.render_SINR()

            # SU take action and get the reward
            reward = env.access(action)

            # Record the reward gained
            reward_sum = reward_sum + reward
            # Record the number of successful transmission
            success_sum = success_sum + env.success
            # Record the number of collisions with PU
            fail_PU_sum = fail_PU_sum + env.fail_PU
            # Record the number of collisions with SU
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

            # Print the record after replace DQN_target
            if ((step + 1) % (batch_size * replace_target_iter) == 0):
                print('Training time = %d;  success = %d;  fail_PU = %d;  fail_collision = %d' %
                        ((step + 1), success_hist_1[-1], fail_PU_hist_1[-1], fail_collision_hist_1[-1]))
                print('overall_reward_1 = %.4f' % overall_reward_1[-1])

            # swap observation
            observation = observation_

    '''
    Compare with Myopic strategy
    '''
    if flag_Myopic:
        # For fair comparison, initialize the DSA environment with the same properties
        env = copy.deepcopy(env_copy)

        # SU sense the environment and get the sensing result (contains sensing errors)
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

            # SU choose action based on myopic method
            for k in range(n_su):
                # The PU existing probability
                PU_prob = observation[k,:] * (1 - env.sense_error_prob[k, :]) + \
                          (1 - observation[k,:]) * env.sense_error_prob[k, :]

                # The current expected reward
                expected_reward = PU_prob * (env.goodToBad_prob * env.punish_interfer_PU) \
                + PU_prob * (env.stayGood_prob * np.log2(1 + env.SINR[k, :])) \
                + (1 - PU_prob) * (env.stayBad_prob * env.punish_interfer_PU) \
                + (1 - PU_prob) * (env.badToGood_prob * np.log2(1 + env.SINR[k, :]))

                # Find the action with the largest expected reward
                if np.amax(expected_reward) > 0:
                    action[k] = np.argmax(expected_reward)
                else:
                    action[k] = env.n_channel


            # update the environment based on independent Markov chains
            env.render()
            env.render_SINR()

            # SU take action and get the reward
            reward = env.access(action)

            # Record the reward gained
            reward_sum = reward_sum + reward
            # Record the number of successful transmission
            success_sum = success_sum + env.success
            # Record the number of collisions with PU
            fail_PU_sum = fail_PU_sum + env.fail_PU
            # Record the number of collisions with SU
            fail_collision_sum = fail_collision_sum + env.fail_collision

            # SU sense the environment and get the sensing result (contains sensing errors)
            observation_ = env.sense()

            if ((step + 1) % batch_size == 0):
                # Record reward, the number of success / interference / collision
                overall_reward_2.append(np.sum(reward_sum)/batch_size/n_su)
                success_hist_2.append(success_sum/n_su)
                fail_PU_hist_2.append(fail_PU_sum/n_su)
                fail_collision_hist_2.append(fail_collision_sum/n_su)

                # After one batch, refresh the record
                reward_sum = np.zeros(n_su)
                success_sum = 0
                fail_PU_sum = 0
                fail_collision_sum = 0

                # Print the record after replace DQN_target
                if ((step + 1) % (batch_size * replace_target_iter) == 0):
                    print('Training time = %d;  success = %d;  fail_PU = %d;  fail_collision = %d' %
                          ((step + 1), success_hist_2[-1], fail_PU_hist_2[-1], fail_collision_hist_2[-1]))
                    print('overall_reward_2 = %.4f' % overall_reward_2[-1])
            # swap observation
            observation = observation_


    '''
    Compare with DQN_MLP1
    '''
    if flag_DQN_MLP1:
        env = copy.deepcopy(env_copy)

        # Initialize the DQN_MLP1 (one hidden layer)
        DQN_MLP1_list = []
        epsilon_index = np.zeros(n_su, dtype=int)
        for k in range(n_su):
            DQN_tmp = MLP1(env.n_actions, env.n_features,
                          learning_rate=learning_rate,
                          reward_decay=0.9,
                          e_greedy=e_greedy[0],
                          replace_target_iter=replace_target_iter,
                          memory_size=batch_size
                          )

            DQN_MLP1_list.append(DQN_tmp)

        #SUs sense the environment and get the sensing result (contains sensing errors)
        observation = env.sense()


        # Initialize some record values
        reward_sum = np.zeros(n_su)
        reward_hist = []
        overall_reward_4 = []
        success_hist_4 = []
        fail_PU_hist_4 = []
        fail_collision_hist_4 = []
        success_sum = 0
        fail_PU_sum = 0
        fail_collision_sum = 0

        action = np.zeros(n_su).astype(np.int32)
        for step in range(total_episode):
            # SU choose action based on observation
            for k in range(n_su):
                action[k] = DQN_MLP1_list[k].choose_action(observation[k,:])

            # update the environment based on independent Markov chains
            env.render()
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
                DQN_MLP1_list[k].store_transition(state, action[k], reward[k], state_)

            # Each SU learns their DQN model
            if ((step+1) % batch_size == 0):
                for k in range(n_su):
                    DQN_MLP1_list[k].learn()

                # Record reward, the number of success / interference / collision
                overall_reward_4.append(np.sum(reward_sum)/batch_size/n_su)
                success_hist_4.append(success_sum/n_su)
                fail_PU_hist_4.append(fail_PU_sum/n_su)
                fail_collision_hist_4.append(fail_collision_sum/n_su)

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
                        ((step + 1), success_hist_4[-1], fail_PU_hist_4[-1], fail_collision_hist_4[-1]))
                print('overall_reward_4 = %.4f' % overall_reward_4[-1])

            # swap observation
            observation = observation_


    '''
    Compare with DQN_MLP2
    '''
    if flag_DQN_MLP2:
        env = copy.deepcopy(env_copy)

        # Initialize the DQN_MLP2 (two hidden layers)
        DQN_MLP2_list = []
        epsilon_index = np.zeros(n_su, dtype=int)
        for k in range(n_su):
            DQN_tmp = MLP2(env.n_actions, env.n_features,
                          learning_rate=learning_rate,
                          reward_decay=0.9,
                          e_greedy=e_greedy[0],
                          replace_target_iter=replace_target_iter,
                          memory_size=batch_size
                          )

            DQN_MLP2_list.append(DQN_tmp)


        #SUs sense the environment and get the sensing result (contains sensing errors)
        observation = env.sense()


        # Initialize some record values
        reward_sum = np.zeros(n_su)
        reward_hist = []
        overall_reward_5 = []
        success_hist_5 = []
        fail_PU_hist_5 = []
        fail_collision_hist_5 = []
        success_sum = 0
        fail_PU_sum = 0
        fail_collision_sum = 0

        action = np.zeros(n_su).astype(np.int32)
        for step in range(total_episode):
            # SU choose action based on observation
            for k in range(n_su):
                action[k] = DQN_MLP2_list[k].choose_action(observation[k,:])

            # update the environment based on independent Markov chains
            env.render()
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
                DQN_MLP2_list[k].store_transition(state, action[k], reward[k], state_)

            # Each SU learns their DQN model
            if ((step+1) % batch_size == 0):
                for k in range(n_su):
                    DQN_MLP2_list[k].learn()

                # Record reward, the number of success / interference / collision
                overall_reward_5.append(np.sum(reward_sum)/batch_size/n_su)
                success_hist_5.append(success_sum/n_su)
                fail_PU_hist_5.append(fail_PU_sum/n_su)
                fail_collision_hist_5.append(fail_collision_sum/n_su)

                # After one batch, refresh the record
                reward_sum = np.zeros(n_su)
                success_sum = 0
                fail_PU_sum = 0
                fail_collision_sum = 0

            # Update epsilon
            if ((step + 1) % epsilon_update_period == 0):
                for k in range(n_su):
                    epsilon_index[k] = min(len(e_greedy) - 1, epsilon_index[k] + 1)
                    DQN_MLP2_list[k].epsilon = e_greedy[epsilon_index[k]]
                print('epsilon update to %.1f' % (DQN_MLP2_list[k].epsilon))

            # Print the record after replace target net
            if ((step + 1) % (batch_size * replace_target_iter) == 0):
                print('Training time = %d;  success = %d;  fail_PU = %d;  fail_collision = %d' %
                        ((step + 1), success_hist_5[-1], fail_PU_hist_5[-1], fail_collision_hist_5[-1]))
                print('overall_reward_5 = %.4f' % overall_reward_5[-1])

            # swap observation
            observation = observation_

    '''
    Compare with QLearning
    '''
    if flag_QLearning:
        # For fair comparison, initialize the DSA environment with the same properties
        env = copy.deepcopy(env_copy)

        # Initialize the Q-table for each SU
        QL_list = []
        epsilon_index = np.zeros(n_su, dtype=int)
        for k in range(n_su):
            QL_tmp = QLearningTable(actions=list(range(env.n_actions)), learning_rate=learning_rate, reward_decay=0.9, e_greedy=e_greedy[0])
            QL_list.append(QL_tmp)

        # SU sense the environment and get the sensing result (contains sensing errors)
        observation = env.sense()

        # Initialize the state and state_
        state = [[] for i in range(n_su)]
        state_ = [[] for i in range(n_su)]
        for k in range(n_su):
            state[k] = observation[k, :]

        # Initialize some record values
        reward_sum = np.zeros(n_su)
        overall_reward_3 = []
        success_hist_3 = []
        fail_PU_hist_3 = []
        fail_collision_hist_3 = []
        success_sum = 0
        fail_PU_sum = 0
        fail_collision_sum = 0

        action = np.zeros(n_su).astype(np.int32)
        for step in range(total_episode):
            # SU choose action based on observation
            for k in range(n_su):
                action[k] = QL_list[k].choose_action(str(state[k]))

            # update the environment based on independent Markov chains
            env.render()
            env.render_SINR()

            # SU take action and get the reward
            reward = env.access(action)

            # Record reward / interference / collision
            reward_sum = reward_sum + reward
            # Record the number of successful transmission
            success_sum = success_sum + env.success
            # Record the number of collisions with PU
            fail_PU_sum = fail_PU_sum + env.fail_PU
            # Record the number of collisions with SU
            fail_collision_sum = fail_collision_sum + env.fail_collision

            # SU sense the environment and get the sensing result (contains sensing errors)
            observation_ = env.sense()

            # Store one episode (s, a, r, s')
            for k in range(n_su):
                state[k] = observation[k, :]
                state_[k] = observation_[k, :]

            # Each SU learns their QL model
            for k in range(n_su):
                QL_list[k].learn(str(state[k]), action[k], reward[k], str(state_[k]))

            if ((step + 1) % batch_size == 0):
                # Record reward, the number of success / interference / collision
                overall_reward_3.append(np.sum(reward_sum)/batch_size/n_su)
                success_hist_3.append(success_sum/n_su)
                fail_PU_hist_3.append(fail_PU_sum/n_su)
                fail_collision_hist_3.append(fail_collision_sum/n_su)

                # After one batch, refresh the record
                reward_sum = np.zeros(n_su)
                success_sum = 0
                fail_PU_sum = 0
                fail_collision_sum = 0

            # Update epsilon
            if ((step + 1) % epsilon_update_period == 0):
                for k in range(n_su):
                    epsilon_index[k] = min(len(e_greedy) - 1, epsilon_index[k] + 1)
                    QL_list[k].epsilon = e_greedy[epsilon_index[k]]
                print('epsilon update to %.1f' % (QL_list[k].epsilon))

            # Print the record after replace DQN_target
            if ((step + 1) % (batch_size * replace_target_iter) == 0):
                print('Training time = %d;  success = %d;  fail_PU = %d;  fail_collision = %d' %
                        ((step + 1), success_hist_3[-1], fail_PU_hist_3[-1], fail_collision_hist_3[-1]))
                print('overall_reward_3 = %.4f' % overall_reward_3[-1])

            # swap observation
            observation = observation_


    file_folder = '.\\result\\channel_%d_su_%d_punish_-2' % (n_channel, n_su)

    np.save(file_folder + '\\PU_TX_x', env.PU_TX_x)
    np.save(file_folder + '\\PU_TX_y', env.PU_TX_y)
    np.save(file_folder + '\\PU_RX_x', env.PU_RX_x)
    np.save(file_folder + '\\PU_RX_y', env.PU_RX_y)
    np.save(file_folder + '\\SU_TX_x', env.SU_TX_x)
    np.save(file_folder + '\\SU_TX_y', env.SU_TX_y)
    np.save(file_folder + '\\SU_RX_x', env.SU_RX_x)
    np.save(file_folder + '\\SU_RX_y', env.SU_RX_y)

    if flag_DQN_RC:
        np.save(file_folder + '\\success_hist_1', success_hist_1)
        np.save(file_folder + '\\fail_PU_hist_1', fail_PU_hist_1)
        np.save(file_folder + '\\fail_collision_hist_1', fail_collision_hist_1)
        np.save(file_folder + '\\overall_reward_1', overall_reward_1)
    if flag_Myopic:
        np.save(file_folder + '\\success_hist_2', success_hist_2)
        np.save(file_folder + '\\fail_PU_hist_2', fail_PU_hist_2)
        np.save(file_folder + '\\fail_collision_hist_2', fail_collision_hist_2)
        np.save(file_folder + '\\overall_reward_2', overall_reward_2)
    if flag_QLearning:
        np.save(file_folder + '\\success_hist_3', success_hist_3)
        np.save(file_folder + '\\fail_PU_hist_3', fail_PU_hist_3)
        np.save(file_folder + '\\fail_collision_hist_3', fail_collision_hist_3)
        np.save(file_folder + '\\overall_reward_3', overall_reward_3)
    if flag_DQN_MLP1:
        np.save(file_folder + '\success_hist_4', success_hist_4)
        np.save(file_folder + '\\fail_PU_hist_4', fail_PU_hist_4)
        np.save(file_folder + '\\fail_collision_hist_4', fail_collision_hist_4)
        np.save(file_folder + '\\overall_reward_4', overall_reward_4)
    if flag_DQN_MLP2:
        np.save(file_folder + '\\success_hist_5', success_hist_5)
        np.save(file_folder + '\\fail_PU_hist_5', fail_PU_hist_5)
        np.save(file_folder + '\\fail_collision_hist_5', fail_collision_hist_5)
        np.save(file_folder + '\\overall_reward_5', overall_reward_5)


