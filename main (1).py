# from stable_baselines3 import DQN
from train_test_mod_pass import q_learn,double_q_learn,min_dist_comp,SARSA,always_go_comp
import matplotlib.pyplot as plt
import random
import numpy as np
from pandas import array
import tqdm.notebook as tqdm
import matplotlib.pyplot as plt
from scipy.stats import poisson
import tqdm
from tempfile import TemporaryFile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
# from utils import plot_curves



def plot_curves(arr_list, legend_list, color_list, ylabel, fig_title):
    """
    Args:
        arr_list (list): list of results arrays to plot
        legend_list (list): list of legends corresponding to each result array
        color_list (list): list of color corresponding to each result array
        ylabel (string): label of the Y axis

        Note that, make sure the elements in the arr_list, legend_list and color_list are associated with each other correctly.
        Do not forget to change the ylabel for different plots.
    """
    # set the figure type
    fig, ax = plt.subplots(figsize=(12, 8))
    # print(len(arr_list))
    # print(len(legend_list))
    # print(len(color_list))

    # PLEASE NOTE: Change the labels for different plots
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Episode Number X 10")
    # print(arr_list)
    # ploth results
    h_list = []
    for arr, legend, color in zip(arr_list, legend_list, color_list):
        # compute the standard error
        arr_err = arr.std(axis=0) / np.sqrt(arr.shape[0])
        # plot the mean
        h, = ax.plot(range(arr.shape[1]), arr.mean(axis=0), color=color, label=legend)
        # plot the confidence band
        arr_err *= 1.96
        ax.fill_between(range(arr.shape[1]), arr.mean(axis=0) - arr_err, arr.mean(axis=0) + arr_err, alpha=0.2,
                        color=color)
        # save the plot handle
        h_list.append(h)

    # plot legends
    ax.set_title(f"{fig_title}")
    ax.legend(handles=h_list)

    # save the figure
    plt.savefig(f"{fig_title}.png", dpi=200)

    plt.show()



if __name__ == '__main__':

    random.seed(1234)
    np.random.seed(1234)
    os.getcwd()

    # TRAIN DOUBLE Q-LEARNING
    rewards_results_dq = min_dist_comp(trial_num=10, trial_length=1000, min_dist=3)
    
    # # # the array is saved in the file geekfile.npy 
    # np.save('pass_count.npy', rewards_results_dq)

    # print(rewards_results_dq)
    
    # print(rewards_results_q)
    rewards_results_min_3 = min_dist_comp(trial_num=10, trial_length=1000, min_dist=3)
    np.save('min_dist_3.npy', rewards_results_min_3)
    os.getcwd()
    
    rewards_results_min_2 = min_dist_comp(trial_num=10, trial_length=1000, min_dist=2)
    rewards_results_q = q_learn(trial_num=10, trial_length=1000, learn_starts=1)
    np.save('q.npy', rewards_results_q)
    np.save('min_dist_2.npy', rewards_results_min_2)
    rewards_results_SARSA = SARSA(trial_num=10, trial_length=1000, learn_starts=1)
    np.save('sarsa50k.npy', rewards_results_SARSA)
    rewards_results_go = always_go_comp(trial_num=10, trial_length=1000)
    np.save('always_go.npy', rewards_results_go)
    # print(np.around(rewards_results,2))

    plot_curves([np.array(rewards_results_dq), np.array(rewards_results_q), np.array(rewards_results_min_3), np.array(rewards_results_min_2), np.array(rewards_results_SARSA), np.array(rewards_results_go)],
            ["Double-Q", "Q", "Min Dist 3 Stops", "Min Dist 2 Stops", "SARSA", "Always Go"],
            ["b", "k", "g", "r", "y", "c"],
            "Cumulative Reward", "Transit Simulation - 10 Trials")
    #mplot_curves([rewards_results_dq], ["SARSA"], ["y"], "Cumulative Reward", "Transit Simulation - 10 Trials")


    # COMPARE PERFORMANCE
    # compare_performance(['random', 'none', 'min-dist1', 'min-dist2', 'min-dist3'], episodes=200)

    # rews = run_ddqn(trial_length=300, trial_num=10)
    # plot_curves([rews], ['DDQN'], ['green'], 'average reward')


    # params = {
    #     'lr': [1e-3,3e-2,5e-2],
    #     'gamma': [0.99,0.9,0.9],
    #     'steps': [50,80,120]
    # }
    # for trial in range(1):
    #     env = Transit_Environment()
        # model = DQN(
        #     "MlpPolicy", env, tensorboard_log="./a2c_tensorboard/", 
        #     seed=1234, learning_rate=params['lr'][trial], gamma=params['gamma'][trial],
        #     n_steps=params['steps'][trial])
        # model = DQN(
        #     "MlpPolicy", env, tensorboard_log="./dqn_tensorboard/", 
        #     seed=1234, buffer_size=10_000, batch_size=64, learning_starts=1_000,
        #     exploration_fraction=.6, exploration_final_eps=.01, exploration_initial_eps=.9, 
        #     learning_rate=0.0001, train_freq=100, target_update_interval=1_000)
        # model.learn(total_timesteps=100_000)

