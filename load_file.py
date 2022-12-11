# Python program explaining
# save() function




# from stable_baselines3 import DQN
from train_test_mod import double_q_learn, compare_performance, q_learn, min_dist_comp, always_go_comp, SARSA, q_learn2
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
    ax.set_xlabel("Episode Number X 1000")
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




# TRAIN DOUBLE Q-LEARNING


# the array is saved in the file geekfile.npy 
rewards_results_dq = np.load('double_q.npy')
rewards_results_min_3 = np.load('min_dist_3.npy')
rewards_results_q = np.load('q.npy')
# print(rewards_results_q)
rewards_results_min_2 = np.load('min_dist_2.npy')
rewards_results_SARSA = np.load('sarsa.npy')
rewards_results_go = np.load('always_go.npy')
ave_cum_chart = []
result_chart = []

# print(rewards_results_SARSA)
print("SARSA")
print(np.mean(rewards_results_SARSA[:,20:]))
print("QL")
print(np.mean(rewards_results_q[:,20:]))
print("DQL")
print(np.mean(rewards_results_dq[:,20:]))
print("min 3")
print(np.mean(rewards_results_min_3[:,20:]))
print("min 2")
print(np.mean(rewards_results_min_2[:,20:]))
print("Go")
print(np.mean(rewards_results_go[:,20:]))




for i in range(10):
    #  = run(random_selection=True)
    ave_cum = []
    for j in range(500):
        ave_cumv = np.mean(rewards_results_SARSA[i,100*j:100*(j+1)])
        # print(ave_cumv)
        ave_cum.append(ave_cumv)
    ave_cum_chart.append(np.array(ave_cum))
    # result_chart.append(np.array(result[0:3500]))
# print(np.around(rewards_results,2))
plot_curves([np.array(ave_cum_chart)], ["SARSA"], ["y"], "Cumulative Reward", "Transit Simulation - 10 Trials")

# plot_curves([np.array(rewards_results_dq), np.array(rewards_results_q), np.array(rewards_results_min_3), np.array(rewards_results_min_2), np.array(rewards_results_SARSA), np.array(rewards_results_go)],
#         ["Double-Q", "Q", "Min Dist 3 Stops", "Min Dist 2 Stops", "SARSA", "Always Go"],
#         ["b", "k", "g", "r", "y", "c"],
#         "Cumulative Reward", "Transit Simulation - 10 Trials")