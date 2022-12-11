from env import Transit_Environment
import random
import numpy as np
# import torch as T
# from agents import DDQNAgent
from copy import deepcopy
import matplotlib.pyplot as plt


def state_agg(s, agg_factor=9):
    # define the state space 
    # 0 - 9 represent passenger number waiting at each stop
    # 10 - 12 represent bus 1 thru 3 fullness level
    # 13 - 15 represent current or next stop location
    # 16 - 18 represent if bus is currently at that stop (0 approaching stop, 1 at stop)
    
    bus_locs = np.sort(s[13:16])[::-1]
    bus_locs_idx = np.argsort(s[13:16])[::-1]
    lead_loc = bus_locs[0]
    lead_fullness = s[10+bus_locs_idx[0]]

    # if lead bus has not reached commerical hub
    # calculate waiting passengers

    if bus_locs[0] - 7 < 0:
        if sum(s[bus_locs[0]:7]) > 3:
            lead_pass = 4
        elif sum(s[bus_locs[0]:7]) > 2:
            lead_pass = 3
        elif sum(s[bus_locs[0]:7]) > 1:
            lead_pass = 2
        elif sum(s[bus_locs[0]:7]) > 0:
            lead_pass = 1
        else:
            lead_pass = 0
    else:
        lead_pass = 0


    if lead_fullness > 3:
        lead_fullness = 3
    elif lead_fullness > 2:
        lead_fullness = 2
    elif lead_fullness > 0:
        lead_fullness = 1
    else:
        lead_fullness = 0

    n_pass = sum(s[0:10])
    if n_pass > 9:
        n_pass = 3
    elif n_pass > 5:
        n_pass = 1
    elif n_pass > 2:
        n_pass = 1
    else:
        n_pass = 0

    
    dist = [bus_locs[0]-bus_locs[1], bus_locs[0]-bus_locs[2]]

    dist = [round(d/9 * agg_factor) for d in dist]

    at_stops = [s[16+i] for i in bus_locs_idx]
    s_ = [lead_loc] + dist + at_stops + [lead_fullness] + [n_pass] + [lead_pass]
    return s_

def convert_act(a, s):
    bus_locs_idx = np.argsort(s[13:16])[::-1]
    act = [0]*len(bus_locs_idx)
    for i in range(len(bus_locs_idx)):
        act[i] = deepcopy(a[bus_locs_idx[i]])
    return act

def q_learn(trial_num=10, trial_length=700, learn_starts=0):
    # save the rewards for plot
    tsteps_per_ep = 180
    lr = .001
    eps_start = .5
    eps_end = .01
    tsteps_explore = int(0.7*trial_length*tsteps_per_ep)
    eps_dec_step = (eps_end - eps_start)/tsteps_explore

    dist_agg_factor = 3
    gamma = 0.99

    env = Transit_Environment()

    running_avg_step = 10
    rewards_results = np.empty((trial_num, int(trial_length/running_avg_step)))

    # np.random.seed(0)
    # random.seed(0)

    for e in range(trial_num):
        print(f'trial {e}')
        Q = np.zeros(
            shape=(10,dist_agg_factor,dist_agg_factor,2,2,2,4,4,5,len(env.actions)))

        n_steps = 0

        eps = eps_start

        running_avg = []

        for t in range(trial_length):
            state = env.reset()
            done = 0
            score = 0
            # print(f'episode {t}')
            in_episode_count = 0
            while not done:
                # if want to discard the first N steps

                if in_episode_count < learn_starts:
                    in_episode_count += 1
                    next_state, reward, done, _ = env.step(7)
                    state = deepcopy(next_state)
                    continue

                # selection
                agg_s = state_agg(state, agg_factor=dist_agg_factor-1)

                a_idx = np.argmax(Q[tuple(agg_s)])
                # a_idx = convert_act(env.actions[a_idx], state)
                # next_state, reward, done, _ = env.step(a_idx)


                corr_act = convert_act(env.actions[a_idx], state)

                a_idx = env.actions.index(corr_act)

                next_state, reward, done, _ = env.step(a_idx)

                agg_next_s = state_agg(next_state, agg_factor=dist_agg_factor-1)
                
                # print(f'State {state} aggregated {agg_s} Next State {next_state} aggregated {agg_next_s}')
                # print(f'Epsilon {round(eps,2)}')

                next_q = np.max(Q[tuple(agg_next_s)])
                current_q = Q[tuple(agg_s + [a_idx])]
                Q[tuple(agg_s + [a_idx])] += lr*(reward+gamma*next_q - current_q)

                state = deepcopy(next_state)
                n_steps += 1
                score += reward
                in_episode_count += 1
                if eps > eps_end:
                    eps += eps_dec_step 

            if t % running_avg_step == 0:
                rewards_results[e, int(t/running_avg_step)] = np.mean(running_avg)
                running_avg = []
            running_avg.append(score)
        # if train:
        #     plot_learning(steps, scores, fig_path, epsilons=eps_history)
    return rewards_results


def double_q_learn(trial_num=10, trial_length=700, learn_starts=0):
    # save the rewards for plot
    tsteps_per_ep = 180
    lr = .001
    eps_start = .6
    eps_end = .01
    tsteps_explore = int(0.8*trial_length*tsteps_per_ep)
    eps_dec_step = (eps_end - eps_start)/tsteps_explore

    dist_agg_factor = 10
    gamma = 0.98

    env = Transit_Environment()

    # running_avg_step = 10
    running_avg_step = 10
    rewards_results = np.empty((trial_num, int(trial_length/running_avg_step)))

    # np.random.seed(0)
    # random.seed(0)

    for e in range(trial_num):
        print(f'trial {e}')
        Q_a = np.zeros(
            shape=(10,dist_agg_factor,dist_agg_factor,2,2,2,4,4,5,len(env.actions)))
        Q_b = Q_a.copy()

        n_steps = 0

        eps = eps_start

        running_avg = []

        for t in range(trial_length):
            state = env.reset()
            done = 0
            score = 0
            in_episode_count = 0
            # print(f'episode {t}')
            while not done:
                if in_episode_count < learn_starts:
                    in_episode_count += 1
                    
                    next_state, reward, done, _ = env.step(7)
                    state = deepcopy(next_state)
                    score += reward
                    continue

                # selection
                agg_s = state_agg(state, agg_factor=dist_agg_factor-1)

                avg_q_vals = (Q_a[tuple(agg_s)] + Q_b[tuple(agg_s)])/2

                a_idx = np.argmax(avg_q_vals)


                corr_act = convert_act(env.actions[a_idx], state)

                a_idx = env.actions.index(corr_act)
                

                next_state, reward, done, _ = env.step(a_idx)

                agg_next_s = state_agg(next_state, agg_factor=dist_agg_factor-1)
                
                # print(f'Epsilon {round(eps,2)}')

                q_update = np.random.choice(['A','B'])
                if q_update == 'A':
                    next_q = np.max(Q_b[tuple(agg_next_s)])
                    current_q = Q_a[tuple(agg_s + [a_idx])]            
                    Q_a[tuple(agg_s + [a_idx])] += lr*(reward+gamma*next_q - current_q)
                else:
                    next_q = np.max(Q_a[tuple(agg_next_s)])
                    current_q = Q_b[tuple(agg_s + [a_idx])]            
                    Q_b[tuple(agg_s + [a_idx])] += lr*(reward+gamma*next_q - current_q)

                state = deepcopy(next_state)
                n_steps += 1
                score += reward
                # print(np.mean(Q_a))

                if eps > eps_end:
                    eps += eps_dec_step 

            if t % running_avg_step == 0:
                rewards_results[e, int(t/running_avg_step)] = np.mean(running_avg)
                running_avg = []
            running_avg.append(score)
    return rewards_results


def q_learn2(trial_num=10, trial_length=700, learn_starts=0):
    # save the rewards for plot
    tsteps_per_ep = 180
    lr = .001
    eps_start = .6
    eps_end = .01
    tsteps_explore = int(0.8*trial_length*tsteps_per_ep)
    eps_dec_step = (eps_end - eps_start)/tsteps_explore

    dist_agg_factor = 10
    gamma = 0.98

    env = Transit_Environment()

    # running_avg_step = 10
    running_avg_step = 10
    rewards_results = np.empty((trial_num, int(trial_length/running_avg_step)))

    # np.random.seed(0)
    # random.seed(0)

    for e in range(trial_num):
        print(f'trial {e}')
        Q = np.zeros(
            shape=(10,dist_agg_factor,dist_agg_factor,2,2,2,4,4,5,len(env.actions)))
       

        n_steps = 0

        eps = eps_start

        running_avg = []

        for t in range(trial_length):
            state = env.reset()
            done = 0
            score = 0
            in_episode_count = 0
            # print(f'episode {t}')
            while not done:
                if in_episode_count < learn_starts:
                    in_episode_count += 1
                    
                    next_state, reward, done, _ = env.step(7)
                    state = deepcopy(next_state)
                    score += reward
                    continue

                # selection
                agg_s = state_agg(state, agg_factor=dist_agg_factor-1)

                a_idx = np.argmax(Q[tuple(agg_s)])


                corr_act = convert_act(env.actions[a_idx], state)

                a_idx = env.actions.index(corr_act)
                

                next_state, reward, done, _ = env.step(a_idx)

                agg_next_s = state_agg(next_state, agg_factor=dist_agg_factor-1)
                
                # print(f'Epsilon {round(eps,2)}')

                next_q = np.max(Q[tuple(agg_next_s)])
                current_q = Q[tuple(agg_s + [a_idx])]
                Q[tuple(agg_s + [a_idx])] += lr*(reward+gamma*next_q - current_q)

                state = deepcopy(next_state)
                n_steps += 1
                score += reward
                # print(np.mean(Q_a))

                if eps > eps_end:
                    eps += eps_dec_step 

            if t % running_avg_step == 0:
                rewards_results[e, int(t/running_avg_step)] = np.mean(running_avg)
                running_avg = []
            running_avg.append(score)
    return rewards_results


def act_min_dist(stop_locs, bus_at_stop, n_stops, min_dist):
    sorted_b = np.sort(stop_locs)[::-1]
    sorted_b_idxs = np.argsort(stop_locs)[::-1]
    a = [1,1,1]
    for i in range(len(stop_locs)):
        if i:
            dist = sorted_b[i-1] - sorted_b[i]
            # print(i)
        else:
            dist = sorted_b[i-1] + (n_stops) - sorted_b[i]
        if dist < min_dist and bus_at_stop[sorted_b_idxs[i]]:
            a[sorted_b_idxs[i]] = 0
    return a




def test(method='none', seed=4321):
    states = []
    bus_stop_idx = 13
    bus_loc_idx = 16

    n_buses = 3
    n_stops = 10

    random.seed(seed)
    np.random.seed(seed)

    env = Transit_Environment()
    state = env.reset()
    done = 0
    reward_count = 0

    while not done:
        if method[:-1] == 'min-dist':
            min_stop_distance = int(method[-1])
            # in this case we enforce a minimum stop distance
            stop_locations = env.state[bus_stop_idx:bus_stop_idx+n_buses]
            bus_at_stops = env.state[bus_loc_idx:bus_loc_idx+n_buses]
            a = act_min_dist(stop_locations, bus_at_stops, n_stops, min_stop_distance)
            act = env.actions.index(a)
        if method == 'random':
            # otherwise we select randomly
            act = random.randint(0, len(env.actions)-1)
        if method == 'none':
            act = env.actions.index([1, 1, 1])
        # print(env.state)
        state, reward, done, _ = env.step(act)
        reward_count = reward + reward_count
        # print(f"Time step = {env.t}, Passengers waiting at stops = {env.state[0:10]}, Bus locations = {env.state[13:16]}, Bus at stop = {env.state[16:]},Action = {env.actions[act]}, Bus Fullness, {env.state[10], env.state[11], env.state[12]} Reward = {reward}, Done = {done}")
        states.append(list(state))
    # print(env.tot_pass)
    # print(env.tot_stopped)
    # print(f"Reward count = {reward_count}")
    return states, reward_count, env.tot_pass

def compare_performance(methods, episodes=1, seeds=None):
    ep_rewards = {m: [] for m in methods}
    tot_pax = {m: [] for m in  methods}
    if seeds is None:
        seeds = np.random.randint(0,100,size=episodes)
    for ep in range(episodes):
        for m in methods:
            states, rew, pax = test(method=m, seed=seeds[ep])
            ep_rewards[m].append(rew)
            tot_pax[m].append(pax)
    fig, axs = plt.subplots(2)
    axs[0].bar(range(len(methods)), [sum(ep_rewards[m])/len(ep_rewards[m]) for m in methods], tick_label=methods)
    axs[1].bar(range(len(methods)), [sum(tot_pax[m])/len(tot_pax[m]) for m in methods], tick_label=methods)
    plt.show()
    return

# def run_ddqn(trial_num=5, trial_length=100):
#     # save the rewards for plot
#     rewards_results = np.empty((trial_num, trial_length))
    
#     env = Transit_Environment()
#     T.manual_seed(0)
#     np.random.seed(0)
#     random.seed(0)

#     for e in range(trial_num):
#         print(f'trial {e}')

#         agent = DDQNAgent(gamma=0.99, epsilon=0.6, lr=.0001, input_dims=[env.state.size],
#                         n_actions=len(env.actions), mem_size=300, eps_min=0.01, batch_size=32,
#                         replace=100, eps_dec=1e-6, algo='DDQNAgent', fc_dims=128)

#         n_steps = 0

#         for t in range(trial_length):
#             state = env.reset()
#             done = 0
#             score = 0
#             print(f'step {t}')
#             while not done:
#                 obs = np.array(state, dtype=np.float32)
#                 a_idx = agent.choose_action(obs)
#                 state, reward, done, _ = env.step(a_idx)
#                 obs_ = np.array(state, dtype=np.float32)
#                 agent.store_transition(obs, a_idx, reward, obs_, int(done))
#                 agent.learn()
#                 n_steps += 1
#                 score += reward

#             rewards_results[e,t] = score
#     return rewards_results



def min_dist_comp(trial_num=10, trial_length=700, min_dist=3):
    # save the rewards for plot
    tsteps_per_ep = 180
    lr = .001
    eps_start = .6
    eps_end = .01
    tsteps_explore = int(0.8*trial_length*tsteps_per_ep)
    eps_dec_step = (eps_end - eps_start)/tsteps_explore

    dist_agg_factor = 10
    gamma = 0.98

    n_stops = 10

    env = Transit_Environment()

    # running_avg_step = 10
    running_avg_step = 10
    rewards_results = np.empty((trial_num, int(trial_length/running_avg_step)))

    # np.random.seed(0)
    # random.seed(0)

    for e in range(trial_num):
        print(f'trial {e}')
        

        n_steps = 0

        eps = eps_start

        running_avg = []

        for t in range(trial_length):
            state = env.reset()
            done = 0
            score = 0
            in_episode_count = 0
            # print(f'episode {t}')
            while not done:
                
                in_episode_count += 1
                # print(f"state: {state[13:16]}")
                stop_locs = np.sort(state[13:16])[::-1]
                # print(stop_locs)
                bus_locs_idx = np.argsort(state[13:16])[::-1]
                bus_at_stop = [state[16+i] for i in bus_locs_idx]

                action = act_min_dist(stop_locs, bus_at_stop, n_stops, min_dist)
                # print(action)

                action = [action[i] for i in bus_locs_idx]
                # print(action)

                for i in range(len(env.actions)):
                    if action == env.actions[i]:
                        act = i

                next_state, reward, done, _ = env.step(act)
                state = deepcopy(next_state)
                
                n_steps += 1
                score += reward
                
                track = env.tot_pass
                if eps > eps_end:
                    eps += eps_dec_step 

            if t % running_avg_step == 0:
                rewards_results[e, int(t/running_avg_step)] = np.mean(running_avg)
                running_avg = []
            running_avg.append(score)
    return rewards_results




def always_go_comp(trial_num=10, trial_length=700):
    # print(f'trial {e}')
    # save the rewards for plot
    tsteps_per_ep = 180
    lr = .001
    eps_start = .6
    eps_end = .01
    tsteps_explore = int(0.8*trial_length*tsteps_per_ep)
    eps_dec_step = (eps_end - eps_start)/tsteps_explore

    dist_agg_factor = 10
    gamma = 0.98

    n_stops = 10

    env = Transit_Environment()

    # running_avg_step = 10
    running_avg_step = 10
    rewards_results = np.empty((trial_num, int(trial_length/running_avg_step)))

    # np.random.seed(0)
    # random.seed(0)

    for e in range(trial_num):
        print(f'trial {e}')

        n_steps = 0

        eps = eps_start

        running_avg = []

        for t in range(trial_length):
            state = env.reset()
            done = 0
            score = 0
            in_episode_count = 0
            # print(f'episode {t}')
            while not done:
                
                in_episode_count += 1


                next_state, reward, done, _ = env.step(7)
                state = deepcopy(next_state)
                score += reward
                

                n_steps += 1
                # print(np.mean(Q_a))

                if eps > eps_end:
                    eps += eps_dec_step 

            if t % running_avg_step == 0:
                rewards_results[e, int(t/running_avg_step)] = np.mean(running_avg)
                running_avg = []
            running_avg.append(score)
    return rewards_results



# return action from index
# def ret_action_from_idx(a_idx):
#     if a_idx == 0:
#         action = [0,0,0]
#     elif a_idx == 1:
#         action = [1,0,0]
#     elif a_idx == 2:
#         action = [0,1,0]
#     elif a_idx == 3:
#         action = [1,1,0]
#     elif a_idx == 4:
#         action = [0,0,1]
#     elif a_idx == 5:
#         action = [1,0,1]
#     elif a_idx == 6:
#         action = [0,1,1]
#     elif a_idx == 7:
#         action = [1,1,1]
#     return action

# def e_greedy(espilon):
#     greedy = np.random.binomial(1, 1-epsilon)
#     # choose random action
#     if greedy == 0:
#         a_idx = random.randint(0,7)
#         action = ret_action_from_idx(a_idx)

#     # choose greedy action
#     if greedy == 1:
#         max_Q = max(Q1[s[0],s[1],s[2],s[3],s[4],s[5],s[6],:])
#         # print(max_Q)
#         loc_max_Q = []
#         for i in range(8):
#             #print(i)
#             if Q1[s[0],s[1],s[2],s[3],s[4],s[5],s[6],i] == max_Q:
#                 loc_max_Q = np.append(loc_max_Q, [i], axis = 0)
#                 # print(loc_max_Q)
#         a_idx = int(random.choice(loc_max_Q))
#         action = ret_action_from_idx(a_idx)
#     return action




# action index
def action_idx(action):
    a_idx = action[0] - 1 + action[1] * 2 + action[2] * 4
    print(action_idx)
    return a_idx

# return action from index
def ret_action_from_idx(a_idx):
    if a_idx == 0:
        action = [0,0,0]
    elif a_idx == 1:
        action = [1,0,0]
    elif a_idx == 2:
        action = [0,1,0]
    elif a_idx == 3:
        action = [1,1,0]
    elif a_idx == 4:
        action = [0,0,1]
    elif a_idx == 5:
        action = [1,0,1]
    elif a_idx == 6:
        action = [0,1,1]
    elif a_idx == 7:
        action = [1,1,1]
    return action

def SARSA(trial_num=10, trial_length=700, learn_starts=0):
    # save the rewards for plot
    tsteps_per_ep = 180
    alpha = .1
    states = []
    bus_stop_idx = 13
    bus_loc_idx = 16

    n_buses = 3
    n_stops = 10
    epsilon = .1
    # tsteps_explore = int(0.8*trial_length*tsteps_per_ep)
    # eps_dec_step = (eps_end - eps_start)/tsteps_explore

    epsilon = .1
    alpha = .1
    gamma = .99

    env = Transit_Environment()

    # running_avg_step = 10
    running_avg_step = 10
    rewards_results = np.empty((trial_num, int(trial_length/running_avg_step)))

    # np.random.seed(0)
    # random.seed(0)

    for e in range(trial_num):
        print(f'trial {e}')
        ### initiate Q - D22
        # Q = np.zeros(((4,4,4,4,4,4,4,4,4,4,5,5,5,10,10,10,2,2,2,8)))
        Q = np.zeros((10,3,3,2,2,2,2,8), dtype="float16")
        for a1 in range(10):
            for a2 in range(3):
                for a3 in range(3):
                    for a11 in range(2):
                        for a12 in range(2):
                            for a13 in range(2):
                                for a14 in range(2):
                                    for a15 in range(8):
        
                                        Q[a1,a2,a3,a11,a12,a13,a14,a15] = -660

        n_steps = 0

        # eps = eps_start

        running_avg = []

        for t in range(trial_length):
            state = env.reset()

            s = np.zeros((16),dtype=int)

            ##################################
            ### Aggregate State ###
            ##################################
            
            lead_b = max(state[13:16])
            last_b = min(state[13:16])
            mid_b = -1

            one_lead = 0
            one_min = 0
            for i in range(3):
                # print(f"S[3] S[i]:{state[13+i]},lead_b:{lead_b}")
                if state[13 + i] != lead_b and state[13 + i] != last_b:
                    mid_b = state[13 + i]
                    mid_at_s = state[13 + i + 3]
                
                elif state[13 + i] == lead_b and one_lead == 0:
                    one_lead = 1
                   
                    lead_full = state[13 + i - 3]
                    lead_at_s = state[13 + i + 3]

                elif state[13 + i] == lead_b and one_lead == 1:
                    one_lead = 2
                    mid_at_s = state[13 + i + 3]
                    mid_b = state[13 + i]


                elif state[13 + i] == last_b and one_min == 0:
                    one_min = 1
                    last_at_s = state[13 + i + 3]
                
                elif state[13 + i] == last_b and one_min == 1:
                    one_min = 2
                    mid_at_s = state[13 + i + 3]
                    mid_b = state[13 + i]


            #############################################
            ### You can update what triggers 0,1, or 2
            #############################################

            if lead_b - mid_b < 2:
                dist_1 = 0
            elif lead_b - mid_b < 4:
                dist_1 = 1
            else:
                dist_1 = 2

            if lead_b - last_b < 3:
                dist_2 = 0
            elif lead_b - mid_b < 7:
                dist_2 = 1
            else:
                dist_2 = 2

            if lead_full == 4:
                lead_full = 1
            else:
                lead_full = 0

            if np.sum(state[0:10]) > 4:
                pass_level = 1
            else:
                pass_level = 0

            # s = [lead_b, dist_1, dist_2, lead_full, lead_at_s, mid_at_s, last_at_s]
            s = [lead_b, dist_1, dist_2, pass_level, lead_at_s, mid_at_s, last_at_s]

            

            greedy = np.random.binomial(1, 1-epsilon)
            # choose random action
            if greedy == 0:
                a_idx = random.randint(0,7)
                action = ret_action_from_idx(a_idx)

            # choose greedy action
            if greedy == 1:
                max_Q = max(Q[s[0],s[1],s[2],s[3],s[4],s[5],s[6],:])
                # print(max_Q)
                loc_max_Q = []
                for i in range(8):
                    #print(i)
                    if Q[s[0],s[1],s[2],s[3],s[4],s[5],s[6],i] == max_Q:
                        loc_max_Q = np.append(loc_max_Q, [i], axis = 0)
                        # print(loc_max_Q)
                a_idx = int(random.choice(loc_max_Q))
                action = ret_action_from_idx(a_idx)


            cumulative_reward = 0

            reward_count = 0
            states = []

            done = 0
            score = 0
            in_episode_count = 0
            while not done:

                if in_episode_count < learn_starts:
                    in_episode_count += 1
                    
                    next_state, reward, done, _ = env.step(7)
                    state = deepcopy(next_state)
                    score += reward
                    continue


                ##################################
                ### Aggregate State ###
                ##################################

                # print(state)
                
                lead_b = max(state[13:16])
                last_b = min(state[13:16])
                # print(lead_b,last_b)
                mid_b = -1

                one_lead = 0
                one_min = 0
                for i in range(3):
                    if state[13 + i] != lead_b and state[13 + i] != last_b:
                        # print(f"Mid:{i}")
                        mid_b = state[13 + i]
                        mid_at_s = state[13 + i + 3]
                        a_mid = i

                    if state[13 + i] == lead_b and one_lead == 0:
                        # print(f"Lead:{i}")
                        one_lead = 1
                        lead_full = state[13 + i - 3]
                        lead_at_s = state[13 + i + 3]
                        a_lead = i

                    elif state[13 + i] == lead_b and one_lead == 1:
                        # print(f"Mid:{i}")
                        one_lead = 2
                        mid_at_s = state[13 + i + 3]
                        mid_b = state[13 + i]
                        a_mid = i


                    if state[13 + i] == last_b and one_min == 0:
                        # print(f"last:{i}")
                        one_min = 1
                        last_at_s = state[13 + i + 3]
                        a_last = i
                    
                    elif state[13 + i] == last_b and one_min == 1:
                        # print(f"Mid:{i}")
                        one_min = 2
                        mid_at_s = state[13 + i + 3]
                        mid_b = state[13 + i]
                        a_mid = i

                # print(f"lead, mid, last loc:{lead_b, mid_b, last_b}")

                #############################################
                ### You can update what triggers 0,1, or 2
                #############################################

                if lead_b - mid_b < 2:
                    dist_1 = 0
                elif lead_b - mid_b < 4:
                    dist_1 = 1
                else:
                    dist_1 = 2

                if lead_b - last_b < 3:
                    dist_2 = 0
                elif lead_b - mid_b < 7:
                    dist_2 = 1
                else:
                    dist_2 = 2

                if lead_full == 4:
                    lead_full = 1
                else:
                    lead_full = 0

                if np.sum(state[0:10]) > 4:
                    pass_level = 1
                else:
                    pass_level = 0

                # s = [lead_b, dist_1, dist_2, lead_full, lead_at_s, mid_at_s, last_at_s]
                s = [lead_b, dist_1, dist_2, pass_level, lead_at_s, mid_at_s, last_at_s]

                # print(a_lead,a_mid,a_last)
                # print(s)
                # print(action)

                action_c = np.zeros(3, dtype=int)
                for a in range(3):
                    if a_lead == a:
                        action_c[a] = action[0]
                    if a_mid == a:
                        action_c[a] = action[1]
                    if a_last == a:
                        action_c[a] = action[2]
                
            
                action_c = [int(action_c[0]),int(action_c[1]),int(action_c[2])]
                # print(action_c)
                action_c = int(env.actions.index(action_c))
                

                # print(f"reconfigured action {action_c}")
                n_state, reward, done, _ = env.step(action_c)


                ##################################
                ### Aggregate State ###
                ##################################
                
                lead_b = max(n_state[13:16])
                last_b = min(n_state[13:16])
                mid_b = -1

                one_lead = 0
                one_min = 0
                for i in range(3):
                    if n_state[13 + i] != lead_b and n_state[13 + i] != last_b:
                        mid_b = n_state[13 + i]
                        mid_at_s = n_state[13 + i + 3]
                        n_a_mid = i

                    if n_state[13 + i] == lead_b and one_lead == 0:
                        one_lead = 1
                        lead_full = n_state[13 + i - 3]
                        lead_at_s = n_state[13 + i + 3]
                        n_a_lead = i

                    elif n_state[13 + i] == lead_b and one_lead == 1:
                        one_lead = 2
                        mid_at_s = n_state[13 + i + 3]
                        mid_b = n_state[13 + i]
                        n_a_mid = i


                    if n_state[13 + i] == last_b and one_min == 0:
                        one_min = 1
                        last_at_s = state[13 + i + 3]
                        n_a_last = i
                    
                    elif n_state[13 + i] == last_b and one_min == 1:
                        one_min = 2
                        mid_at_s = n_state[13 + i + 3]
                        mid_b = n_state[13 + i]
                        n_a_mid = i

                
                #############################################
                ### You can update what triggers 0,1, or 2
                #############################################

                if lead_b - mid_b < 2:
                    dist_1 = 0
                elif lead_b - mid_b < 4:
                    dist_1 = 1
                else:
                    dist_1 = 2

                if lead_b - last_b < 3:
                    dist_2 = 0
                elif lead_b - mid_b < 7:
                    dist_2 = 1
                else:
                    dist_2 = 2
                

                if lead_full == 4:
                    lead_full = 1
                else:
                    lead_full = 0

                if np.sum(n_state[0:10]) > 4:
                    n_pass_level = 1
                else:
                    n_pass_level = 0


                n_s = [lead_b, dist_1, dist_2, n_pass_level, lead_at_s, mid_at_s, last_at_s]




                # print(n_state)
                # print(n_s)
                greedy = np.random.binomial(1, 1-epsilon)
                # choose random action
                if greedy == 0:
                    n_a_idx = random.randint(0,7)
                    n_action = ret_action_from_idx(a_idx)

                # choose greedy action
                if greedy == 1:
                    max_Q = max(Q[n_s[0],n_s[1],n_s[2],n_s[3],n_s[4],n_s[5],n_s[6],:])
                    # print(max_Q)
                    loc_max_Q = []
                    for i in range(8):
                        #print(i)
                        if Q[n_s[0],n_s[1],n_s[2],n_s[3],n_s[4],n_s[5],n_s[6],i] == max_Q:
                            loc_max_Q = np.append(loc_max_Q, [i], axis = 0)
                            # print(loc_max_Q)
                    n_a_idx = int(random.choice(loc_max_Q))
                    n_action = ret_action_from_idx(a_idx)



                #######################################
                ### Update Q value for state action ###
                #######################################

                Q_val = Q[s[0],s[1],s[2],s[3],s[4],s[5],s[6],a_idx]
                n_Q = Q[n_s[0],n_s[1],n_s[2],n_s[3],n_s[4],n_s[5],n_s[6],n_a_idx]
                Q[s[0],s[1],s[2],s[3],s[4],s[5],s[6],a_idx] = Q_val + alpha * (reward + gamma * n_Q - Q_val)


                state = n_state
                s = n_s
                action = n_action
                a_idx = n_a_idx

                reward_count = float(reward + reward_count)


                
                n_steps += 1
                score += reward
                # print(np.mean(Q_a))

                

            if t % running_avg_step == 0:
                rewards_results[e, int(t/running_avg_step)] = np.mean(running_avg)
                running_avg = []
            running_avg.append(score)
    return rewards_results