import random
import numpy as np
# from agents import DDQNAgent
import os
# from utils import plot_learning
import matplotlib.pyplot as plt
import gym
from gym import spaces
# from stable_baselines3.common.env_checker import check_env
# import torch as T
from copy import deepcopy
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Transit_Environment(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # define the state space 
        # 0 - 9 represent passenger number waiting at each stop
        # 10 - 12 represent bus 1 thru 3 fullness level
        # 13 - 15 represent current or next stop location
        # 16 - 18 represent if bus is currently at that stop (0 approaching stop, 1 at stop)

        self.actions  = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]]

        self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
        
        # define the start state
        # no waiting passangers - empty buses - random location
        self.start_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.random.randint(0,9), np.random.randint(0,9), np.random.randint(0,9), 1, 1, 1], dtype=np.uint8)

        # for GYM

        # Example when using discrete actions:
        self.action_space = spaces.Discrete(len(self.actions),)
        # Example for using image as input:
        state_low = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
        state_high = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 9, 9, 9, 1, 1, 1], dtype=np.uint8)
        self.observation_space = spaces.Box(low=state_low, high=state_high,
                                            shape=(self.state.size,), dtype=np.uint8)

        # maximal time steps
        self.max_time_steps = 180

        # track the time step
        self.t = 0

        # time between stops
        self.bus_time_between_stops = np.random.normal(3,1,100)

        # max capacity
        self.max_capacity = 4

        # bus just arrived
        self.bus_arrive_1 = 0
        self.bus_arrive_2 = 0 
        self.bus_arrive_3 = 0 

        # bus next stop info
        self.b1_time_2_stop = 0
        self.b2_time_2_stop = 0
        self.b3_time_2_stop = 0

        # total number of passengers generated
        self.tot_pass = 0
        self.tot_stopped = 0

        # extra parameters
        self.hubs = [7,8,9]
        self.dwell_time_per_pax = 0.3

    def pull_value(self,distro):
        # pull value from normal distro
        value = np.random.choice(distro)
        # minimum time to next stop is 1 minute
        if value < 1:
            value = 1
        return np.around(value)
    
    def passenger_generation(self, stop, location):
        """
        Stop information allows the agent to generate based on hub or residential
        location - which stop
        stop - state information containing how many passengers are at the stop
        """
        if location in self.hubs:
            pass_added = np.random.binomial(1, .015)
            self.tot_pass = self.tot_pass + pass_added
            stop = stop + pass_added
        else:
            pass_added = np.random.binomial(1, .05)
            self.tot_pass = self.tot_pass + pass_added
            stop = stop + pass_added

        if stop > 3:
            self.tot_stopped = self.tot_stopped + 1
            stop = 3

        return stop

    def enter_bus(self, bus_fullness, stop_fullness):
        """
        Transfers people from bus stop onto bus
        Accepts everyone as long as there is room
        """
        room_left = self.max_capacity - bus_fullness
        if stop_fullness > room_left:
            new_bus_fullness = bus_fullness + room_left
            new_stop_fullness = stop_fullness - room_left
            pax_entered  = room_left
        else:
            new_bus_fullness = bus_fullness + stop_fullness
            new_stop_fullness = stop_fullness - stop_fullness
            pax_entered = stop_fullness

        return new_bus_fullness, new_stop_fullness, pax_entered

    def exit_bus(self, bus_fullness, location):
        """
        Transfers people from bus to stop
        """
        pax_exited = 0
        if location in self.hubs:
            bus_tot = bus_fullness
            for i in range(bus_fullness):
                off = np.random.binomial(1, .5)
                bus_tot = bus_tot - off
                pax_exited += off
        else:
            bus_tot = bus_fullness
            for i in range(bus_fullness):
                off = np.random.binomial(1, .15)
                bus_tot = bus_tot - off
                pax_exited += off
        return bus_tot, pax_exited
    

    def reset(self):
        """
        Reset the agent's state to the start state [0, 0]
        Return both the start state and reward
        """
        # reset the agent state to be [0, 0]
        self.state = self.start_state

        # reset the time step tracker
        self.t = 0
        self.tot_pass = 0

        # initial passenger generation
        for i in range(10):
            self.state[i] = self.passenger_generation(self.state[i], i)
        
        return self.state

    def step(self, action):
        """
        Args:
            state: 
            act: an integer from 0 to len(self.actions)
        Output args:
            next_state: a list variable containing x, y integer coordinates (i.e., [1, 1])
            reward: an integer. it can be either 0 or 1.
        """
        act = self.actions[action]
        # print(self.state[])

        # Increase the time step
        self.t += 1

        ###########################
        ### Passenger Generation
        ###########################

        for i in range(10):
            self.state[i] = self.passenger_generation(self.state[i], i)

        pax_activity_per_bus = [0]*3
        ###########################
        ### Passenger enter bus
        ###########################

        # if bus 1 is at a stop
        if self.state[16] == 1:
            stop_number = self.state[13]
            self.state[10], self.state[stop_number], pax = self.enter_bus(self.state[10], self.state[stop_number])
            pax_activity_per_bus[0] += pax
        # if bus 2 is at a stop
        if self.state[17] == 1:
            stop_number = self.state[14]
            self.state[11], self.state[stop_number], pax = self.enter_bus(self.state[11], self.state[stop_number])
            pax_activity_per_bus[1] += pax
        # if bus 3 is at a stop
        if self.state[18] == 1:
            stop_number = self.state[15]
            self.state[12], self.state[stop_number], pax = self.enter_bus(self.state[12], self.state[stop_number])
            pax_activity_per_bus[2] += pax
        ########################
        ### Passenger Exit
        ########################

        if self.bus_arrive_1 == 1:
            self.bus_arrive_1 = 0
            bus_fullness = self.state[10]
            self.state[10], pax = self.exit_bus(bus_fullness, self.state[13])
            pax_activity_per_bus[0] += pax
        if self.bus_arrive_2 == 1:
            self.bus_arrive_2 = 0
            bus_fullness = self.state[11]
            self.state[11], pax = self.exit_bus(bus_fullness, self.state[14])
            pax_activity_per_bus[1] += pax
        if self.bus_arrive_3 == 1:
            self.bus_arrive_3 = 0
            bus_fullness = self.state[12]
            self.state[12], pax = self.exit_bus(bus_fullness, self.state[15])
            pax_activity_per_bus[2] += pax
                    
        ##########################################
        ### Bus movement and action decisions
        ##########################################

        # update state based on action
        # means if action selects moving to next state and bus is at stop:
      
        if self.state[16] == 1:
            self.b1_time_2_stop = 0

        if self.state[17] == 1:
            self.b2_time_2_stop = 0

        if self.state[18] == 1:
            self.b3_time_2_stop = 0

        if act[0] == 1 and self.state[16] == 1:
            self.b1_time_2_stop = int(self.pull_value(self.bus_time_between_stops)+self.dwell_time_per_pax*pax_activity_per_bus[0])
            self.state[16] = 0
            if self.state[13] < 9:
                self.state[13] = self.state[13] + 1
            else:
                self.state[13] = 0

        if act[1] == 1 and self.state[17] == 1:
            self.state[17] = 0
            self.b2_time_2_stop = int(self.pull_value(self.bus_time_between_stops)+self.dwell_time_per_pax*pax_activity_per_bus[1])
            if self.state[14] < 9:
                self.state[14] = self.state[14] + 1
            else:
                self.state[14] = 0

        if act[2] == 1 and self.state[18] == 1:
            self.state[18] = 0
            self.b3_time_2_stop = int(self.pull_value(self.bus_time_between_stops)+self.dwell_time_per_pax*pax_activity_per_bus[2])
            if self.state[15] < 9:
                self.state[15] = self.state[15] + 1
            else:
                self.state[15] = 0
        
        # (Joseph) advancing the clock 1 time step
        if self.b1_time_2_stop == 0:
            self.bus_arrive_1 = 1
            self.state[16] = 1
        else:
            self.b1_time_2_stop = self.b1_time_2_stop - 1

        if self.b2_time_2_stop == 0:
            self.bus_arrive_2 = 1
            self.state[17] = 1
        else:
            self.b2_time_2_stop = self.b2_time_2_stop - 1

        if self.b3_time_2_stop == 0:
            self.bus_arrive_3 = 1
            self.state[18] = 1
        else:
            self.b3_time_2_stop = self.b3_time_2_stop - 1

        ##########################################
        reward = -float(np.sum(self.state[0:10]) + 0.5*np.sum(self.state[10:13]))

        # Check the termination
        if self.t == 180:
            done = True
        else:
            done = False
        # print(f"time to next stop b1: {self.b1_time_2_stop}, b2:{self.b2_time_2_stop}, b3:{self.b3_time_2_stop}")
        return self.state, reward, done, {}

    def render(self, mode='human'):
        ...
    
    def close(self):
        ...

