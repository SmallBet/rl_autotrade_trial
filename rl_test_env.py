import pandas
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy
import numpy as np
import math
import keras.models as models
from keras import Model
from keras import Input
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import ConvLSTM2D
from keras.layers import GRU
from keras.layers import Flatten
from keras.layers import GaussianNoise
from keras.layers import AveragePooling1D
from keras.layers import TimeDistributed
from keras import optimizers
from keras.optimizers import Adam
from keras import losses
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os.path
import operator
from functools import reduce

from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.policy import LinearAnnealedPolicy
from rl.agents.dqn import DQNAgent
from rl.core import Env

import gym
from gym.utils import seeding

class LeveragingEnv(Env):
    metadata = {'render.modes': ['human']}
    ACTION = [-1,1]
    def __init__(self,normalized_data,market_data,leverage_ratio,initial_fund,price_col,original_price_col):
        self.initial_fund = initial_fund
        self.fund = initial_fund
        self.leverage_ratio = leverage_ratio
        self.market_data = market_data
        self.viewer = None
        self.enable_render = False
        self.current_time = 0
        self.price_col = price_col
        self.action_space = self.ACTION
        self.observation_space = market_data
        self.fund_plot = []
        self.action_plot = []
        self.episode = 0
        self.ax = plt.axes()
        self.normalized_data = normalized_data
        self.asset = 0
        self.tdiff = 0
        self.original_price_col = original_price_col
    def set_train_ratio(self,train_ratio):
        self.train_ratio = train_ratio
    def step(self, action):

        
        if self.current_time == 0:
            self.action_plot.clear()
        action_price = self.market_data[self.current_time][self.original_price_col]
        equidity = self.fund + self.asset * action_price
        r_action = self.action_space[action] 


        """
        if(r_action<=0):
            #put
            if(self.asset>0):
                self.fund += -r_action * self.asset * action_price
                self.asset -= -r_action * self.asset
        else:
            #call
            self.asset += r_action * self.fund/action_price
            self.fund -= r_action * self.fund
        """
        time_diff = 2
        new_price = self.market_data[self.current_time+time_diff ][self.original_price_col]
        new_price_1 = self.market_data[self.current_time+time_diff][self.original_price_col]
        new_price_2 = self.market_data[self.current_time+time_diff][self.original_price_col]
        pkf = 1*(new_price_2 - action_price)+2*(new_price_1 - action_price)+3*(new_price - action_price)
        diff = (pkf/6) / action_price * 100 / 5
        #print(diff)

        
        
        
        self.current_time += 1
        if(self.current_time >= self.end_time):
            episode_over = True
        else:
            episode_over = False

        if np.abs(diff) > 1:
            diff = diff / np.abs(diff)

        if diff >0:
            #profit = 1 * r_action
            
            if r_action >= 0:
                profit = 1*r_action*np.abs(diff)
            else:
                profit = 1*r_action*np.abs(diff)
            
        else:
            
            if r_action < 0:
                profit = 1*-r_action*np.abs(diff)
            else:
                profit = -1*r_action*np.abs(diff)
            
            #profit = -1 * r_action
        self.action_plot.append([r_action,self.market_data[self.current_time-1]])
        self.tdiff += profit
        #print(r_action,'\tStatus:',self.market_data[self.current_time-1],'\tFund:',self.fund,'\tProfit:',profit)
        self.fund_plot.append(self.tdiff)
        #print(str(action)+' '+str(profit)+' '+str(action_price)+' '+str(new_price))
        return self.normalized_data[self.current_time], profit, episode_over, {'Fund':self.fund,'Profit':profit}
    def reset(self):
        if self.episode % 3 == 1:
            #self.ax.clear()
            pass
        self.fund = self.initial_fund
        self.asset = 0
        self.tdiff = 0
        self.current_time = self.start_time
        self.ax.clear()

        plt.plot(self.fund_plot,label = str(self.episode)+' episode')
        plt.draw() 
        plt.pause(0.01)

            #plt.plot(self.fund_plot,label = str(self.episode)+' episode')
        self.fund_plot.clear()
        self.episode += 1
        return self.normalized_data[self.current_time]
    def save_action_plot(self,action_path):
        sa = pd.DataFrame(self.action_plot)
        sa.to_csv(action_path)
    def set_data_interval(self,start,end):
        self.start_time = start
        self.end_time = end
    def render(self, mode='human', close=False):
        pass

    def take_action(self, action):
        pass
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]