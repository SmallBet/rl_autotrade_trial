import pandas
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy
import numpy as np
import math

 
from keras import Model
import keras.models as models
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
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.layers import AveragePooling1D
from keras.layers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras import losses
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os.path
import operator
from functools import reduce
import rl.policy
from  keras.activations import linear
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.policy import MaxBoltzmannQPolicy
from rl.policy import BoltzmannQPolicy
from rl.policy import LinearAnnealedPolicy
from rl.agents.dqn import DQNAgent
from rl.core import Env
import rl_test_env
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute,Reshape
from data_parse import parse_multiple_csv
from data_parse import parse_diff
from data_parse import parse_combine
from keras import backend as K
print(K)





numpy.random.seed(7)

model_path = 'rl_test.h5'
model_exist = False
if os.path.isfile(model_path):
	model_exist = True
	print('Model exist')

n_rows_read = 2500
data = parse_multiple_csv(\
	['data/^HSI.csv','data/^VIX.csv','data/0966.HK.csv'],
	[[1,2,3,4],[1,2,3,4],[1,2,3,4,6]],\
	n_rows_read)

data = parse_combine(data,[0,1,2,3],0)

data = parse_combine(data,[1,2,3,4],1)

data = parse_combine(data,[2,3,4,5],2)

original_data = np.copy(data)


rs = pd.DataFrame(data)
rs.to_csv('rs.csv')
#data = parse_diff(data,[0,2])
print(data.shape)
print(data)


window = 14
state_size = data.shape[1]
num_actions = len(rl_test_env.LeveragingEnv.ACTION)

activation_func = 'relu'


scaler = MinMaxScaler(feature_range=(0, 1))
data   = scaler.fit_transform(data)
print(data)
dropout = 0.1

model = Sequential()
#model.add(LSTM(256,activation=activation_func,input_shape=(window,state_size),return_sequences=True))
#model.add(TimeDistributed(Dense(activation=activation_func, units=64)))
#model.add(Dropout(dropout))
#model.add(Conv1D(filters=8, kernel_size=window,use_bias = True,padding='same', activation=activation_func, input_shape=(window,state_size)))
#model.add(BatchNormalization())
model.add(LSTM(128,activation=activation_func,input_shape=(window,state_size),return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(128,activation=activation_func,input_shape=(window,state_size),return_sequences=False))
model.add(Dropout(dropout))
model.add(Dense(128,activation=activation_func))
model.add(Dense(64,activation=activation_func))
model.add(Dense(32,activation=activation_func))
model.add(Dense(num_actions,activation='linear'))
print(model.summary())


'''
action_input = Input(shape=(num_actions,), name='action_input')
observation_input = Input(shape=(1,) + data[0].shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Dense(400)(flattened_observation)
x = Activation('relu')(x)
x = Concatenate()([x, action_input])
x = Dense(300)(x)
x = Activation('relu')(x)
x = Dense(num_actions)(x)
x = Activation('linear')(x)
model = Model(inputs=[action_input, observation_input], outputs=x)
'''
train_ratio = 0.1
train_start = 0
train_end = 1700
test_end = 2000


env = rl_test_env.LeveragingEnv(data,original_data,1.0,100.0,2,3)
env.set_train_ratio(train_ratio)
memory = SequentialMemory(limit=50000, window_length=window)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=5000)
#policy = EpsGreedyQPolicy(10)
#policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=window*3, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.enable_dueling_network = True
if model_exist:
	dqn.load_weights(model_path)
	dqn.policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=.5, value_min=.1, value_test=.05, nb_steps=5000)
	
env.set_data_interval(train_start,train_end)
train_history = dqn.fit(env, nb_steps=5000,visualize=False,verbose=2,action_repetition=5)
env.set_data_interval(train_start,test_end)
print('Whole')
train_history = dqn.test(env,nb_episodes=2)
dqn.save_weights(model_path ,overwrite = True)
env.save_action_plot('action_validate.csv')
plt.axvline(x=train_end-train_start)
plt.show()
