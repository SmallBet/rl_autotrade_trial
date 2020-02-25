import pandas
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy
import numpy as np
import math
import keras.models as models
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
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import losses
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os.path
import operator
from functools import reduce

def parse_multiple_csv(paths,cols,n_rows_read):
	num_of_files = len(paths)
	index_frame = []
	index_value = []

	for i in range(num_of_files):
		index_frame.append(pd.read_csv(paths[i],usecols=cols[i],nrows= n_rows_read,engine='python'))
		index_value.append(index_frame[i].values)
		index_value[i] = index_value[i].astype('float32')

	values = np.concatenate(index_value,axis=-1)
	nrows = values.shape[0]
	i = 0
	while i < nrows:
		for j in range(values.shape[1]):
			if(np.isnan(values[i][j]) or values[i][j] == 0):
				values = np.delete(values,i,axis=0)
				nrows -= 1
				i -= 1
		i += 1
	return values

def parse_diff(data,cols):
	cols = np.array(cols)
	for i in range(data.shape[0]-1,-1,-1):
		if i != 0:
			for j in cols:
				data[i][j] = (data[i][j] - data[i-1][j])/data[i-1][j]
		else:
			for j in cols:
				data[i][j] = 0
	return data
	
def parse_combine(data,cols,insert_index):
	cols = np.array(cols)
	data = np.array(data)
	new_col = []
	for i in range(data.shape[0]):
		buf = 0
		for j in cols:
			buf += data[i][j]
		buf /= cols.shape[0]
		new_col.append(buf)
	#print(data.shape)
	data = np.delete(data,cols,axis=1)
	#print(data.shape)
	#print(new_col)
	new_col = np.array(new_col)
	return np.insert(data,insert_index,values=new_col,axis=1)

