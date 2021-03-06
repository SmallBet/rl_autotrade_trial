import pandas
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy
import numpy as np
import tensorflow as tf
import math
if_amd = False
if if_amd == True:
        import plaidml.tensorflow.keras
        plaidml.tensorflow.keras.install_backend()
        import os
        os.environ["KERAS_BACKEND"] = "plaidml.tensorflow.keras.backend"
from tensorflow.keras import Model
import tensorflow.keras.models as models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os.path
import operator
from functools import reduce

model_path = 'predict_actual_price.h5'
model_exist = False
if os.path.isfile(model_path):
	model_exist = True
	print('Model exist')
def conv(val):
    if val == np.nan:
        return 0 # or whatever else you want to represent your NaN with
    return val



#plt.show()
numpy.random.seed(7)

scaler = MinMaxScaler(feature_range=(0, 1))

n_rows_read = 3500
n_rows_used = 3000

# load vix index
vix_index_frame = pd.read_csv('^VIX.csv',usecols=[4],nrows=n_rows_read, engine='python')
vix_index = vix_index_frame.values
vix_index = vix_index.astype('float32')
# normalize the dataset
vix_index = scaler.fit_transform(vix_index)

# load hsi index
hsi_index_frame = pandas.read_csv('^HSI.csv',usecols=[4],nrows = n_rows_read, engine='python')
#hsi_index_frame.dropna(axis=0, how='any', inplace=True)
hsi_index = hsi_index_frame.values
hsi_index = hsi_index.astype('float32')
# normalize the dataset
hsi_index = scaler.fit_transform(hsi_index)


#load the stock close price and volumn
stock_a_frame = pandas.read_csv('0966.HK.csv',usecols=[4,6,1],nrows = n_rows_read,engine='python')
#stock_a_frame.dropna(axis=0, how='any', inplace=True)
stock_a = stock_a_frame.values
stock_a = stock_a.astype('float32')
# normalize the dataset
stock_a = scaler.fit_transform(stock_a)

i = 0
for i in range(len(stock_a)-1):
	if(i<len(stock_a)):
		if(np.isnan(stock_a[i][0]) or np.isnan(stock_a[i][1])):
			stock_a = np.delete(stock_a,i,axis = 0)
			hsi_index = np.delete(hsi_index,i,axis = 0)
			vix_index = np.delete(vix_index,i,axis = 0)
i = 0
for i in range(len(hsi_index)-1):
	if(i<len(hsi_index)):
		if(np.isnan(hsi_index[i]) or np.isnan(hsi_index[i])):
			stock_a = np.delete(stock_a,i,axis = 0)
			hsi_index = np.delete(hsi_index,i,axis = 0)
			vix_index = np.delete(vix_index,i,axis=0)


print(hsi_index.shape)
#print(hsi_index[0:5])

print(stock_a.shape)
print(stock_a[:,0].shape)
#get volumn
volumn = stock_a[:,1]
volumn = np.reshape(volumn,(volumn.shape[0],1))
#plt.plot(hsi_index)
#combine index and stock price
result = np.concatenate([hsi_index,volumn,vix_index],axis=-1)


		
#save result
result = np.array(result)
print(result.shape)
#print(result[126])
rs = pd.DataFrame(result)
rs.to_csv('rs.csv')


#reshape into train and test data
window = 7
prediction = 1
amount_of_features = result.shape[1]
batch_len = window+1
buf =[]
for index in range(n_rows_used - batch_len):
	buf.append(result[index:index+batch_len])
buf = np.array(buf)
train_ratio = 0.95
train_row = train_ratio * len(buf)
train = buf[:int(train_row),:]
print('train shape',train.shape)
trainX = train[:,:-1]
trainY = []
for index in range(len(trainX)):
	bu = []
	for i in range(prediction):
		bu.append(stock_a[index+window+i][0])
	trainY.append(bu)
trainY = np.array(trainY)
#trainY = train[:,-1][:,-2]



testX = buf[int(train_row):,:-1]

#testY = buf[int(train_row):,-1][:,-2]
#Generate seperate prediction test
basePoint = []
testX = []
for index in range(int(train_row),n_rows_used,prediction):
	testX.append(result[index:index+window])
	basePoint.append(index-int(train_row))
testX = np.array(testX)

testY=[]
for index in range(0,len(testX)):
	bu = []
	for i in range(prediction):
		bu.append(stock_a[int(train_row)+window+(index)*prediction+i][0])
	testY.append(bu)
testY = np.array(testY)


trainX = np.reshape(trainX,(trainX.shape[0],window,amount_of_features))
testX  = np.reshape(testX,(testX.shape[0],window,amount_of_features))



print('trainX shape ' ,trainX.shape)
print('trainY shape ' ,trainY.shape)
print('testX  shape' ,testX.shape)

print('trainX 5\n',trainX[0:5])
print('trainY 5\n',trainY[0:5])

print('testX 5\n',testX[0:5])
print('testY 5\n',testY[0:5])

#save data
data1 = pd.DataFrame(trainY)
data1.to_csv('data1.csv')

#create model

dropout = 0.1
noise_level = 0.0001
if(model_exist == False):
	model = Sequential()
	model.add(LSTM(128,activation='relu',input_shape=(window,amount_of_features),return_sequences=True))
	model.add(TimeDistributed(Dense(activation='relu', units=64)))
	model.add(Dropout(dropout))
	#model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),input_shape=(window,amount_of_features),padding='same', return_sequences=True))
	model.add(Conv1D(filters=128, kernel_size=window,use_bias = True, activation='relu', input_shape=(window,amount_of_features)))
	model.add(LSTM(512,activation='relu',input_shape=(window,amount_of_features),return_sequences=True))
	model.add(TimeDistributed(Dense(activation='relu', units=64)))
	model.add(Dropout(dropout))
	model.add(LSTM(512,activation='relu',input_shape=(window,amount_of_features),return_sequences=True))
	model.add(TimeDistributed(Dense(activation='relu', units=64)))
	model.add(Dropout(dropout))
	model.add(LSTM(512,activation='relu',input_shape=(window,amount_of_features),return_sequences=True))
	model.add(TimeDistributed(Dense(activation='relu', units=64)))
	model.add(Dropout(dropout))
	model.add(LSTM(256,activation='relu',input_shape=(window,amount_of_features),return_sequences=False))
	model.add(Dropout(dropout))
	model.add(Dense(128,activation='relu'))
	model.add(Dense(prediction,activation='relu'))
	adam = optimizers.Adam(lr=0.001)
	sgd  = optimizers.SGD(lr=0.001)
	model.compile(loss='mean_absolute_error',optimizer=adam)
else:
	model = load_model(model_path)
#optimizers
adam = optimizers.Adam(lr=0.001)
sgd  = optimizers.SGD(lr=0.001)
rmsprop  = optimizers.RMSprop(lr=0.0001)

model.optimizer = adam

#print(model.summary())


#start training
history = model.fit(trainX,trainY,batch_size=window*5,epochs=10)
model.save(model_path)

#evaluate
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))

testScore = model.evaluate(testX,testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

p = model.predict(testX)
diff = []
ratio = []
for u in range(len(testY)):
    pr = p[u][0]
    ratio.append((testY[u]/pr)-1)
    diff.append(abs(testY[u]- pr))
    #print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))

model.evaluate(testX,testY)

print('shape predict:',p.shape)
print('testY shape:',testY.shape)
p = np.reshape(p,(p.shape[0]*p.shape[1]))
testY = np.reshape(testY,(testY.shape[0]*testY.shape[1]))
print('shape predict:',p.shape)
print('testY shape:',testY.shape)

basePointY = []
for i in range(len(basePoint)):
	basePointY.append(p[basePoint[i]])
basePointY = np.array(basePointY)

plt2.plot(p,color='red', marker='|',label='Predict')
plt2.plot(testY,color='blue', label='Y test')
plt2.plot(basePoint,basePointY,marker='.',c='green',linewidth=0,linestyle=None,label='base point')
plt2.legend(loc='upper left')
plt2.show()


