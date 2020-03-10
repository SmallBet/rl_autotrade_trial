import plaidml.keras
plaidml.keras.install_backend()
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
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

model_path = 'predict_diff.h5'
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



#load the stock close price and volumn
stock_a_frame = pandas.read_csv('0966.HK.csv',usecols=[4,6],nrows = n_rows_read,engine='python')
#stock_a_frame.dropna(axis=0, how='any', inplace=True)
stock_a = stock_a_frame.values
stock_a = stock_a.astype('float32')

 

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

stock_a_copy = np.copy(stock_a)
hsi_index_copy = np.copy(hsi_index)

#change prices to diff
for i in range(len(stock_a)):
	if(i==0):
		stock_a[i][0]=0
	else:

		stock_a[i][0]=stock_a_copy[i][0]-stock_a_copy[i-1][0]
		stock_a[i][0] = np.arctan(stock_a[i][0])/np.pi/2
		if stock_a[i][0] <=0:
			stock_a[i][0] = 0.5 + stock_a[i][0]/2
		else:
			stock_a[i][0] = 0.5 + stock_a[i][0]/2
 
	#stock_a[i][0] = np.arctan(stock_a[i][0])/np.pi/2

for i in range(len(hsi_index)):
	if(i==0):
		hsi_index[i][0]=hsi_index_copy[i][0]
	else:
		if(hsi_index[i-1][0]<=0):
			hsi_index[i][0]=hsi_index_copy[i][0]
		else:
			hsi_index[i][0]=hsi_index_copy[i][0]
	#hsi_index[i][0] = np.arctan(hsi_index[i][0]/500)/np.pi/2


plt.plot(hsi_index)
plt.show()

# normalize the dataset
hsi_index = scaler.fit_transform(hsi_index)
# normalize the dataset



print(hsi_index.shape)
#print(hsi_index[0:5])

print(stock_a.shape)
print(stock_a[:,0].shape)
#get volumn
volumn = stock_a[:,1]
volumn = np.reshape(volumn,(volumn.shape[0],1))
volumn = scaler.fit_transform(volumn)

#get price
stock_price = stock_a[:,0]
stock_price = np.reshape(stock_price,(stock_price.shape[0],1))
#stock_price = scaler.fit_transform(stock_price)

#combine index and stock price
result = np.concatenate([hsi_index,volumn,vix_index],axis=-1)


		
#save result
result = np.array(result)
print(result.shape)
#print(result[126])
rs = pd.DataFrame(result)
rs.to_csv('rs.csv')


#reshape into train and test data
window = 28
prediction = 1
amount_of_features = result.shape[1]
sequence_length = window+1
buf =[]
for index in range(n_rows_used - sequence_length):
	buf.append(result[index:index+sequence_length])
buf = np.array(buf)
train_ratio = 0.8
train_row = train_ratio * len(buf)
train = buf[:int(train_row),:]
print('train shape',train.shape)
trainX = train[:,:-1]
trainY = []
for index in range(len(trainX)):
	bu = []
	for i in range(prediction):
		bu.append(stock_price[index+window+i][0])
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
		bu.append(stock_price[int(train_row)+(index)*prediction+i+window][0])
	testY.append(bu)
testY = np.array(testY)


trainX = np.reshape(trainX,(trainX.shape[0],trainX.shape[1],amount_of_features))
testX  = np.reshape(testX,(testX.shape[0],testX.shape[1],amount_of_features))



print('trainX shape ' ,trainX.shape)
print('trainY shape ' ,trainY.shape)
print('testX  shape' ,testX.shape)

#print('trainX 5\n',trainX[0:5])
#print('trainY 5\n',trainY[0:5])


#save data
data1 = pd.DataFrame(trainY)
data1.to_csv('data1.csv')

#plt.plot(trainY)
#plt.show()

#create model
dropout = 0.1
noise_level = 0.0001
if(model_exist == False):
	model = Sequential()
	model.name = 'diff predict'
	model.add(LSTM(128,activation='relu',input_shape=(window,amount_of_features),return_sequences=True))
	model.add(Dropout(dropout))
	#model.add(Conv1D(filters=128, kernel_size=window,use_bias = True, activation='relu', input_shape=(window,amount_of_features)))
	model.add(LSTM(512,activation='relu',input_shape=(window,amount_of_features),return_sequences=False))
	model.add(Dropout(dropout))
	model.add(Dense(512,activation='relu'))
	model.add(Dense(512,activation='relu'))
	
	model.add(Dense(prediction,activation='relu'))
	adam = optimizers.Adam(lr=0.001)
	sgd  = optimizers.SGD(lr=0.001)
	model.compile(loss='mae',optimizer=adam)
else:
	model = load_model(model_path)
#optimizers
adadelta = optimizers.Adadelta(lr=0.5, rho=0.99)
ndam = optimizers.Nadam(lr=0.0001,beta_1=0.999, beta_2=0.999)
adam = optimizers.Adam(lr=0.001)
sgd  = optimizers.SGD(lr=0.0001)
rmsprop  = optimizers.RMSprop(lr=0.00001)

model.optimizer = rmsprop

#print(model.summary())


#start training
history = model.fit(trainX,trainY,batch_size=window*5,epochs=100)
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
plt2.plot(testY,color='blue',linestyle='dotted' ,label='Y test')
plt2.plot(basePoint,basePointY,marker='.',c='green',linewidth=0,linestyle=None,label='base point')
plt2.legend(loc='upper left')
plt2.show()


