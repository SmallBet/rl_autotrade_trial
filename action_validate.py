import pandas
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy
import numpy as np
import math
import keras.models as models

action_path = 'action_validate.csv'
action_frame = pd.read_csv(action_path,usecols=[1],engine='python')
action_data = action_frame.values
action_data = action_data.astype('int32')

price_path = 'rs.csv'
price_frame=pd.read_csv(price_path,usecols=[3],nrows = action_data.shape[0]+1,engine='python')
price_data = price_frame.values
price_data = price_data.astype('float32')


print(action_data.shape)
print(price_data.shape)
print(price_data)

fund = 15000.
asset = False
asset_price = 0.
fund_plot = []
precise_action = 0
correct_precise_action = 0
bound =1000
for i in range(action_data.shape[0]):
    if i <= bound:
        continue
    precise_action += 1
    if action_data[i] == 1:
        #call
        if price_data[i] <= price_data[i+1]:
            correct_precise_action += 1
    else:
        
        #put
        if price_data[i] >= price_data[i+1]:
            correct_precise_action += 1
print(correct_precise_action)
print(precise_action)
print('precise accuracy:' + str(correct_precise_action / precise_action))
    

fund_plot = np.array(fund_plot)

price_c_plot = []
price_p_plot = []
action_c_plot = []
action_p_plot = []
real_action = 0
accurate_action = 0
profit = 0
#train data end
bound = 1400
power = 500
p_plot = []
for i in range(action_data.shape[0]):
    p_plot.append(profit)
    #print(profit)
    if i != 0 and i != action_data.shape[0]-1 and action_data[i] != action_data[i-1]:
        if profit >= 5000:
            power = 1000
        if profit >= 1000:
            power = 1500
        if action_data[i] == -1:
            #put

            action_p_plot.append(i)
            price_p_plot.append(price_data[i])
            #print(price_data[i])
            #accurate if last call price lower than now
            if i> bound and len(price_c_plot)>0:
                if price_c_plot[len(price_c_plot)-1] < price_p_plot[len(price_p_plot)-1]:
                    accurate_action +=1
                profit += (- price_c_plot[len(price_c_plot)-1] + price_p_plot[len(price_p_plot)-1])* power * 2 - 10
                
                
        else:
            #call

            action_c_plot.append(i)
            price_c_plot.append(price_data[i])
            #accurate if last put price higher then now
            if i > bound and len(price_p_plot)>0:
                if price_p_plot[len(price_p_plot)-1] > price_c_plot[len(price_c_plot)-1]:
                    accurate_action +=1
                profit += (- price_c_plot[len(price_c_plot)-1] + price_p_plot[len(price_p_plot)-1])* power * 2 - 10
        if i > bound:
            real_action += 1
        continue
#plt.plot(p_plot,c = 'green')
plt.plot(price_data,c = 'yellow')
plt.plot(action_c_plot,price_c_plot,c = 'red',label='call',marker='.',linewidth=0,linestyle=None)
plt.plot(action_p_plot,price_p_plot,c = 'blue',label='put',marker='.',linewidth=0,linestyle=None)

print(accurate_action)
print(real_action)
print('accuracy of test data:'+str(accurate_action/real_action)+' profit:'+str(profit))

plt.xlabel('time(market days)') 
plt.ylabel('price') 
plt.legend()
plt.show()