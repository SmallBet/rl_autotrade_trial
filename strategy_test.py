#cat narcissism strategy test
from data_parse import parse_multiple_csv
from data_parse import parse_combine
import matplotlib.pyplot as plt
import numpy as np

def is_in_range(target,start,end):
    if target >= start and target <= end and start <= end:
        return True
    else:
        return False
time_exceed = 0
time_susses = 0
profit = 0
def cap_loss(target,p):
    global time_exceed
    global time_susses
    global profit
    if target < 0:
        #print(-np.abs(p*0.01))
        time_exceed += 1
        #return -np.abs(p*0.0175)
        #return target
    #if target > 0 and target > np.abs(p* 0.05):
        #return np.abs(p* 0.05)
        profit -= target
    if target > 0:
        time_susses += 1
        profit += target
    return target

#open high low close
#4832 rows 0966
#4882 rows 0001
#4811 rows 0066
price_data = parse_multiple_csv(['data/VZ.csv'],[[1,2,3,4]],520)
original_data = np.copy(price_data)

other_price_data = parse_multiple_csv(['data/VZ.csv'],[[1,2,3,4]],520)
weighted_data = parse_combine(other_price_data,[0,1,2,3],0)

data_len = original_data.shape[0]
initial_fund = 2500
fund = initial_fund
asset = 0
asset_price = 0
status = 1
print(data_len)
fund_plot = []
power = 500
leverage = 1
is_power = True
time_diff = 2
no_diff = False
for i in range(0,data_len-1):

    if i % 30 == 0:
        fund -= 20
    if is_power:
        power = int(fund * 0.8 / weighted_data[i][0])
    else:
        power = 500
    sep_diff = 0.01
    #if that day lowest drop below last day average

    diff = 0
    no_diff = True
    '''
    if d_i-1 > d_i-2
        and d_i-1 is accessible at d_i
        then use d_i-1
    '''

    if weighted_data[i-1][0] >= weighted_data[i-2][0] and\
            is_in_range(weighted_data[i-time_diff][0]*0.99,original_data[i][2],original_data[i][1]):
        #call

        if status == -1:
            if  diff <= sep_diff or no_diff:
            #switch status
                print('call day:'+str(i)+' price:'+str(weighted_data[i-time_diff][0]))
                fund = fund +  cap_loss(asset *(weighted_data[i-time_diff][0] - asset_price),asset *asset_price) * leverage - np.abs(asset *asset_price * 0.0003)
                #fund = fund - original_data[i][4] * 200
                if is_power:
                    power = int(fund * 0.8 / weighted_data[i][0])
                else:
                    power = 500
                asset_price = weighted_data[i-time_diff][0]*0.995#(weighted_data[i-time_diff][0] + original_data[i][2])/2
                asset = power
                status = 1
            else:
                if asset != 0:
                    print('\tend put:'+str(i)+' price:'+str(weighted_data[i-time_diff][0]))
                    fund = fund +  cap_loss(asset *(weighted_data[i-time_diff][0] - asset_price),asset *asset_price) * leverage - np.abs(asset *asset_price * 0.0003)
                    asset_price = 0
                    asset = 0
    '''
    if d_i-1 < d_i-2
        and d_i-1 is accessible at d_i
        then use d_i-1
    '''
    if weighted_data[i-1][0] <= weighted_data[i-2][0] and\
            is_in_range(weighted_data[i-time_diff][0]*1.005,original_data[i][2],original_data[i][1]):
        #put
        if status == 1:
            if diff >= -sep_diff or no_diff:
                #switch status
                print('put day:'+str(i)+' price:'+str(weighted_data[i-time_diff][0]))
                fund = fund + cap_loss(asset *(weighted_data[i-time_diff][0] - asset_price),asset *asset_price) * leverage - np.abs(asset *asset_price * 0.00003)
                #fund = fund - original_data[i][4] * 200
                if is_power:
                    power = int(fund * 0.8 / weighted_data[i][0])
                else:
                    power = 500
                asset_price = weighted_data[i-time_diff][0]*1.005#(weighted_data[i-time_diff][0] + original_data[i][1])/2
                asset = -power
                status = -1
            else:
                if asset != 0:
                    print('\tend call:'+str(i)+' price:'+str(weighted_data[i-time_diff][0]))
                    fund = fund + cap_loss(asset *(weighted_data[i-time_diff][0] - asset_price),asset *asset_price) * leverage - np.abs(asset *asset_price * 0.00003)
                    asset_price = 0
                    asset = 0
    #least negative profit which negative more than 2.5% will have a lost cap using the least profit
    if np.min([asset *(original_data[i][0] - asset_price),asset *(original_data[i][1] - asset_price),asset *(original_data[i][2] - asset_price)]) < 0 and\
        np.abs(np.min([asset *(original_data[i][0] - asset_price),asset *(original_data[i][1] - asset_price),asset *(original_data[i][2] - asset_price)])) / np.abs(asset * weighted_data[i][0]) >= 0.0175:
        '''
        if is_in_range(weighted_data[i-time_diff][0],original_data[i][2],original_data[i][1]):
            fund = fund +  cap_loss(asset *(weighted_data[i-time_diff][0] - asset_price),asset *asset_price) * leverage - np.abs(asset *asset_price * 0.0003)
            asset_price = 0
            asset = 0
        else:
        '''
        print('cap lost:'+str(i)+' price:'+str(weighted_data[i][0]))
        min_profit = np.min([asset *(original_data[i][0] - asset_price),asset *(original_data[i][1] - asset_price),asset *(original_data[i][2] - asset_price)])
        
        fund = fund + cap_loss(min_profit,asset *asset_price) * leverage - np.abs(asset *asset_price * 0.0003) - 5
        asset_price = 0
        asset = 0
    no_diff = False
    diff = (weighted_data[i-1][0] - original_data[i][3])/original_data[i][3]
    #use close price to do action for cap loss
    if diff >= -0.0175 and status == 1 :
        #min_profit = asset *(original_data[i][] - asset_price)
        fund = fund + cap_loss(asset *(original_data[i][3] - asset_price) ,asset *asset_price) * leverage - np.abs(asset *asset_price * 0.0003) - 5
        asset_price = 0
        asset = 0
    if diff <= 0.0175 and status == -1:
        min_profit = np.min([asset *(original_data[i][0] - asset_price),asset *(original_data[i][1] - asset_price),asset *(original_data[i][2] - asset_price)])
        fund = fund + cap_loss(asset *(original_data[i][3] - asset_price)* leverage,asset *asset_price) - np.abs(asset *asset_price * 0.0003) - 5
        asset_price = 0
        asset = 0
    
    fund_plot.append(fund)
    #print(fund)

print('total days:'+str(data_len))
print(time_exceed)
print(time_susses)
print('end term fund:$'+str(fund))
print('estimated return rate:'+str(fund/initial_fund*100-100)+'%')
print(profit)
plt.plot(fund_plot)
plt.xlabel('time(market days)') 
plt.ylabel('asset(log)') 
#plt.plot(weighted_data)
plt.show()