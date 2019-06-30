#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'bin'))
	print(os.getcwd())
except:
	pass

#%%
import pandas as pd
import numpy as np
import tushare as ts
# from stock import *


data = pd.read_csv('C:\\Users\\longf.DESKTOP-7QSFE46\GitHub\\A-Stock-Prediction-System-with-GAN-and-DRL\\dataset\\DailyTotal-600690.SH.csv')

data.describe()


#%%
data.isna().sum()


#%%
data.index


#%%
data.columns


#%%
data.rename(columns={'Unnamed: 0':'index'}, inplace=True)


#%%
data


#%%
import matplotlib.pyplot as plt
import random


for i in range(10):
    plt.figure()
    index = random.randint(0,data.shape[1])
    print(index)
    y = data[data.columns[index]]
    x = data['index']
    plt.plot(x, y)


#%%
data


#%%
data.drop_duplicates()


#%%
data.columns.difference(data.columns)


#%%
data_dup = data.T.drop_duplicates(keep='first').T 
# 去掉重复的数据列 使用转置再转置的方式 非常牛逼


#%%
data_dup


#%%
data_dup.columns


#%%
data.columns.difference(data_dup.columns)


#%%
import matplotlib.pyplot as plt
import random


for i in range(5):
    plt.figure()
    index = random.randint(0,data_dup.shape[1])
    print(index)
    y = data_dup[data_dup.columns[index]]
    x = data_dup['index']
    
    plt.plot(x, y)


#%%
data_dup.isna().sum()


#%%
import datetime

fig = plt.figure(figsize=(14, 16))
fig, axes = plt.subplots(nrows=4, ncols=1)

x = data_dup['cal_date']
x_t = pd.to_datetime(data_dup['cal_date'], format='%Y%m%d')

y_close = data_dup['close_x_x'] # 收盘价
y_amount = data_dup['amount'] # 成交额
y_turnover = data_dup['turnover_rate'] # 换手率
y_eps = data_dup['basic_eps']# 每股收益

axes[0].set(title='Close Price', 
               xlabel='date',
               ylabel='price')
axes[0].plot(x_t,y_close)
axes[1].set(title='Amount', 
               xlabel='date',
               ylabel='price')
axes[1].plot(x_t,y_amount)
axes[2].set(title='Turnover', 
               xlabel='date',
               ylabel='rate')
axes[2].plot(x_t,y_turnover)
axes[3].set(title='income', 
               xlabel='date',
               ylabel='price')
axes[3].plot(x_t,y_eps)

plt.show()


#%%
pd.to_datetime(data['cal_date'], format='%Y%m%d')


#%%
data_dup


#%%
data.rank()


#%%
import sys

projectpath = 'C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\A-Stock-Prediction-System-with-GAN-and-DRL'
sys.path.append(projectpath)
sys.path


#%%
# import stockstats

def get_technical_indicators(dataset):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['close_x_x'].rolling(window=7).mean()
    dataset['ma21'] = dataset['close_x_x'].rolling(window=21).mean()
    
    # Create MACD
    dataset['26ema'] = pd.DataFrame(dataset['close_x_x']).ewm(span=26).mean()
    dataset['12ema'] = pd.DataFrame(dataset['close_x_x']).ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])

    # Create Bollinger Bands
    dataset['20sd'] = pd.stats.moments.rolling_std(dataset['price'],20)
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
    
    # Create Exponential moving average
    dataset['ema'] = pd.DataFrame(dataset['close_x_x']).ewm(com=0.5).mean()
    
    # Create Momentum
    dataset['momentum'] = dataset['close_x_x']-1
    
    return dataset

get_technical_indicators(data_dup)


#%%



#%%
data_dup.rank()


#%%

import stockstats as ss

stock = ss.StockDataFrame(data)

# volume delta against previous day
stock['volume_delta']

# open delta against next 2 day
stock['open_2_d']

# open price change (in percent) between today and the day before yesterday
# 'r' stands for rate.
stock['open_-2_r']

# CR indicator, including 5, 10, 20 days moving average
stock['cr']
stock['cr-ma1']
stock['cr-ma2']
stock['cr-ma3']

# volume max of three days ago, yesterday and two days later
stock['volume_-3,2,-1_max']

# volume min between 3 days ago and tomorrow
stock['volume_-3~1_min']

# KDJ, default to 9 days
stock['kdjk']
stock['kdjd']
stock['kdjj']

# three days KDJK cross up 3 days KDJD
stock['kdj_3_xu_kdjd_3']

# 2 days simple moving average on open price
stock['open_2_sma']

# MACD
stock['macd']
# MACD signal line
stock['macds']
# MACD histogram
stock['macdh']

# bolling, including upper band and lower band
stock['boll']
stock['boll_ub']
stock['boll_lb']

# close price less than 10.0 in 5 days count
stock['close_10.0_le_5_c']

# CR MA2 cross up CR MA1 in 20 days count
stock['cr-ma2_xu_cr-ma1_20_c']

# 6 days RSI
stock['rsi_6']
# 12 days RSI
stock['rsi_12']

# 10 days WR
stock['wr_10']
# 6 days WR
stock['wr_6']

# CCI, default to 14 days
stock['cci']
# 20 days CCI
stock['cci_20']

# TR (true range)
stock['tr']
# ATR (Average True Range)
stock['atr']

# DMA, difference of 10 and 50 moving average
stock['dma']

# DMI
# +DI, default to 14 days
stock['pdi']
# -DI, default to 14 days
stock['mdi']
# DX, default to 14 days of +DI and -DI
stock['dx']
# ADX, 6 days SMA of DX, same as stock['dx_6_ema']
stock['adx']
# ADXR, 6 days SMA of ADX, same as stock['adx_6_ema']
stock['adxr']

# TRIX, default to 12 days
stock['trix']
# MATRIX is the simple moving average of TRIX
stock['trix_9_sma']

# VR, default to 26 days
stock['vr']
# MAVR is the simple moving average of VR
stock['vr_6_sma']


#%%



