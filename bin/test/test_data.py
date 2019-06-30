#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd()))
	print(os.getcwd())
except:
	pass

#%%
import pandas as pd
import numpy as np
import tushare as ts
from bin.stock import *
import datetime

#%%
# para = Parameters(ts_code='603259.SH', start_date='20180101', end_date='20181231')
'''
# 获取参数
stockdata = StockData(para)
daily = stockdata.getDaily().drop(columns='ts_code')
cal = stockdata.getTradeCalender().drop(columns=['exchange','is_open'])
merged = pd.merge(cal, daily, left_on='cal_date', right_on='trade_date', how='left')
#print(merged.shape)
#print(merged)
#print(cal)
# print(cal.shape)
# print(daily)
# print(daily.shape)
# print(daily)
# print(stockdata.getDailyIndicator())
#%%
stockfinance = StockFinance(para)

income = stockfinance.getIncome()
balancesheet = stockfinance.getBalanceSheet().drop(columns='ts_code')
cashflow = stockfinance.getCashflow().drop(columns='ts_code')
forecast = stockfinance.getForecast().drop(columns=['ts_code', 'type', 'summary', 'change_reason'])
express = stockfinance.getExpress().drop(columns='ts_code')
dividend = stockfinance.getDividend().drop(columns='div_proc')
financeindicator = stockfinance.getFinacialIndicator().drop(columns='ts_code')
# financeaudit = stockfinance.getFinacialAudit().drop(columns='ts_code')
# finacialmain = stockfinance.getFinacialMain()
#print('income \n',income)
#print('balancesheet \n',balancesheet)
#print('cashflow \n',cashflow)
#print('forecast \n',forecast)
#print('express \n',express)
#print('dividend \n',dividend)
#print('financeindicator \n',financeindicator)
# print('financeaudit \n',financeaudit)
# print('finacialmain \n',finacialmain)

'''


#%%

'''
market = Market(para)

HSGTflow = market.getMoneyflow_HSGT().sort_values(
            by='trade_date', ascending=True, ).reset_index().drop(columns='index')
print('HSGTflow \n', HSGTflow)
margin = market.getMargin().drop(columns='exchange_id').sort_values(
            by='trade_date', ascending=True)
print('margin \n', margin)
if margin.shape[0]:
	margin = margin.groupby(by='trade_date').mean().reset_index()
print('margin \n', margin)
pledge = market.getPledgeState().drop(columns='ts_code').sort_values(
            by='end_date', ascending=True).groupby('end_date').mean().reset_index()
print('pledge \n', pledge)
repurchase = market.getRepurchase().drop(columns=['end_date','proc','exp_date']).sort_values(
            by='ann_date', ascending=True).groupby('ann_date')
print('repurchase \n', repurchase)
desterilization = market.getDesterilization().drop(columns=['holder_name','share_type']).sort_values(
            by='ann_date', ascending=True).groupby('ann_date').fillna().sum().reset_index()
print('desterilization \n', desterilization)
block = market.getBlockTrade().drop(columns=['buyer','seller']).sort_values(
            by='trade_date', ascending=True).groupby('trade_date').sum().reset_index()
print('block \n', block)

'''





#%%
'''
interest = Interest(para)
shibor = interest.getShibor()
print('shibor \n', shibor)
shiborquote = interest.getShiborQuote().drop(columns='bank').groupby(by='date').mean()
print('shibor quote \n', shiborquote)
shiborLPR = interest.getShibor_LPR()
print('shibor LPR \n', shiborLPR)
libor = interest.getLibor().drop(columns='curr_type')
print('libor \n', libor)
hibor = interest.getHibor()
print('hibor \n', hibor)
wen = interest.getWenZhouIndex()
print('wenzhou \n', wen)

'''







#%%
'''
p = pd.date_range(start='20080101',end='20181231')
print(p)
s = p.year * 10000 + p.month*100 + p.day
date = pd.DataFrame(s).rename(columns={0:'date'})
'''



#%%
'''
news = News(para)
zixun = news.getNews()
print(zixun)
'''
#%%
a = range(1, 10)
print(list(a))

#%%
import pandas as pd

data = pd.read_csv('C:\Users\longf.DESKTOP-7QSFE46\GitHub\A-Stock-Prediction-System-with-GAN-and-DRL\dataset\DailyTotal-600196.SH.csv')
data.describe()

#%%
