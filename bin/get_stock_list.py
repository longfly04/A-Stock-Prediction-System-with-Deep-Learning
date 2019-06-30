#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

''' 
获取上证50的股票列表
'''

__author__ = 'LongFly'

import os
import sys
sys.path.append('C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\A-Stock-Prediction-System-with-GAN-and-DRL')

import tushare as ts
import pandas as pd 
import numpy as np 
from bin.stock import *

para = Parameters()
stk_data = StockData(para)
stk_list = stk_data.getStockList()
stk_cal = stk_data.getTradeCalender()
# print(stk_list)

df = pd.DataFrame(columns=('symbol','name'))
with open('dataset\\上证50(000016).txt','r', encoding='UTF-8') as f:
    for line in f.readlines():
        l = list(line.rstrip('\n').split())
        df = df.append({'symbol':l[0].strip(), 'name':l[1].strip()}, ignore_index=True)
sz50 = pd.DataFrame()
for i in df['symbol']:
    sz50 = sz50.append(stk_list[stk_list['symbol']==i], ignore_index=True)
save = sz50[sz50['list_date']<'20080101']
print(save)
save.to_csv('dataset\\上证50_上市日期为2008年之前.csv')

