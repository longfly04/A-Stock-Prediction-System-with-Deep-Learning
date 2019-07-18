''' 
get_minutes_data.py 获取单个股票分钟数据，
可采集的数据包括股票、基金、期货、期权、指数、数字货币
时间以1分钟为单位

获取以分钟为单位的财经新闻，结合Bert获取短期市场行情数据
'''

__author__ = 'LongFly'

import os
import sys
import arrow

sys.path.append('C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\A-Stock-Prediction-System-with-GAN-and-DRL')

import tushare as ts
import pandas as pd 
import numpy as np 
from bin.base.stock import *

STARTDATE = '20180101'
ENDDATE = '20181231'
TSCODE = '600690.SH'

def getMinutesDataList(ts_code=None, 
                            freq='1min', 
                            span=6, 
                            asset='E', 
                            adj=None,
                            adjfactor=False,
                            factors=['tor', 'vr'], 
                            ma=[7, 21],):
    '''
    概述：
        将Parameter按照时间切割，通过API接口获取数据后再拼接数据，返回一个parameters的列表；
        受API限制，每次只能返回7000行数据，考虑到每天数据量为4个小时，240分钟，每个月数据量为5000-6000行，
        所以把时间按月拆分，但是由于是短期交易，所以并不需要太长的历史数据
    参数：
        ts_code：股票代码
        freq：交易频率：1min 5min 15min 30min 60min
        span：时间范围 默认6个月
    '''
    now = arrow.now()
    print("The current time is %s %s." + %(str(now.date()), str(now.time())))
    # 将时间字符串化
    datalist = []
    for i in range(span):
        trade_date_start = str(now.shift(months=-(i+1)).date())
        trade_date_end = str(now.shift(months=-i).date())
        data = General_API(ts_code=ts_code, 
                            start_date=trade_date_start, 
                            end_date=trade_date_end,
                            asset=asset, 
                            adj=adj,
                            adjfactor=adjfactor,
                            factors=factors, 
                            freq=freq, 
                            ma=ma,
                            ).getMinuteStock()
        data = data.sort_values(by='trade_time', ascending=True).reset_index(drop=True)
        datalist.append(data)
    return datalist

def main():
	data = getMinutesDataList(ts_code=TSCODE, span=3)
	print(data)

if __name__ == '__main__':
	main()