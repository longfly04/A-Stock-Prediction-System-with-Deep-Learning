''' 
get_minutes_data.py 获取单个股票分钟数据，
可采集的数据包括股票、基金、期货、期权、指数、数字货币
时间以1分钟为单位

获取以分钟为单位的财经新闻，结合Bert获取短期市场行情数据

获取新闻联播文字稿，提取财经信息，为下一个交易日走势预测提供帮助

'''

__author__ = 'LongFly'

import os
import sys
sys.path.append('C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\A-Stock-Prediction-System-with-GAN-and-DRL')

import arrow
import time
import tushare as ts
import pandas as pd 
import numpy as np 

from bin.base.stock import *

STARTDATE = '2019-01-01'
ENDDATE = '2019-07-01'
TSCODE = '600690.SH'

def getParameter(ts_code=None, start=None, end=None, by='day'):
    '''
    概述：
        获取每天的参数，由于流量限制，所以参数需要切割成按日获取
        返回一个parameters的列表
        确定参数是按年返回还是按日返回

    参数：
        ts_code:股票代码
        start：开始日期
        end：结束日期
        by：day，month，year 以日、月、年为单位返回参数
    返回：
        paralist：参数列表
    '''
    start = arrow.get(start)
    end = arrow.get(end)
    if by=='year':# 按年为单位获取数据，一般用于公司财报
        paralist = []
        for i in range(start.year, end.year):
            para = Parameters(ts_code=ts_code, 
                                start_date=str(i)+'0101', 
                                end_date=str(i)+'1231',
                                year=i)
            paralist.append(para)
        return paralist
    elif by=='month':# 按月为单位获取数据，一般用于新闻联播
        pass
    elif by=='day':# 按天为单位获取数据，一般用于财经信息
        start = start.date()
        end = end.date()
        p = pd.date_range(start=start, end=end)
        date = pd.DataFrame(p).rename(columns={0:'date'})
        # 获取20080101到20181231的时间列表
        paralist = []
        for i in range(len(date)-2, 0, -1):
            para = Parameters(ts_code=ts_code, 
                                start_date=str(date.iloc[i]['date']), 
                                end_date=str(date.iloc[i+1]['date']),
                                date=str(date.iloc[i]['date']))
            paralist.append(para)
        return paralist

def getCompanyPublic(paralist, save=False):
    '''
    概述：
        上市公司公告，几乎都是空的，参数设置为按年获取
    参数：
        paralist：Parameters类的列表
    返回：
        DataFrame类的数据
    '''
    total = pd.DataFrame()
    for para in paralist:
        news = News(para)
        pub = news.getCompanyPublic(year=para.year)
        print('Reading the public of '+ str(para.year) + '...')
        print(pub)
        total = total.append(pub)
    print('Get {0} company public data at {1} dimentions and {2} rows.'.format(TSCODE, total.shape[1], total.shape[0])) 
    if save:
        total.to_csv('dataset\\Company Public-' + paralist[0].start_date + '-to-' + paralist[-1].start_date + '.csv') 
    return total

def getCCTVNews(paralist, save=False, sleep=0):
    '''
    概述：
        CCTV 接口限制每分钟100次,参数可以设置为按月获取
    参数：
        paralist：Parameters类的列表
        save:是否本地存储为csv文件
        sleep：请求间延迟时间，单位秒
    返回：
        DataFrame类的数据
    '''
    total = pd.DataFrame()
    for para in paralist:
        news = News(para)
        cctv = news.getCCTV()
        print('Reading the CCTV News of '+ str(para.date) + '...')
        print(cctv)
        total.append(cctv)
        time.sleep(sleep)
    print('Get {0} CCTV news data at {1} dimentions and {2} rows.'.format(TSCODE, total.shape[1], total.shape[0])) 
    if save:
        total.to_csv('dataset\\CCTV News-' + paralist[0].start_date + '-to-' + paralist[-1].start_date + '.csv') 
    return total

def getNews(paralist, save=False, sleep=0):
    '''
    概述：
        新闻资讯数据从2018年10月7日开始有数据，之前没有数据，参数设置为按天获取
    参数：
        paralist：Parameters类的列表
        save：数据是否本地存储为csv文件
        sleep：请求间延迟时间，单位秒
    返回：
        DataFrame类的数据
    '''
    total = pd.DataFrame()
    for para in paralist:
        news = News(para)
        zixun = news.getNews()
        print('Reading the news of '+ para.start_date + '...')
        print(zixun)
        total = total.append(zixun.sort_values(by='datetime', ascending=True), ignore_index=True)
        time.sleep(sleep)
    print('Get {0} news data at {1} dimentions and {2} rows.'.format(TSCODE, total.shape[1], total.shape[0])) 
    if save:
        total.to_csv('dataset\\News-' + paralist[0].start_date + '-to-' + paralist[-1].start_date + '.csv') 
    return total

def getMinutesStock(ts_code=None, 
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
        span：时间范围，默认6个月
        asset：查询资产的类型：E股票 I沪深指数 C数字货币 FT期货 FD基金 O期权，默认E
        adj：复权类型(只针对股票)：None未复权 qfq前复权 hfq后复权 , 默认None
        adjfactor：复权因子，在复权数据是，如果此参数为True，返回的数据中则带复权因子，默认为False。
        factors：股票因子（asset='E'有效）支持 tor换手率 vr量比
        ma:均线，支持任意合理int数值
    返回：
        DataFrame类型数据
    '''
    now = arrow.now()
    print("The current time is %s %s." %(str(now.date()), str(now.time())))
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
        try:
            data = data.sort_values(by='trade_time', ascending=True).reset_index(drop=True)
            datalist.append(data)
        except Exception as e:
            print('Get minutes stock failed.\n', e)
    return pd.DataFrame(datalist)

def main():
    data = getMinutesStock(ts_code=TSCODE, span=3)
    print(data)
    paralist = getParameter(ts_code=TSCODE, start=STARTDATE, end=ENDDATE)
    data = getNews(paralist, sleep=2)
    print(data)

if __name__ == '__main__':
    main()