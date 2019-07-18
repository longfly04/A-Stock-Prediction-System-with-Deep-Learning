#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

' 获取每日的新闻资讯 '

__author__ = 'LongFly'

import os
import sys

sys.path.append('C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\A-Stock-Prediction-System-with-GAN-and-DRL')

import tushare as ts
import pandas as pd 
from bin.base.stock import *

STARTDATE = 20080101
ENDDATE = 20181231
TSCODE = '600196.SH'

def getParameter(ts_code=None, start=None, end=None, by_year=False):
    # 获取每天的新闻 由于流量限制 所以参数需要切割成按日获取
    # 返回一个parameters的列表
    # year 确定参数是按年返回还是按日返回
    if by_year:
        start = int(start)
        end = int(end)
        paralist = []
        for i in range(start, end+1):
            para = Parameters(ts_code=ts_code, 
                                start_date=str(i)+'0101', 
                                end_date=str(i)+'1231',
                                year=i)
            paralist.append(para)
        return paralist
    else:
        p = pd.date_range(start=start, end=end)
        # print(p)
        s = p.year * 10000 + p.month*100 + p.day
        date = pd.DataFrame(s).rename(columns={0:'date'})
        # 获取20080101到20181231的时间列表
        paralist = []
        for i in range(len(date)-2, 0, -1):
            para = Parameters(ts_code=ts_code, 
                                start_date=str(date.iloc[i]['date']), 
                                end_date=str(date.iloc[i+1]['date']),
                                date=str(date.iloc[i]['date']))
            paralist.append(para)
        return paralist

def getNews(paralist, save=False):
    # 新闻资讯数据从2018年10月7日开始有数据 之前没有数据
    total = pd.DataFrame()
    for para in paralist:
        news = News(para)
        zixun = news.getNews()
        print('Reading the news of '+ para.start_date + '...')
        print(zixun)
        total = total.append(zixun.sort_values(by='datetime', ascending=True), ignore_index=True)
    print('Get {0} news data at {1} dimentions and {2} rows.'.format(TSCODE, total.shape[1], total.shape[0])) 
    if save:
        total.to_csv('dataset\\News-' + paralist[0].start_date + '-to-' + paralist[-1].start_date + '.csv') 
    return total

def getCompanyPublic(paralist, save=False):
    # 上市公司公告 几乎都是空的
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

def getCCTVNews(paralist, save=False):
    # CCTV 接口限制每分钟100次
    total = pd.DataFrame()
    for para in paralist:
        news = News(para)
        cctv = news.getCCTV()
        print('Reading the CCTV News of '+ str(para.date) + '...')
        print(cctv)
        total.append(cctv)
    print('Get {0} CCTV news data at {1} dimentions and {2} rows.'.format(TSCODE, total.shape[1], total.shape[0])) 
    if save:
        total.to_csv('dataset\\CCTV News-' + paralist[0].start_date + '-to-' + paralist[-1].start_date + '.csv') 
    return total

def main():
    paralist_daily = getParameter(ts_code=TSCODE, start='20181001', end='20190530')
    paralist_yearly = getParameter(ts_code=TSCODE, start='2008', end='2018', by_year=True)
    
    # public = getCompanyPublic(paralist_yearly)
    # print(public)
    news = getNews(paralist_daily)
    print(news)
    # cctv = getCCTVNews(paralist_daily, save=True)
    # print(cctv)


if __name__ == "__main__":
    main()