#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

''' 
get_daily_data.py 获取单个股票的所有每日数据，并进行拼接
时间从2008年1月1日起，到2019年4月30日为止
对于股票单日数据是一个二维表格的，需要reshape展成一维
股票每日数据，存在维度爆炸的可能
每日数据中，包括一些文本数据 
'''

__author__ = 'LongFly'

import os
import sys

sys.path.append('C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\A-Stock-Prediction-System-with-GAN-and-DRL')

import tushare as ts
import pandas as pd 
import numpy as np 
from bin.stock import *


STARTDATE = 20080101
ENDDATE = 20181231
TSCODE = '600690.SH'



def getParameter(ts_code=None):
    # 切割起始时间，按年获取数据，再进行拼接
    # 返回一个parameters的列表
    start = int(STARTDATE / 10000)
    end = int(ENDDATE / 10000)
    paralist = []
    for i in range(start, end + 1):
        para = Parameters(ts_code=ts_code, 
                            start_date=str(i)+'0101', 
                            end_date=str(i)+'1231')
        paralist.append(para)
    return paralist
            
    # 获取参数

def getDailyStock(paralist, save=False):
    '''
    获取每日的股价数据以及基本数据
    一边获取数据 一边修改列名
    '''
    total = pd.DataFrame()
    for para in paralist:
        stockdata = StockData(para)
        cal = stockdata.getTradeCalender().drop(columns=['exchange','is_open'])
        daily = stockdata.getDaily().drop(columns='ts_code').rename(columns= lambda x: 'daily_'+x) 
        daily_indicator = stockdata.getDailyIndicator().drop(columns='ts_code').rename(columns= lambda x: 'daily_indicator_'+x) 
        moneyflow = stockdata.getMoneyflow().drop(columns='ts_code').rename(columns= lambda x: 'moneyflow_'+x) 
        
        daily_total = pd.merge(
                        pd.merge(
                            pd.merge(cal, daily, left_on='cal_date', right_on='daily_trade_date', how='left'),
                                daily_indicator, left_on='cal_date', right_on='daily_indicator_trade_date', how='left'), 
                                    moneyflow, left_on='cal_date', right_on='moneyflow_trade_date', how='left')
        # 整合个股每日行情、资金信息
        res_qfq = stockdata.getRestoration(adj='qfq').drop(columns='ts_code').rename(columns= lambda x: 'res_qfq_'+x) 
        res_hfq = stockdata.getRestoration(adj='hfq').drop(columns='ts_code').rename(columns= lambda x: 'res_hfq_'+x)
        
        restoration = pd.merge(res_hfq,res_qfq, left_on='res_hfq_trade_date', right_on='res_qfq_trade_date')
        # 整合复权信息

        df = pd.merge(daily_total, restoration, left_on='cal_date', right_on='res_hfq_trade_date', how='left')
        total = total.append(df.sort_values(by='cal_date', ascending=True), ignore_index=True)
    print('Get {0} stock market data at {1} dimentions and {2} rows.'.format(TSCODE, total.shape[1], total.shape[0])) 
    if save:
        total.to_csv('dataset\\Dailystock-'+TSCODE+'.csv')  
    return total

def getDailyFinance(paralist, save=False):
    '''
    获取每日的上市公司财务信息
    '''
    total = pd.DataFrame()
    for para in paralist:
        stockdata = StockData(para)
        cal = stockdata.getTradeCalender().drop(columns=['exchange','is_open'])
        stockfinance = StockFinance(para)
        income = stockfinance.getIncome().drop(columns=['ts_code',]).rename(columns= lambda x: 'income_'+x) 
        balancesheet = stockfinance.getBalanceSheet().drop(columns='ts_code').rename(columns= lambda x: 'balancesheet_'+x) 
        cashflow = stockfinance.getCashflow().drop(columns='ts_code').rename(columns= lambda x: 'cashflow_'+x) 
        forecast = stockfinance.getForecast().drop(columns=['ts_code', 'type', 'summary', 'change_reason']).rename(columns= lambda x: 'forecast_'+x) 
        express = stockfinance.getExpress().drop(columns='ts_code').rename(columns= lambda x: 'express_'+x) 
        dividend = stockfinance.getDividend().drop(columns=['ts_code', 'div_proc']).rename(columns= lambda x: 'dividend_'+x) 
        financeindicator = stockfinance.getFinacialIndicator().drop(columns='ts_code').rename(columns= lambda x: 'financeindicator_'+x) 

        finance_total = pd.merge(cal, income, left_on='cal_date', right_on='income_ann_date', how='left')
        finance_total = pd.merge(finance_total, financeindicator, left_on='cal_date', right_on='financeindicator_ann_date', how='left')
        finance_total = pd.merge(finance_total, balancesheet, left_on='cal_date', right_on='balancesheet_ann_date', how='left')
        finance_total = pd.merge(finance_total, cashflow, left_on='cal_date', right_on='cashflow_ann_date', how='left')
        finance_total = pd.merge(finance_total, forecast, left_on='cal_date', right_on='forecast_ann_date', how='left')
        finance_total = pd.merge(finance_total, express, left_on='cal_date', right_on='express_ann_date', how='left')
        finance_total = pd.merge(finance_total, dividend, left_on='cal_date', right_on='dividend_ann_date', how='left')
        
        total = total.append(finance_total.sort_values(by='cal_date', ascending=True), ignore_index=True)
    print('Get {0} stock finance data at {1} dimentions and {2} rows.'.format(TSCODE, total.shape[1], total.shape[0])) 
    if save:
        finance_total.to_csv('dataset\\DailyFinance-' + TSCODE + '.csv')  
    return finance_total

def getDailyMarket(paralist, save=False):
    '''
    获取每日市场基本信息
    '''
    total = pd.DataFrame()
    for para in paralist:
        stockdata = StockData(para)
        cal = stockdata.getTradeCalender().drop(columns=['exchange','is_open'])
        market = Market(para)
        HSGTflow = market.getMoneyflow_HSGT().rename(columns= lambda x: 'HSGTflow_'+x) 
        margin = market.getMargin().drop(columns='exchange_id').rename(columns= lambda x: 'margin_'+x) 
        if margin.shape[0]:# 如果有记录数据 才进行聚合操作 否则会损失column数据
            margin = margin.groupby('margin_trade_date').mean().reset_index()
        pledge = market.getPledgeState().drop(columns='ts_code').rename(columns= lambda x: 'pledge_'+x) 
        if pledge.shape[0]:
            pledge = pledge.groupby('pledge_end_date').mean().reset_index()
        repurchase = market.getRepurchase().drop(columns=['end_date','proc','exp_date']).rename(columns= lambda x: 'repurchase_'+x) 
        if repurchase.shape[0]:
            repurchase = repurchase.groupby('repurchase_ann_date').mean().reset_index()
        desterilization = market.getDesterilization().drop(columns=['holder_name','share_type']).rename(columns= lambda x: 'desterilization_'+x) 
        if desterilization.shape[0]:
            desterilization = desterilization.groupby('desterilization_ann_date').mean().reset_index()
        block = market.getBlockTrade().drop(columns=['buyer','seller']).rename(columns= lambda x: 'block_'+x) 
        if block.shape[0]:
            block = block.groupby('block_trade_date').sum().reset_index()
        
        # 为限售解禁和大宗交易添加两列数据 便于接下来合并数据

        market_total = cal.merge(HSGTflow, 
                        left_on='cal_date', right_on='HSGTflow_trade_date', how='left').merge(margin, 
                            left_on='cal_date', right_on='margin_trade_date', how='left').merge(pledge, 
                                left_on='cal_date', right_on='pledge_end_date', how='left').merge(repurchase, 
                                    left_on='cal_date', right_on='repurchase_ann_date', how='left').merge(desterilization, 
                                        left_on='cal_date', right_on='desterilization_ann_date', how='left').merge(block, 
                                            left_on='cal_date', right_on='block_trade_date', how='left')
        # print(market_total)
        total = total.append(market_total.sort_values(by='cal_date', ascending=True), ignore_index=True)
    print('Get {0} daily market data at {1} dimentions and {2} rows.'.format(TSCODE, total.shape[1], total.shape[0])) 
    if save:
        total.to_csv('dataset\\Dailymarket-'+TSCODE+'.csv')  
    return total

def getDailyInterest(paralist, save=False):
    '''
    获取每日宏观经济 利率信息
    '''
    total = pd.DataFrame()
    for para in paralist:
        stockdata = StockData(para)
        cal = stockdata.getTradeCalender().drop(columns=['exchange','is_open'])
        
        interest = Interest(para)
        shibor = interest.getShibor().rename(columns= lambda x: 'shibor_'+x) 
        shiborquote = interest.getShiborQuote().drop(columns='bank').rename(columns= lambda x: 'shiborquote_'+x) 
        if shiborquote.shape[0]:
            shiborquote = shiborquote.groupby('shiborquote_date').mean().reset_index()
        shiborLPR = interest.getShibor_LPR().rename(columns= lambda x: 'shiborLPR_'+x) 
        libor = interest.getLibor().drop(columns='curr_type').rename(columns= lambda x: 'libor_'+x) 
        hibor = interest.getHibor().rename(columns= lambda x: 'hibor_'+x) 
        wen = interest.getWenZhouIndex().rename(columns= lambda x: 'wen_'+x) 

        interest_total = cal.merge(shibor, 
                        left_on='cal_date', right_on='shibor_date', how='left').merge(shiborquote, 
                            left_on='cal_date', right_on='shiborquote_date', how='left').merge(shiborLPR, 
                                left_on='cal_date', right_on='shiborLPR_date', how='left').merge(libor, 
                                    left_on='cal_date', right_on='libor_date', how='left').merge(hibor, 
                                        left_on='cal_date', right_on='hibor_date', how='left').merge(wen, 
                                            left_on='cal_date', right_on='wen_date', how='left')
        # print(market_total)
        total = total.append(interest_total.sort_values(by='cal_date', ascending=True), ignore_index=True)
    print('Get {0} interest data at {1} dimentions and {2} rows.'.format(TSCODE, total.shape[1], total.shape[0])) 
    if save:
        total.to_csv('dataset\\Dailyinterest-'+TSCODE+'.csv')  
    return total
    

def main():
    paralist = getParameter(ts_code=TSCODE)
    stock_total = getDailyStock(paralist, save=True)
    finance_total = getDailyFinance(paralist, save=True)
    market_total = getDailyMarket(paralist, save=True)
    interest_total = getDailyInterest(paralist, save=True) 

    total = stock_total.merge(finance_total, 
                on='cal_date', how='left').merge(market_total, 
                on='cal_date', how='left').merge(interest_total,
                on='cal_date', how='left')
    print('Get {0} daily total data at {1} dimentions and {2} rows.'.format(TSCODE, total.shape[1], total.shape[0])) 
    total.to_csv('dataset\\DailyTotal-'+TSCODE+'.csv')


if __name__ == "__main__":
    main()
    
