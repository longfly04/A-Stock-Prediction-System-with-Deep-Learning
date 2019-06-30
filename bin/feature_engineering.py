#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

' 特征工程 '

__author__ = 'LongFly'

import tushare as ts
import pandas as pd 
from utils import *
import time,datetime
import numpy as np
from scipy.fftpack import fft, ifft
import datetime
import math

# import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
import sys
sys.path.append('C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\A-Stock-Prediction-System-with-GAN-and-DRL')
from bin.stock import *

import json
'''
数据预处理和特征工程
对股价521维指标进行去重、生成技术指标并通过VAE进行编码输出300维向量
对每日新闻数据通过bert service进行编码 生成768维向量
'''

def drop_and_fill(data):# 去掉重复的列数据 将空值填上数据 符合时间序列的连续性 输入为股价数据集 输出为去重之后的股价数据集
    data_new = data.T.drop_duplicates(keep='first').T 
    # 去掉重复的数据列 使用转置再转置的方式 非常牛逼
    data_new = data_new.fillna(axis=0, method='ffill')
    # 用之前的值填充空值 确保时间序列的连续性 剩下的空值用0填充

    return data_new

def choose_color(num=1):
    import random

    cnames = {#'aliceblue':            '#F0F8FF',
            #    'antiquewhite':         '#FAEBD7',
                'aqua':                 '#00FFFF',
            #    'aquamarine':           '#7FFFD4',
            #    'azure':                '#F0FFFF',
            #    'beige':                '#F5F5DC',
            #    'bisque':               '#FFE4C4',
                'black':                '#000000',
            #    'blanchedalmond':       '#FFEBCD',
                'blue':                 '#0000FF',
                'blueviolet':           '#8A2BE2',
                'brown':                '#A52A2A',
                'burlywood':            '#DEB887',
                'cadetblue':            '#5F9EA0',
            #   'chartreuse':           '#7FFF00',
                'chocolate':            '#D2691E',
            #    'coral':                '#FF7F50',
                'cornflowerblue':       '#6495ED',
            #    'cornsilk':             '#FFF8DC',
                'crimson':              '#DC143C',
            #    'cyan':                 '#00FFFF',
                'darkblue':             '#00008B',
                'darkcyan':             '#008B8B',
                'darkgoldenrod':        '#B8860B',
                'darkgray':             '#A9A9A9',
                'darkgreen':            '#006400',
                'darkkhaki':            '#BDB76B',
                'darkmagenta':          '#8B008B',
                'darkolivegreen':       '#556B2F',
                'darkorange':           '#FF8C00',
                'darkorchid':           '#9932CC',
                'darkred':              '#8B0000',
                'darksalmon':           '#E9967A',
                'darkseagreen':         '#8FBC8F',
                'darkslateblue':        '#483D8B',
                'darkslategray':        '#2F4F4F',
                'darkturquoise':        '#00CED1',
                'darkviolet':           '#9400D3',
                'deeppink':             '#FF1493',
                'deepskyblue':          '#00BFFF',
                'dimgray':              '#696969',
                'dodgerblue':           '#1E90FF',
                'firebrick':            '#B22222',
            #    'floralwhite':          '#FFFAF0',
                'forestgreen':          '#228B22',
            #    'fuchsia':              '#FF00FF',
             #   'gainsboro':            '#DCDCDC',
             #   'ghostwhite':           '#F8F8FF',
            #    'gold':                 '#FFD700',
                'goldenrod':            '#DAA520',
                'gray':                 '#808080',
                'green':                '#008000',
                'greenyellow':          '#ADFF2F',
            #    'honeydew':             '#F0FFF0',
                'hotpink':              '#FF69B4',
                'indianred':            '#CD5C5C',
                'indigo':               '#4B0082',
            #    'ivory':                '#FFFFF0',
                'khaki':                '#F0E68C',
                'lavender':             '#E6E6FA',
            #    'lavenderblush':        '#FFF0F5',
                'lawngreen':            '#7CFC00',
            #    'lemonchiffon':         '#FFFACD',
                'lightblue':            '#ADD8E6',
            #    'lightcoral':           '#F08080',
            #    'lightcyan':            '#E0FFFF',
            #    'lightgoldenrodyellow': '#FAFAD2',
                'lightgreen':           '#90EE90',
             #   'lightgray':            '#D3D3D3',
             #   'lightpink':            '#FFB6C1',
             #   'lightsalmon':          '#FFA07A',
                'lightseagreen':        '#20B2AA',
                'lightskyblue':         '#87CEFA',
                'lightslategray':       '#778899',
                'lightsteelblue':       '#B0C4DE',
            #    'lightyellow':          '#FFFFE0',
                'lime':                 '#00FF00',
                'limegreen':            '#32CD32',
             #   'linen':                '#FAF0E6',
             #   'magenta':              '#FF00FF',
                'maroon':               '#800000',
                'mediumaquamarine':     '#66CDAA',
                'mediumblue':           '#0000CD',
                'mediumorchid':         '#BA55D3',
                'mediumpurple':         '#9370DB',
                'mediumseagreen':       '#3CB371',
                'mediumslateblue':      '#7B68EE',
                'mediumspringgreen':    '#00FA9A',
                'mediumturquoise':      '#48D1CC',
                'mediumvioletred':      '#C71585',
                'midnightblue':         '#191970',
            #    'mintcream':            '#F5FFFA',
            #    'mistyrose':            '#FFE4E1',
            #    'moccasin':             '#FFE4B5',
            #    'navajowhite':          '#FFDEAD',
                'navy':                 '#000080',
            #    'oldlace':              '#FDF5E6',
                'olive':                '#808000',
                'olivedrab':            '#6B8E23',
            #    'orange':               '#FFA500',
            #    'orangered':            '#FF4500',
                'orchid':               '#DA70D6',
            #    'palegoldenrod':        '#EEE8AA',
                'palegreen':            '#98FB98',
            #    'paleturquoise':        '#AFEEEE',
                'palevioletred':        '#DB7093',
            #    'papayawhip':           '#FFEFD5',
            #    'peachpuff':            '#FFDAB9',
                'peru':                 '#CD853F',
            #    'pink':                 '#FFC0CB',
            #    'plum':                 '#DDA0DD',
            #    'powderblue':           '#B0E0E6',
                'purple':               '#800080',
                'red':                  '#FF0000',
                'rosybrown':            '#BC8F8F',
                'royalblue':            '#4169E1',
                'saddlebrown':          '#8B4513',
                'salmon':               '#FA8072',
           #     'sandybrown':           '#FAA460',
                'seagreen':             '#2E8B57',
            #    'seashell':             '#FFF5EE',
                'sienna':               '#A0522D',
            #    'silver':               '#C0C0C0',
                'skyblue':              '#87CEEB',
                'slateblue':            '#6A5ACD',
                'slategray':            '#708090',
            #    'snow':                 '#FFFAFA',
             #   'springgreen':          '#00FF7F',
                'steelblue':            '#4682B4',
                'tan':                  '#D2B48C',
                'teal':                 '#008080',
            #    'thistle':              '#D8BFD8',
                'tomato':               '#FF6347',
                'turquoise':            '#40E0D0',
            #    'violet':               '#EE82EE',
             #   'wheat':                '#F5DEB3',
            #    'white':                '#FFFFFF',
            #    'whitesmoke':           '#F5F5F5',
            #    'yellow':               '#FFFF00',
            #    'yellowgreen':          '#9ACD32'
            }
    if num==1:
        return random.choice(list(cnames.keys()))
    else:
        return [random.choice(list(cnames.keys())) for _ in range(num)]
                    
def get_data_statistics(data):# 获取数据的统计信息
    print('1.数据集共有{}个样本，{}个特征。'.format(data.shape[0], data.shape[1]))
    print('2.数据集基本信息：')
    print(data.describe())
    print('3.数据集包含空值情况统计：')
    print(data.isna().sum())
    print('4.数据集特征和数据类型情况：')
    df = pd.DataFrame({'Data Type':data.dtypes})
    print(df)
    
def bert_service_start(switch=True):# 启动bert服务 并启动服务监控 输入参数为是否开关服务
    from bert_serving.server.helper import get_args_parser
    from bert_serving.server import BertServer
    from bert_serving.client import BertClient

    args = get_args_parser().parse_args(['-model_dir', 'models\chinese_L-12_H-768_A-12'])
    server = BertServer(args)
    if switch:
        server.start()
    else:  
        pass
        # BertServer.shutdown(port=5555)

def bert_service_client(inputs):# 对一个任意长度的字符串进行编码 并返回一个768维的向量 输入为字符串 输出为一维向量 768元素
    from bert_serving.server.helper import get_args_parser
    from bert_serving.server import BertServer
    from bert_serving.client import BertClient

    bc = BertClient()
    # json.dumps(bc.server_status, ensure_ascii=False)
    code = bc.encode([inputs])
    print('The coder of %s is' %inputs)
    print(code)
    return code

def get_technical_indicators(data, last_days=2691, plot=True, save=False):
    # 获取股价技术指标 输入参数为数据集、持续时间和是否绘制图表 输出技术指标 key表示对哪一个指标进行统计分析
    # 7日均线和21日均线
    import stockstats

    dataset_tech = data[['daily_open', 'daily_close', 'daily_high', 'daily_low', 'daily_vol', 'daily_amount']]
    dataset_tech = dataset_tech.rename(columns= lambda x: x.lstrip('daily_')).rename(columns={'vol':'volume', 'ow':'low', 'mount':'amount'})
    
    stock = stockstats.StockDataFrame(dataset_tech)

    technical_keys = ['macd', # moving average convergence divergence. Including signal and histogram. 
                        'macds',# MACD signal line
                        'macdh', # MACD histogram

                        'volume_delta', # volume delta against previous day
                        'volume_-3,2,-1_max', # volume max of three days ago, yesterday and two days later
                        'volume_-3~1_min', # volume min between 3 days ago and tomorrow

                        'kdjk', # KDJ, default to 9 days
                        'kdjd', 
                        'kdjj',
                        'kdjk_3_xu_kdjd_3', # three days KDJK cross up 3 days KDJD
                        
                        'boll', # bolling, including upper band and lower band
                        'boll_ub',
                        'boll_lb', 

                        'open_2_sma', # 2 days simple moving average on open price
                        'open_2_d', # open delta against next 2 day
                        'open_-2_r', # open price change (in percent) between today and the day before yesterday, 'r' stands for rate.
                        'close_10.0_le_5_c', # close price less than 10.0 in 5 days count

                        'cr', # CR indicator, including 5, 10, 20 days moving average
                        'cr-ma1', 
                        'cr-ma2', 
                        'cr-ma3', 
                        'cr-ma2_xu_cr-ma1_20_c', # CR MA2 cross up CR MA1 in 20 days count

                        'rsi_6', # 6 days RSI
                        'rsi_12', # 12 days RSI

                        'wr_10', # 10 days WR
                        'wr_6', # 6 days WR

                        'cci', # CCI, default to 14 days
                        'cci_20', ## 20 days CCI

                        'dma', # DMA, difference of 10 and 50 moving average
                        'pdi', # DMI  +DI, default to 14 days
                        'mdi', # -DI, default to 14 days
                        'dx', # DX, default to 14 days of +DI and -DI
                        'adx', # ADX, 6 days SMA of DX, same as stock['dx_6_ema']
                        'adxr', # ADXR, 6 days SMA of ADX, same as stock['adx_6_ema']

                        'tr', #TR (true range)
                        'atr', # ATR (Average True Range)
                        'trix', # TRIX, default to 12 days
                        'trix_9_sma', # MATRIX is the simple moving average of TRIX

                        'vr', # VR, default to 26 days
                        'vr_6_sma' # MAVR is the simple moving average of VR
                        ]
    for key in technical_keys:
        dataset_tech[key] = pd.DataFrame(stock[key])
    
    dataset_tech['ma7'] = dataset_tech['close'].rolling(window=7).mean()
    dataset_tech['ma21'] = dataset_tech['close'].rolling(window=21).mean()
    dataset_tech['ema'] = dataset_tech['close'].ewm(com=0.5).mean()
    dataset_tech['momentum'] = dataset_tech['close']-1

    if plot:# 绘制技术指标
        plot_dataset = dataset_tech
        plot_dataset = dataset_tech.iloc[-last_days:, :]
        shape_0 = plot_dataset.shape[0]
        x = list(plot_dataset.index)
        colors = choose_color(10)
        
        # 0.股价、成交量，成交额、MA移动平均线
        plt.figure(figsize=(16,10), dpi=150)
        linewidth = 1
        plt.subplot(3, 1, 1)
        plt.title('Close Price and Volume Statistics')
        plt.plot(plot_dataset['close'], label='Close Price')
        plt.plot(plot_dataset['ma7'], label='MA-7', linestyle='--', linewidth=linewidth)
        plt.plot(plot_dataset['ma21'], label='MA-21', linestyle='--', linewidth=linewidth)
        plt.plot(plot_dataset['ema'], label='EMA', linestyle=':', linewidth=linewidth)
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.bar(x, plot_dataset['volume'], label='Volume', width=linewidth)
        plt.bar(x, -plot_dataset['amount'], label='Amount', width=linewidth )
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(plot_dataset['volume_delta'], label='volume delta', linestyle='-', linewidth=linewidth/2, color='k')
        plt.legend()
        if save:
            plt.savefig('project\\feature_engineering\\30_price_amount.png')
        plt.show()

        # 1.MACD 
        plt.figure(figsize=(16,10), dpi=150)
        linewidth = 1
        plt.subplot(2, 1, 1)
        plt.title('Close price and MACD')
        plt.plot(plot_dataset['close'], label='Close Price')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(plot_dataset['macd'], label='macd', linewidth=linewidth)
        plt.plot(plot_dataset['macds'], label='macd signal line', linestyle='--', linewidth=linewidth)
        plt.bar(plot_dataset['macdh'].loc[plot_dataset['macdh']>=0].index, plot_dataset['macdh'].loc[plot_dataset['macdh']>=0], label='macd histgram', width=linewidth, color='r')
        plt.bar(plot_dataset['macdh'].loc[plot_dataset['macdh']<0].index, plot_dataset['macdh'].loc[plot_dataset['macdh']<0], label='macd histgram', width=linewidth, color='g')
        plt.legend()
        if save:
            plt.savefig('project\\feature_engineering\\31_MACD.png')
        plt.show()

        # 2.KDJ and BOLL
        plt.figure(figsize=(16,10), dpi=150)
        linewidth = 1
        plt.subplot(2, 1, 1)
        plt.title('Bolling band and KDJ')
        plt.plot(plot_dataset['close'], label='Close Price')
        plt.plot(plot_dataset['boll'], label='Bolling', linestyle='--', linewidth=linewidth)
        plt.plot(plot_dataset['boll_ub'],color='c', label='Bolling up band', linewidth=linewidth)
        plt.plot(plot_dataset['boll_lb'],color='c', label='Bolling low band', linewidth=linewidth)
        plt.fill_between(x, plot_dataset['boll_ub'], plot_dataset['boll_lb'], alpha=0.35)
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(plot_dataset['kdjk'], label='KDJ-K', linewidth=linewidth)
        plt.plot(plot_dataset['kdjd'], label='KDJ-K', linewidth=linewidth)
        plt.plot(plot_dataset['kdjj'], label='KDJ-K', linewidth=linewidth)
        plt.scatter(plot_dataset['kdjk'].loc[plot_dataset['kdjk_3_xu_kdjd_3']==True].index, plot_dataset['kdjk'].loc[plot_dataset['kdjk_3_xu_kdjd_3']==True], 
                    marker='^', color='r', label='three days KDJK cross up 3 days KDJD')
        plt.legend()
        if save:
            plt.savefig('project\\feature_engineering\\32_boll_kdj.png')
        plt.show()

        # 3.Open price and RSI
        plt.figure(figsize=(16,10), dpi=150)
        linewidth = 1
        plt.subplot(2, 1, 1)
        plt.title('Open price and RSI')
        plt.plot(plot_dataset['open'], label='Open Price')
        plt.bar(x, plot_dataset['open_2_d'], label='open delta against next 2 day')
        plt.plot(plot_dataset['open_-2_r'], label='open price change (in percent) between today and the day before yesterday', linewidth=linewidth)
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(plot_dataset['rsi_12'], label='12 days RSI ', color='c') 
        plt.plot(plot_dataset['rsi_6'], label='6 days RSI', linewidth=linewidth, color='r')  
        plt.legend()
        if save:
            plt.savefig('project\\feature_engineering\\33_open_rsi.png')
        plt.show()

        # 4.CR and WR

        plt.figure(figsize=(16,10), dpi=150)
        linewidth = 1
        plt.subplot(2, 1, 1)
        plt.title('WR and CR in 5/10/20 days')
        plt.plot(plot_dataset['wr_10'], label='10 days WR', linestyle='-', linewidth=linewidth, color='g')
        plt.plot(plot_dataset['wr_6'], label='6 days WR', linestyle='-', linewidth=linewidth, color='r')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.bar(x, plot_dataset['cr'], label='CR indicator', linestyle='--', linewidth=linewidth, color='skyblue')
        plt.plot(plot_dataset['cr-ma1'], label='CR 5 days MA', linestyle='-', linewidth=linewidth)
        plt.plot(plot_dataset['cr-ma2'], label='CR 10 days MA', linestyle='-', linewidth=linewidth)
        plt.plot(plot_dataset['cr-ma3'], label='CR 20 days MA', linestyle='-', linewidth=linewidth)
        plt.legend()
        if save:
            plt.savefig('project\\feature_engineering\\34_cr_ma.png')
        plt.show()

        # 5.CCI TR VR 
        # 
        plt.figure(figsize=(16,10), dpi=150)
        linewidth = 1
        plt.subplot(2, 1, 1)
        plt.title('CCI TR and VR')
        plt.plot(plot_dataset['tr'], label='TR (true range)', linewidth=linewidth)
        plt.plot(plot_dataset['atr'], label='ATR (Average True Range)', linewidth=linewidth)
        plt.plot(plot_dataset['trix'], label='TRIX, default to 12 days', linewidth=linewidth)
        plt.plot(plot_dataset['trix_9_sma'], label='MATRIX is the simple moving average of TRIX', linewidth=linewidth)
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(plot_dataset['cci'], label='CCI, default to 14 days', linestyle='-', linewidth=linewidth, color='r')
        plt.plot(plot_dataset['cci_20'], label='20 days CCI', linestyle='-', linewidth=linewidth, color='g')
        plt.bar(x, plot_dataset['vr'], label='VR, default to 26 days')
        plt.bar(x, -plot_dataset['vr_6_sma'], label='MAVR is the simple moving average of VR')
        plt.legend()
        if save:
            plt.savefig('project\\feature_engineering\\35_cci_tr_vr.png')
        plt.show()

        # 6.DMI
        plt.figure(figsize=(16,10), dpi=150)
        linewidth = 1
        plt.subplot(3, 1, 1)
        plt.title('DMI and DMA')
        plt.bar(x, plot_dataset['pdi'], label='+DI, default to 14 days', color='r')
        plt.bar(x, -plot_dataset['mdi'], label='-DI, default to 14 days', color='g')
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(plot_dataset['dma'], label='DMA, difference of 10 and 50 moving average', linewidth=linewidth, color='k')
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(plot_dataset['dx'], label='DX, default to 14 days of +DI and -DI', linewidth=linewidth)
        plt.plot(plot_dataset['adx'], label='6 days SMA of DX', linewidth=linewidth)
        plt.plot(plot_dataset['adxr'], label='ADXR, 6 days SMA of ADX', linewidth=linewidth)
        plt.legend()
        if save:
            plt.savefig('project\\feature_engineering\\36_close_DMI.png')
        plt.show()

    return dataset_tech   

def get_fft(data, plot=True, save=False):# 傅里叶变换 输入为股价数据集 输出为傅里叶变换的dataframe 
    # 缺失值的处理很重要！ 对数据中的缺失值，fillna 用前面的值代替
    data_FT = data['daily_close'].astype(float)
    technical_data = np.array(data_FT, dtype=float)
    close_fft = fft(technical_data)
    fft_df = pd.DataFrame({'fft_real':close_fft.real, 'fft_imag':close_fft.imag, 'fft':close_fft})
    fft_df['fft_absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['fft_angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    if plot:# 绘制傅里叶变换的图像
        plt.figure(figsize=(14, 7), dpi=100)
        plt.plot(data_FT, label='Close Price')
        fft_list = np.array(fft_df['fft'])
        for num_ in [3, 9, 27, 100]:
            fft_list_m10 = np.copy(fft_list)
            fft_list_m10[num_:-num_]=0
            ifft_list = pd.DataFrame(ifft(fft_list_m10)).set_index(data_FT.index)
            plt.plot(ifft_list, label='Fourier transform with {} components'.format(num_))
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.title('Stock prices & Fourier transforms')
        plt.legend()
        if save:
            plt.savefig('project\\feature_engineering\\40_Fourier_transforms.png')
        plt.show()

        from collections import deque
        items = deque(np.asarray(fft_df['fft_absolute'].tolist()))
        items.rotate(int(np.floor(len(fft_df)/2)))
        # 绘制的频谱数量
        plot_len = 100
        items_plot = list(items)
        items_plot = items_plot[int(len(fft_df)/2-plot_len/2) : int(len(fft_df)/2+plot_len/2)]

        plt.figure(figsize=(10, 7), dpi=100)
        plt.stem(items_plot)
        plt.title(str(plot_len) + ' Components of Fourier transforms ')
        if save:
            plt.savefig('project\\feature_engineering\\41_Fourier_components.png')
        plt.show()

    fft_ = fft_df.set_index(data_FT.index).drop(columns='fft') # 去掉复数的部分

    return fft_

def get_ARIMA(data, plot=True, save=False):# 获取时间序列特征，使用ARIMA 输入为股价数据集
    from statsmodels.tsa.arima_model import ARIMA
    from pandas import DataFrame
    from pandas import datetime
    from pandas import read_csv
    from pandas import datetime
    from statsmodels.tsa.arima_model import ARIMA
    from sklearn.metrics import mean_squared_error

    series = data['daily_close'].astype(float)
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    # print(model_fit.summary())
    summary = model_fit.summary()

    if plot:
        from pandas.plotting import autocorrelation_plot
        plt.figure()
        autocorrelation_plot(series, label='Close price correlations')
        if save:
            plt.savefig('project\\feature_engineering\\50_Close_price_correlations.png')
    else:
        pass

    X = series.values
    size = int(len(X) * 0.9)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)

    if plot:
        plt.figure(figsize=(12, 6), dpi=100)
        plt.plot(test, label='Real')
        plt.plot(predictions, color='red', label='Predicted')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.title('ARIMA model')
        plt.legend()
        if save:
            plt.savefig('project\\feature_engineering\\51_ARIMA_model.png')
        plt.show()
    else:
        pass
    
    return summary, predictions

def get_feather_importance(data, plot=True, save=False):# 获取特征重要性指标 通过xgboost验证
    import xgboost as xgb

    y = data['daily_close'].astype(float)
    # 训练数据中的特征，因为开盘价、收盘价、最高价、最低价都与收盘价y强相关，这些特征会影响其他特征的作用
    # 所以在评估时，将其删除
    # 以下是在测试中重要性大于0.2的特征
    X = data.drop(columns=['cal_date','daily_close','daily_open','daily_low','daily_high','tech_momentum',
                            'tech_ma7', 'tech_ma21', 'tech_ema', 'tech_middle', 'tech_close_-1_s', 'tech_open_2_sma', 'tech_open_2_s',
                            'tech_boll_lb', 'tech_close_10_sma', 'tech_close_10.0_le', 'tech_middle_14_sma',
                            'tech_middle_20_sma', 'tech_close_20_sma', 'tech_close_26_ema','tech_boll','tech_boll_ub',
                            'daily_pre_close','res_qfq_close','res_hfq_close','tech_close_50_sma','tech_atr_14',
                            'tech_atr'
                            ])
    
    train_samples = int(X.shape[0] * 0.9)
    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:]
    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:]
    
    regressor = xgb.XGBRegressor(gamma=0.0,
                                n_estimators=150,
                                base_score=0.7,
                                colsample_bytree=1,
                                learning_rate=0.05,
                                objective='reg:squarederror')
    xgbModel = regressor.fit(X_train,y_train,
                         eval_set = [(X_train, y_train), (X_test, y_test)],
                         verbose=False)
    eval_result = regressor.evals_result()
    training_rounds = range(len(eval_result['validation_0']['rmse']))
    importance = xgbModel.feature_importances_.tolist()
    feature = X_train.columns.tolist()
    feature_importance = pd.DataFrame({'Importance':importance}, index=feature)

    plot_importance = feature_importance.nlargest(40, columns='Importance')
    # 取前40个最重要的特征

    if plot:
        plt.plot(training_rounds,eval_result['validation_0']['rmse'],label='Training Error')
        plt.plot(training_rounds,eval_result['validation_1']['rmse'],label='Validation Error')
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.title('Training Vs Validation Error')
        plt.legend()
        if save:
            plt.savefig('project\\feature_engineering\\60_Training_Vs_Validation_Error.png')
        plt.show()

        fig = plt.figure(figsize=(16,8))
        plt.barh(plot_importance.index, plot_importance['Importance'])
        plt.title('Feature importance of the data')
        if save:
            plt.savefig('project\\feature_engineering\\61_Feature_importance.png')
        plt.show()
    
    return feature_importance


if __name__ == "__main__":
    # 读取数据并去重
    data_csv = pd.read_csv('dataset\\DailyTotal-600690.SH.csv')
    cal_date = pd.to_datetime(data_csv['cal_date'], format='%Y%m%d').to_list()
    data = drop_and_fill(data_csv)
    data = pd.DataFrame(data={col:data[col].tolist() for col in data.columns}, index=cal_date)
    # 获取统计技术指标
    tech_indicator = get_technical_indicators(data, plot=False, save=True, last_days=500)
    tech_indicator = tech_indicator.drop(columns='change')
    # 获取傅里叶变换
    fft = get_fft(data, plot=False, save=True)
    # 获取差分整合移动自平均模型 训练时间较长 但是预测结果居然与实际价格数据几乎一致
#    summary, prediction = get_ARIMA(data, plot=False, save=True)
    
    tech_indicator = tech_indicator.drop(columns=['open', 'close', 'high', 'low', 'volume', 'amount'])
    tech_indicator = tech_indicator.rename(columns= lambda x: 'tech_'+x)
    # 横向叠加数据
    data = pd.concat([data, tech_indicator, fft], axis=1)
    # 获取特征统计特性
    # get_data_statistics(data)
    # 获取特征重要性
    get_feather_importance(data, plot=True, save=True)
    # 保存数据
    now = time.strftime("%Y%m%d_%H%M%S")
    data.to_csv('dataset\\Feature_engineering_'+ now +'.csv')










    