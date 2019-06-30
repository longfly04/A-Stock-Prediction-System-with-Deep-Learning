import pandas as pd 
import matplotlib.pyplot as plt

def get_technical_indicators(data, last_days=2691, plot=False):
    # 获取股价技术指标 输入参数为数据集、持续时间和是否绘制图表 输出技术指标 key表示对哪一个指标进行统计分析
    # 7日均线和21日均线
    import stockstats 

    dataset_tech = data[['open', 'close_x_x', 'high', 'low', 'vol', 'amount', 'pct_chg']]
    dataset_tech['close'] = dataset_tech['close_x_x']
    dataset_tech['volume'] = dataset_tech['vol']
    dataset_tech = dataset_tech.drop(columns=['close_x_x', 'vol'])
    stock = stockstats.StockDataFrame(dataset_tech)

    technical_keys = ['macd', # moving average convergence divergence. Including signal and histogram. 
                        'macds',# MACD signal line
                        'macdh', # MACD histogram
                        'volume_delta', # volume delta against previous day
                        'open_2_d', # open delta against next 2 day
                        'open_-2_r', # open price change (in percent) between today and the day before yesterday, 'r' stands for rate.
                        'cr', # CR indicator, including 5, 10, 20 days moving average
                        'cr-ma1', 
                        'cr-ma2', 
                        'cr-ma3', 
                        'volume_-3,2,-1_max', # volume max of three days ago, yesterday and two days later
                        'volume_-3~1_min', # volume min between 3 days ago and tomorrow
                        'kdjk', # KDJ, default to 9 days
                        'kdjd', 
                        'kdjj',
                        'kdjk_3_xu_kdjd_3', # three days KDJK cross up 3 days KDJD
                        'open_2_sma', # 2 days simple moving average on open price
                        'boll', # bolling, including upper band and lower band
                        'boll_ub',
                        'boll_lb', 
                        'close_10.0_le_5_c', # close price less than 10.0 in 5 days count
                        'cr-ma2_xu_cr-ma1_20_c', # CR MA2 cross up CR MA1 in 20 days count
                        'rsi_6', # 6 days RSI
                        'rsi_12', # 12 days RSI
                        'wr_10', # 10 days WR
                        'wr_6', # 6 days WR
                        'cci', # CCI, default to 14 days
                        'cci_20', ## 20 days CCI
                        'tr', #TR (true range)
                        'atr', # ATR (Average True Range)
                        'dma', # DMA, difference of 10 and 50 moving average
                        'pdi', # DMI  +DI, default to 14 days
                        'mdi', # -DI, default to 14 days
                        'dx', # DX, default to 14 days of +DI and -DI
                        'adx', # ADX, 6 days SMA of DX, same as stock['dx_6_ema']
                        'adxr', # ADXR, 6 days SMA of ADX, same as stock['adx_6_ema']
                        'trix', # TRIX, default to 12 days
                        'trix_9_sma', # MATRIX is the simple moving average of TRIX
                        'vr', # VR, default to 26 days
                        'vr_6_sma' # MAVR is the simple moving average of VR
                        ]
    for key in technical_keys:
        dataset_tech[key] = pd.DataFrame(stock[key])

    if plot:# 绘制技术指标
        plot_dataset = dataset_tech
        plot_dataset = dataset_tech.iloc[-last_days:, :]
        shape_0 = plot_dataset.shape[0]
        x = list(plot_dataset.index)
        
        plt.figure(dpi=100)
        # 1.股价、成交量，成交额、MA移动平均线、MACD
        plt.subplot(2, 1, 1)
        plt.plot(plot_dataset['close'], label='Close Price')
        plt.xticks(plot_dataset['volume'])
        plt.plot(plot_dataset[''])


        # 2.KDJ

        # 3.RSI

        # 4.BOLL 

        # 5.WR

        # 6.DMI
        
        # 7.All in one
        
        for key in technical_keys:
            plt.plot(dataset_tech[key], label=key, color=choose_color())
        plt.legend()
        plt.show()

        # Plot first subplot
        
        plt.subplot(2, 1, 1)
        plt.plot(plot_dataset[key+'_'+'tech'], label=key+'_'+'technical indicator', linestyle='-')
        plt.plot(plot_dataset[key+'_'+'ma7'], label=key+'_'+'MA 7', color=colors[0],linestyle='--')
        plt.plot(plot_dataset[key+'_'+'ma21'], label=key+'_'+'MA 21', color=colors[1],linestyle='--')
        plt.plot(plot_dataset[key+'_'+'20sd'], label=key+'_'+'20days Standard Deviation', color=colors[6], linestyle='--')
        plt.plot(plot_dataset[key+'_'+'upper_band'], label=key+'_'+'Upper Band', color=colors[2], linestyle=':')
        plt.plot(plot_dataset[key+'_'+'lower_band'], label=key+'_'+'Lower Band', color=colors[3], linestyle=':')
        plt.fill_between(x, plot_dataset[key+'_'+'lower_band'], plot_dataset[key+'_'+'upper_band'], alpha=0.3)
        plt.title(' Technical indicators in {} days .'.format(last_days))
        plt.ylabel(key+'_'+'technical')
        plt.legend()
        # Plot second subplot
        plt.subplot(2, 1, 2)
        plt.title('MACD')
        plt.plot(plot_dataset[key+'_'+'MACD'], label=key+'_'+'MACD', linestyle='-')
        plt.hlines(plot_dataset[key+'_'+'MACD'].mean(), plot_dataset.index[0], plot_dataset.index[shape_0-1], color=colors[4], linestyles=':', label='MACD mean')
        
        plt.plot(plot_dataset[key+'_'+'ema'], label=key+'_'+'Ema',linestyle='--')
        plt.hlines(plot_dataset[key+'_'+'ema'].mean(), plot_dataset.index[0], plot_dataset.index[shape_0-1], colors=colors[5], linestyles=':', label='Exponential moving average mean')
        plt.legend()
        plt.show()
    # 汇总技术指标

    return dataset_tech   

def drop_and_fill(data):# 去掉重复的列数据 将空值填上数据 输入为股价数据集 输出为去重之后的股价数据集
    data_new = data.T.drop_duplicates(keep='first').T 
    # 去掉重复的数据列 使用转置再转置的方式 非常牛逼
    data_new = data_new.fillna(axis=0, method='ffill')
    return data_new

data_csv = pd.read_csv('dataset\\DailyTotal-600690.SH.csv').drop(columns=['Unnamed: 0', 'ts_code_x_x'])
cal_date = pd.to_datetime(data_csv['cal_date'], format='%Y%m%d').to_list()
data = drop_and_fill(data_csv)
data = pd.DataFrame(data={col:data[col].tolist() for col in data.columns}, index=cal_date)

tech_indicator = get_technical_indicators(data, plot=True, last_days=500)