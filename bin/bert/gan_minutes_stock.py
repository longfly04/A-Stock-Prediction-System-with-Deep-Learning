'''
minutes stock GAN：
	基于分钟股价数据的生成模型，数据特征是1个交易日内分钟股价数据、成交量数据，
	数据标签是交易日开收盘价和最高最低价以及成交量，也就是1日的特征累加和。

	基于消息向量的生成模型，特征分成两个部分：
	一是休市阶段，对应于下一个交易日开始的n分钟股价
	二是开市阶段，对应于与其接近的一段时间的股价分钟数据。
	股价一个时间点的数据不太具备参考性，将n分钟时间段的股价数据作为标签。
	相当于标签是一个多维的时间序列，比如将涨跌1%以内的都划分成1个区间，涨跌2%以内的超过1%的划分为1个区间
	分钟数据剧烈波动的可能性比较小，所以考虑采用对数函数划分区间。
	划分区间做标签可以减少标签空间的大小，便于模型学习。

	两个生成模型，需要其编码分布尽可能的接近，或者将两者的编码建立映射关系。

	两个模型的在时间上需保持一致。

	或者说，休市阶段的模型和开始阶段的实时模型相互分开。

'''

import numpy as np 
import pandas as pd 
import sys
import os
from matplotlib import pyplot as plt 

sys.path.append('C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\A-Stock-Prediction-System-with-GAN-and-DRL')

def window_minutes(data, by='day'):
	'''
	window minutes:
		对分钟线数据进行窗口化，并打标签
	'''
	selected_cols = ['trade_time', 'open', 'high', 'low', 'close', 'vol']
	data = pd.DataFrame(data[selected_cols])
	windowed_data = []

	if by=='day': # 以交易日为单位切片
		daily_index = data.index.to_period('D').unique()
		for idx in daily_index:
			daily_data = data[data.index.to_period('D') == idx]
			windowed_data.append(daily_data)
			'''
			daily_open = daily_data['open'][0]
			daily_close = daily_data['close'][-1]
			daily_high = daily_data['close'].max()
			daily_low = daily_data['close'].min()
			daily_volume = daily_data['vol'].sum()
			windowed_data = windowed_data.append(pd.DataFrame(data={'date':idx,
												'daily_open':daily_open,
												'daily_close':daily_close,
												'daily_high':daily_high,
												'daily_low':daily_low,
												'daily_volume':daily_volume,
												'daily_data':daily_data}, index=daily_index))
			'''
	if by=='hour': # 以小时为单位切片
		hour_index = data.index.to_period('H').unique()
		for idx in hour_index:
			hour_data = data[data.index.to_period('H') == idx]
			windowed_data.append(hour_data)

	return windowed_data

def main():
	pass


if __name__ == '__main__':
	main()