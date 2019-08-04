'''
plot minutes：
	收集了六个月的财经信息，基本来源于新浪财经，新闻数量为80000条左右，经过Bert编码，产生的新闻编码空间是768维。
	对编码空间可视化，试试看。
	对分钟股价数据进行可视化，并发现同一标签数据的规律

'''

import numpy as np 
import pandas as pd 
import sys
import os
from matplotlib import pyplot as plt 

sys.path.append('C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\A-Stock-Prediction-System-with-GAN-and-DRL')

def vectorize_news(data, ):
	'''
	vectorize news：
		对news向量化，处理数据,分类并返回dataframe
	'''
	data['channels_list'] = pd.Series().astype(object)
	codes = []
	for i in data.index:
		# 将channel字符串处理成id列表，ID表示新闻的类别
		channel = data.at[i, 'channels']
		s = '\[\]\{\}\''
		for _ in s:
		    channel = channel.replace(_,'')
		l = channel.split(', ')
		for _ in l:
		    _ = _.strip()
		chn = []
		for _ in l:
		    if(_[:2]=='id'):
		        chn.append(int(_[3:]))
		data['channels_list'].at[i] = chn # 将channels转换为类别标签列表

		# 将字符串code转化为列表
		code = data.at[i, 'code']
		code = [float(x) for x in code.lstrip('\[').rstrip('\]').split(', ')]
		codes.append(code)

	vectorized_news = pd.DataFrame(codes, index=data.index)
	vectorized_news['channels_list'] = data['channels_list']

	return vectorized_news

def plot_news_vector(data, tags=None):
	'''
	plot news vector:
		用散点图绘制不同tags的编码分布情况，从768维中抽取2维
	data:
		向量化之后的data
	tags:
		新闻的标签，列表
	'''
	import random

	x_y = list(range(0,768))
	random.shuffle(x_y) # 从768维中随机选2个不同的维 便于进行可视化
	# plot news 用于下一步可视化，10个标签对应10个Dataframe
	plot_news = []
	# cols name 在dataframe中存储18个随机选出的维度，用于作图（9个子图）
	cols_name = 'abcdefghijklmnopqr'

	for i in range(10):
		plot_news.append(pd.DataFrame(columns=[x for x in cols_name]))

	# 遍历一次数据集，将channel列表中列明的标签对应的数据，放到plot news对应的桶里，这里的三重循环非常不优雅，实在没辙。。。。
	for idx in data.index:
		channel = data['channels_list'].at[idx]
		for ch in channel:
			dim = {}
			for i in range(12):
				dim[cols_name[i]] = data[x_y[i]].at[idx]
			plot_news[ch-1] = plot_news[ch-1].append(pd.Series(dim), ignore_index=True)
	
	# 连画图都是个二重循环
	plt.figure()
	for ax in range(6):
		plt.subplot(2,3,ax+1)
		for i in range(10):
			plt.scatter(plot_news[i][cols_name[2*ax]], plot_news[i][cols_name[2*ax+1]], s=10, alpha=0.3, label=tags[i+1])
			plt.title('Bert code in dimention %d and %d' % (x_y[2*ax], x_y[2*ax+1]))
		plt.legend()
	plt.show()

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

def plot_window_minutes(data):
	'''
	plot window minutes:
		对窗口化的股价数据进行可视化
	'''
	import matplotlib.pyplot as plt
	import mpl_finance as mpf
	from matplotlib.pylab import date2num
	import matplotlib.ticker as ticker
	import arrow
	import time
	import datetime

	data = data.rename(columns={'trade_time':'date', 'vol':'volume'})
	# 将列改名
	data[data['volume'] == 0] = np.nan
	data = data.dropna()
	data.sort_values(by='date', ascending=True, inplace=True)
	
	# columns 的顺序一定是 date, open, close, high, low, volume
	# 这样才符合 candlestick_ochl 绘图要求的数据结构
	# 下面这个是改变列顺序最优雅的方法
	data = data[['date','open','close','high','low','volume']]
	title_date = data.index[0].to_period('D')
	# 获取第一个交易数据的日期作为基准日期
	data['date'] = pd.to_datetime(data['date']).apply(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d %H:%M:%S')[-8:-3])
	# 将date转化为时间、再转化为字符串、再对字符串进行切片，只保留 时:分
	# 每次读取1个交易日的数据，但是实际上，每个交易日从9:30开始到15:00结束，有14:58,14:59，这两个时间是没有交易的。
	data = data[data.index.to_period('D') == title_date]

	# 生成横轴的刻度名字
	date_tickers = data.date.values
	
	x_quotes = [tuple([i]+list(quote[1:])) for i,quote in enumerate(data.values)]
	# _quotes 去除了中午休盘时的时间，使得报价是一个连续的区间

	fig, (ax1, ax2) = plt.subplots(2,sharex=True)

	def format_date(x,pos=None):
	    if x<0 or x>len(date_tickers)-1:
	        return ''
	    return date_tickers[int(x)]
	plt.title("Daily line quotation of %d-%d-%d" %(title_date.year, title_date.month, title_date.day))
	ax1.xaxis.set_major_locator(ticker.MultipleLocator(15))
	# 设置x轴刻度每隔多少点标记一个刻度值
	ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
	# 修改主刻度的文字格式化方式
	ax1.set_ylabel('price')
	ax1.grid(True)
	# fig.autofmt_xdate()
	mpf.candlestick_ochl(ax1,x_quotes,colordown='#53c156', colorup='#ff1717',width=0.05)
	# width用来修改蜡烛的宽度
	bar_quotes = np.array(x_quotes)
	# 将x_quote转换为ndarray进行处理，在bar图中设置涨跌柱颜色
	pos = bar_quotes[:, 1] - bar_quotes[:, 2]<0
	neg = bar_quotes[:, 1] - bar_quotes[:, 2]>0
	ax2.bar(bar_quotes[:, 0][pos], bar_quotes[:, 5][pos]/10000, color='green', width=0.5, align='center')
	ax2.bar(bar_quotes[:, 0][neg], bar_quotes[:, 5][neg]/10000, color='red', width=0.5, align='center')
	ax2.set_ylabel('volume/10 thousand')
	ax2.grid(True)

	plt.show()


def main(steps=[]):
	'''
	step1:
		对新闻编码进行可视化
	step2：
		对分钟股价进行可视化
	'''
	for step in steps:
		if step==1:
			# step 1
			code_filename = 'dataset\\News_with_code-2019-07-28-to-2019-01-30.csv'
			data_csv = pd.read_csv(code_filename, nrows=8000).drop(columns=['content']) # 去掉content列 减少内存占用
			data = data_csv.set_index(pd.to_datetime(data_csv['datetime'])).drop(columns=[x for x in data_csv.columns if x.startswith('Unnamed: ')])
			data = vectorize_news(data)
			# 新闻类别 一共十个
			names = ['宏观','行业','公司','数据','市场','观点','央行','其他','焦点','A股']
			ids = range(1,11)
			tag = {}
			for i in ids:
				tag[i]=names[i-1]
			# 为了正常显示中文字符
			plt.rcParams['font.sans-serif'] = ['SimHei']# 用来正常显示中文标签
			plt.rcParams['axes.unicode_minus'] = False# 用来正常显示负号
			plot_news_vector(data, tag)
		if step==2:
			# step 2
			stock_filename = 'dataset\\MinutesStock-600690.SH-6 months from 2019-07-29 .csv'
			data_csv = pd.read_csv(stock_filename, nrows=6000, usecols=['trade_time','open','close','high','low','vol'])
			data = data_csv.set_index(pd.to_datetime(data_csv['trade_time'])).drop(columns=[x for x in data_csv.columns if x.startswith('Unnamed: ')])
			window_data = window_minutes(data, by='day')
			plot_window_minutes(data)

if __name__ == '__main__':
	main(steps=[2,])