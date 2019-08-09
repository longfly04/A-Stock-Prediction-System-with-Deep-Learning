'''
seq2seq models：
	利用成熟的seq2seq模型，训练模型预测交易日。

'''
import numpy as np 
import pandas as pd 
import sys
import os
import time
import datetime as dt
from matplotlib import pyplot as plt 

import seq2seq
from .models import AttentionSeq2Seq

sys.path.append('C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\A-Stock-Prediction-System-with-GAN-and-DRL')

class Timer():
	def __init__(self):
		self.start_dt = None

	def start(self):
		self.start_dt = dt.datetime.now()

	def stop(self):
		end_dt = dt.datetime.now()
		print('Time taken: %s' % (end_dt - self.start_dt))


class DataLoader():
    """
    参数：
        data：数据集；DataFrame
        cols：数据特征；list
        split：训练集和测试集划分比例；float
        pred_window：预测窗口与数据序列窗口比例；float
        seq_len：序列窗口长度；int
        norm_mode：标准化方式，True：窗口内标准化 False：全局标准化；Bool
        y_tag：标签y的特征名称；String
    方法：
        get_train_data:获取训练数据
        get_test_data：获取测试数据
        _next_window：训练集和测试集迭代生成数据窗口
        normalise_data：对数据进行标准化
    """
    def __init__(self, data, cols, split, pred_window, seq_len, norm_mode, y_tag):
        # 训练集和测试集切分
        i_split = int(len(data) * split)
        # 训练窗口内特征与标签的切分
        self.window_split = int(seq_len * (1 - pred_window))
        # 定义标签的列和特征的列
        self.y_tag = y_tag # 标签
        self.feature_cols = cols
        self.seq_len = seq_len
        self.norm_mode = norm_mode
        self.split = split
        
        # 数据切分 
        self.data_train = data.get(self.feature_cols).values[:i_split]
        self.data_test  = data.get(self.feature_cols).values[i_split:]
        self.y_train = data.get(self.y_tag).values[:i_split]
        self.y_test = data.get(self.y_tag).values[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)

        self.normalised_data = self.normalise_data(data) # 使用全局标准化，确保训练和测试集归属于相同的分布，
        # 注意这个变量和normalise_data方法的名称上的区别 :)
        self.normalised_train = self.normalised_data[:i_split]
        self.normalised_test = self.normalised_data[i_split:]

    def data_split(self, data, split, overlap=False):
        '''
        描述：
            按照split比例，切分数据，主要是先划分数据窗口，再切分训练测试数据，覆盖原有的数据集
        输出：
            data_train 
            data_test  
            y_train 
            y_test
        '''
        y_ = data.get(self.y_tag).values
        x_data = data.get(self.feature_cols).values
        x_data = self.normalise_data(x_data)
        y_seq = []
        x_seq = []
        for i in range(int(len(data)) - self.seq_len):
            x = x_data[i:i+self.window_split, :]
            y = y_[i+self.window_split:i+self.seq_len]
            x_seq.append(x)
            y_seq.append(y)
        i_split = int(len(x_seq) * split)
        x_seq = np.array(x_seq)
        y_seq = np.array(y_seq)
        data_train = x_seq[:i_split, :, :]
        data_test = x_seq[i_split:, :, :]
        y_train = y_seq[:i_split, :]
        y_test = y_seq[i_split:, :]
        if overlap:
            pass
        else:
            i = range(0, int(len(data_test)), self.seq_len - self.window_split)
            data_test = data_test[i, :, :]
            y_test = y_test[i, :]

        return data_train, data_test, y_train, y_test

    def get_train_data(self):
        '''
        训练集：以1个时间步为间隔，x是seq_len*(1-predict_window)长度的序列，y是seq_len*predict长度的序列
        输出：data_x 维度（None，100，489） ，data_y 维度（None，10，1）
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - self.seq_len):
            x, y = self._next_window(i, train_flag=True)
            # 因为训练集和测试集都使用next_window方法，所以用train_flag区分训练集还是测试集
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def get_test_data(self, overlap=False):
        '''
        测试集：以predict_window（10个时间步）为间隔，测试模型预测序列的效果，与训练集使用同样的切片和标准化函数 
        overlap表示测试集是否是重叠模式，即是否要以1个时间步划分，最后还要进行平均
        输出：test_x的维度是（None，100，489） test_y维度是（None，10，1）
        '''
        test_x = []
        test_y = []
        if overlap: # 测试集按照1的步长间隔分割
            for i in range(self.len_test - self.seq_len):
                x, y = self._next_window(i=i, train_flag=False)
                test_x.append(x)
                test_y.append(y)
        else: # 测试集按照预测窗口的间隔分割
            for i in range(int((self.len_test - self.seq_len)/(self.seq_len - self.window_split))):
                x, y = self._next_window(i=i*(self.seq_len - self.window_split), train_flag=False)
                test_x.append(x)
                test_y.append(y)
        return np.array(test_x), np.array(test_y)

    def _next_window(self, i, train_flag=True):
        '''
        生成训练窗口数据，flag用来标识训练集还是测试集
        返回：单个窗口的x和y
        '''
        if train_flag:
            window = self.normalise_data(self.data_train[i:i+self.window_split, :]) if self.norm_mode else self.normalised_train[i:i+self.window_split, :]
            y_window = self.y_train[i+self.window_split:i+self.seq_len]
        else:
            window = self.normalise_data(self.data_test[i:i+self.window_split, :]) if self.norm_mode else self.normalised_test[i:i+self.window_split, :]
            y_window = self.y_test[i+self.window_split:i+self.seq_len]
        # 这里要注意返回的时间窗口的秩 x应该是一个100个时间步的489维矩阵 y应该是10个时间步的1维矩阵
        x = window
        y = y_window
        return x, y

    def normalise_data(self, data):
        '''
        对窗口内数据进行标准化 对每个特征在seq_len个时间步内进行标准化
        '''
        normalised_data = pd.DataFrame(data)
        scalar = StandardScaler()
        scalar.fit(normalised_data)
        normalised_data = scalar.transform(normalised_data)
        return normalised_data

		
class Seq2Seq_Model():
	'''
	seq2seq:
		利用seq2seq模型，对编码的新闻序列建模并
		训练出预测分钟涨跌股价模型
	'''
	def __init__(self, config):
		self.model.compile(loss='mse', optimizer='rmsprop')

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		self.model = load_model(filepath)

	def build_model(self, configs):
		timer = Timer()
		timer.start()

		for layer in configs['model']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None

			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, activation=activation))
			if layer['type'] == 'lstm':
				self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate))

		self.model = AttentionSeq2Seq(input_dim=5, input_length=7, hidden_dim=10, output_length=8, output_dim=20, depth=4)
		self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

		print('[Model] Model Compiled')
		timer.stop()

	def train(self, x, y, epochs, batch_size, save_dir):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			EarlyStopping(monitor='val_loss', patience=2),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
		]
		self.model.fit(
			x,
			y,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks
		)
		self.model.save(save_fname)

		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()

	def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
		]
		self.model.fit_generator(
			data_gen,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			callbacks=callbacks,
			workers=1
		)
		
		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()
	

def generate_window_minutes(data, by='day'):
	'''
	window minutes:
		对分钟线数据进行窗口化，并打标签
	'''
	selected_cols = ['trade_time', 'open', 'high', 'low', 'close', 'vol']
	data = pd.DataFrame(data[selected_cols])
	if by=='day': # 以交易日为单位切片
		daily_index = data.index.to_period('D').unique()
		for idx in daily_index:
			daily_data = data[data.index.to_period('D') == idx]
			x_open = daily_data['open'][0]
			x_close = daily_data['close'][-1]
			x_high = daily_data['high'].max()
			x_low = daily_data['low'].min()
			x_volume = daily_data['vol'].sum()
			y_i = [daily_data['trade_time'][0], x_open, x_close, x_high, x_low, x_volume]
			yield [daily_data,y_i]
			# print('generator is running')

			
	if by=='hour': # 以小时为单位切片
		hour_index = data.index.to_period('H').unique()
		for idx in hour_index:
			hour_data = data[data.index.to_period('H') == idx]
			x_open = hour_data['open'][0]
			x_close = hour_data['close'][-1]
			x_high = hour_data['high'].max()
			x_low = hour_data['low'].min()
			x_volume = hour_data['vol'].sum()
			y_i = [hour_data['trade_time'][0], x_open, x_close, x_high, x_low, x_volume]
			yield [hour_data,y_i]
	

def main():
	stock_filename = 'dataset\\MinutesStock-600690.SH-6 months from 2019-07-29 .csv'
	data_csv = pd.read_csv(stock_filename, nrows=6000, usecols=['trade_time','open','close','high','low','vol'])
	data = data_csv.set_index(pd.to_datetime(data_csv['trade_time'])).drop(columns=[x for x in data_csv.columns if x.startswith('Unnamed: ')])
	
	seq2seq_model(data)
	
	

if __name__ == '__main__':
	main()