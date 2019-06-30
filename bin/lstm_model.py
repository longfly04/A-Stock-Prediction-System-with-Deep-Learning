import os
import math
import numpy as np
import pandas as pd
import json
import time
import datetime as dt
from numpy import newaxis

import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\A-Stock-Prediction-System-with-GAN-and-DRL')

class DataLoader():
    """加载数据 并将数据处理给LSTM模型使用
    
    """

    def __init__(self, filename, split, cols):
        # 参数为数据集的路径、训练集和验证集的分割以及数据的列，主要使用的收盘价和成交量
        # 为训练集和测试集数据和数据长度赋值
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        # 定义标签的列和特征的列
        self.y_tag = ['daily_close']
        self.index_tag = ''
        self.ignore_features = ['Unnamed: 0', 'Unnamed: 0.1', 'cal_date', 'daily_trade_date',]
        self.feature_cols = cols
        for col in self.ignore_features or self.y_tag:
            self.feature_cols.remove(col)
        
        # 数据切分 
        self.data_train = dataframe.get(self.feature_cols).values[:i_split]
        self.data_test  = dataframe.get(self.feature_cols).values[i_split:]
        self.y_train = dataframe.get(self.y_tag).values[:i_split]
        self.y_test = dataframe.get(self.y_tag).values[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):
        '''
        数据窗口：以定义好的序列长度为窗口长度，1为步长，从测试集第一个数据开始，截取测试数据
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        y_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])
            y_windows.append(self.y_test[i:i+seq_len])
        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows
        # x特征集为除了最后一列数据之外的数据 y标签为最后一个记录的第一列特征
        x = data_windows
        y = y_windows
        return x,y

    def get_train_data(self, seq_len, normalise):
        '''
        使用next window方法获取数据窗口
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''
        使用生成器的方法产生训练数据 
        Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i
        生成训练窗口数据 x为前49个时间步的数据特征 y为最后一个时间步的第一维数据 这样的划分是否合理呢？？
        '''
        window = self.data_train[i:i+seq_len]
        y_window = self.y_train[i:i+seq_len]
        # 在时间窗口内进行标准化
        # window = self.normalise_windows(window, single_window=True)[0] if normalise else window

        # 这里要注意返回的时间窗口的秩 x应该是一个50个时间步的505维向量 y应该是50个时间步的1维向量
        x = window
        y = y_window[:, 0]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero
        对窗口内数据进行归一化：对以第一行数据为基准，计算窗口内其他数据与第一行数据的比值，再减去1使得基准数为0。
        '''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = StandardScaler.fit_transform(np.arange(50), window)
            normalised_data.append(normalised_window)
        return np.array(normalised_data)

class Timer():
    # 定义一个计时器类 stop方法输出用时
    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))

class Model():
    """建立LSTM模型"""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        timer = Timer()
        timer.start()

        for layer in configs['model']['layers']:
            # 神经层数量
            neurons = layer['neurons'] if 'neurons' in layer else None
            # dropout比例
            dropout_rate = layer['rate'] if 'rate' in layer else None
            # 激活函数类型
            activation = layer['activation'] if 'activation' in layer else None
            # 是返回输出序列中的最后一个输出，还是全部序列。
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            # 输入数据中的时间步数
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            # 输入数据中的维度
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])
        
        # 模型使用了三层LSTM网络，每层100维，最后的LSTM Cell输出到全连接层。
        
        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        # 模型的参数保存下来
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            # 使用earlystop防止过拟合 patience是可以忍耐多少个epoch，monitor所监控的变量没有提升
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
            # 监测验证集误差这个变量，当监测值有改进的时候才保存当前模型，不仅保存权重，也保存模型结构
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
        
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%Y%m%d-%H%M%S'), str(epochs)))
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

    def predict_point_by_point(self, data):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        '''
        这个逐个点预测的方式很好的拟合了股价 但是带有一定的欺骗性 因为每一次预测点仅仅是next day，即使出现误差，在下一个预测中，数据时间窗口因滑动也会忽略掉
        上一次预测的误差结果，这样的话，网络只需要保证预测值不会特别的偏离上一个点即可。这样不是网络拟合的最终目的，我们希望通过股价的多种特征，来预测长时间的
        走势。
        '''
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        #Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        return predicted

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

def main():
    configs = json.load(open('bin\lstm_config.json', 'r', encoding='utf-8'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    # 为了充分利用特征集的所有特征，不再仅仅使用收盘价和成交量
    file_name = 'dataset\\Feature_engineering_20190624_083438.csv'
    data_csv = pd.read_csv(file_name)
    features = np.array(data_csv.columns).astype(str).tolist()
    
    data = DataLoader(
        configs['data']['filename'],
        configs['data']['train_test_split'],
        features
    )

    model = Model()
    model.build_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    
    # in-memory training
    model.train(
        x,
        y,
        epochs = configs['training']['epochs'],
        batch_size = configs['training']['batch_size'],
        save_dir = configs['model']['save_dir']
    )

    '''
    # out-of memory generative training
    steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir']
    )
    '''
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])

#    predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
#    predictions = model.predict_point_by_point(x_test)
#    plot_results(predictions, y_test)


if __name__ == '__main__':
    main()