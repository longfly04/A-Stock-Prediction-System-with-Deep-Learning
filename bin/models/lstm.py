import os
import math
import numpy as np
import pandas as pd
import json
import time
import datetime as dt
from numpy import newaxis
import argparse
import matplotlib.pyplot as plt

from keras.layers import Dense, Activation, Dropout, LSTM, Flatten
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\A-Stock-Prediction-System-with-GAN-and-DRL')

class DataLoader():
    """加载数据 并将数据处理给LSTM模型 
    切分训练和测试集 训练集以1个时间步为间隔 测试集以1个序列长度为间隔（50个时间步）
    """
    def __init__(self, data, split, cols):
        '''
        参数为数据集的路径、训练集和验证集的分割以及数据的列，主要使用的收盘价和成交量
        为训练集和测试集数据和数据长度赋值
        训练特征集为488维数据
        '''
        i_split = int(len(data) * split)
        # 定义标签的列和特征的列
        self.y_tag = ['daily_close'] # 标签
        self.index_tag = [] # 索引
        self.ignore_features = [] # 需要忽略的特征
        self.feature_cols = cols
        for col in self.ignore_features or self.y_tag:
            try:
                self.feature_cols.remove(col)
            except ValueError as e:
                print(col + ' is already removed.' + e)
        
        # 数据切分 
        self.data_train = data.get(self.feature_cols).values[:i_split]
        self.data_test  = data.get(self.feature_cols).values[i_split:]
        self.y_train = data.get(self.y_tag).values[:i_split]
        self.y_test = data.get(self.y_tag).values[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise, overlap=False):
        '''
        测试集：以序列长度（50个时间步）为间隔，测试模型预测序列的效果，与训练集使用同样的切片和标准化函数 overlap表示测试集是否是重叠模式
        输出：test_x的维度是（5，50，488） test_y维度是（5，50，1）
        '''
        test_x = []
        test_y = []
        if overlap:
            for i in range(self.len_test - seq_len):
                x, y = self._next_window(i, seq_len, normalise, train_flag=False)
                test_x.append(x)
                test_y.append(y)
        else:
            for i in range(int(self.len_test/seq_len)):
                x, y = self._next_window(i*seq_len, seq_len, normalise, train_flag=False)
                test_x.append(x)
                test_y.append(y)
        return np.array(test_x), np.array(test_y)

    def get_train_data(self, seq_len, normalise):
        '''
        训练集：以1个时间步为间隔，x和y都是相同长度的序列
        输出：data_x 维度（2364，50，488） ，data_y 维度（2364，50，1）
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise, train_flag=True)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def _next_window(self, i, seq_len, normalise, train_flag=True):
        '''
        生成训练窗口数据，flag用来标识训练集还是测试集
        返回：单个窗口的x和y
        '''
        if train_flag:
            window = self.data_train[i:i+seq_len, :]
            y_window = self.y_train[i:i+seq_len, :]
        else:
            window = self.data_test[i:i+seq_len, :]
            y_window = self.y_test[i:i+seq_len, :]
        # 在时间窗口内进行标准化
        window = self.normalise_windows(window) if normalise else window

        # 这里要注意返回的时间窗口的秩 x应该是一个50个时间步的488维矩阵 y应该是50个时间步的1维矩阵
        x = window
        y = y_window[:, 0]
        return x, y

    def normalise_windows(self, window_data):
        '''
        对窗口内数据进行标准化 对每个特征在50个时间步内进行标准化
        '''
        normalised_data = pd.DataFrame(window_data)
        scalar = StandardScaler()
        scalar.fit(normalised_data)
        normalised_data = scalar.transform(normalised_data)
        return normalised_data

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
            if layer['type'] == 'flatten':
                self.model.add(Flatten())

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])
        self.model.summary()
        # 模型使用了三层LSTM网络，最后的LSTM 将序列输出到全连接层。
        
        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        # 模型的参数保存下来
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%Y%m%d-%H%M%S'), str(epochs)))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            # 使用early stop防止过拟合 patience是可以忍耐多少个epoch，monitor所监控的变量没有提升
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

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        #预测50个时间步的股价 这个预测跟训练的时间步吻合 并且有一定的实际应用价值
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data))):
            curr_slice = data[i, :, :]
            prediction_seqs.append(self.model.predict(curr_slice[newaxis,:,:]))
        prediction = np.array(prediction_seqs).reshape(prediction_len * len(prediction_seqs))
        return prediction

    def predict_sequence_overlap(self, data, window_size):
        # 使用重叠的时间窗口预测股价，并在重叠处使用平均值削弱噪声的影响
        print('[Model] Predicting Sequences Average...')
        # 记录预测结果
        prediction_seqs = []

        for i in range(int(len(data))):
            curr_slice = data[i, :, :]
            predicted = self.model.predict(curr_slice[newaxis,:,:])
            prediction_seqs.append(predicted)
        prediction_seqs = np.array(prediction_seqs).reshape(int(len(data)), window_size)
        prediction_matrix = np.zeros([int(len(data)), int(len(data))+window_size-1]) 
        for i in range(int(len(data))):
            prediction_matrix[i, i:i+window_size] = prediction_seqs[i, :]
        # 对矩阵按列相加
        prediction = prediction_matrix.sum(axis=0)
        for i in range(int(len(prediction))):
            if i >= window_size and i <= len(prediction)-window_size:
                prediction[i] = prediction[i]/window_size
            elif i < window_size:
                prediction[i] = prediction[i]/(i+1)
            elif i > len(prediction)-window_size:
                prediction[i] = prediction[i]/(len(prediction)-i)
        return prediction



        

            

def plot_results(predicted_data, true_data, prediction_len):
    plot_predicted = predicted_data
    plot_true = true_data

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(plot_true, label='True Data')
    plt.plot(plot_predicted, label='Prediction')
    plt.legend()
    plt.show()

def get_data_statistics(data, fill_inf=False):# 获取数据的统计信息 默认不对无穷数进行处理

    indicators = { # 所有的数据指标名称与对应指标类型
        'daily':'日线行情',
        'daily_indicator':'每日指标',
        'moneyflow': '个股资金流向',
        'res_qfq':'前复权行情',
        'res_hfq': '后复权行情',
        'income': '利润表',
        'balancesheet': '资产负债表',
        'cashflow': '现金流量表',
        'forecast': '业绩预告',
        'express': '业绩快报',
        'dividend': '分红送股',
        'financeindicator': '财务指标',
        'HSGTflow': '沪深港通资金流向',
        'margin': '融资融券交易汇总',
        'pledge': '股权质押统计',
        'repurchase': '股票回购',
        'desterilization': '限售股解禁',
        'block': '大宗交易',
        'shibor': '上海银行间同业拆放利率',
        'shiborquote': '上海银行间同业拆放利率报价汇总',
        'shiborLPR': '贷款基础利率',
        'libor': '伦敦同业拆借利率',
        'hibor': '香港银行同行业拆借利率',
        'wen': '温州民间融资综合利率指数',
        'tech': '技术分析',
        'fft': '傅里叶变换',
    } 

    print('1.数据集共有{}个样本，{}个特征。'.format(data.shape[0], data.shape[1]))

    print('2.数据集基本信息：')
    print(data.describe())

    print('3.数据集特征和数据类型情况：')
    datatype = pd.DataFrame({'Feature Names':data.columns, 'Data Types':str(data.dtypes)})
    indicators_idx = {} # 指标索引 找到数据集中不同类别指标的位置
    indicators_count = 0
    for key in indicators:
        indicators_idx[key+'_idx'] = [f.startswith(key+'_') for f in datatype['Feature Names'].tolist()]
        print(indicators[key] + ' 特征数量为：' + str(indicators_idx[key+'_idx'].count(True)) + '个 ')
        indicators_count = indicators_count + indicators_idx[key+'_idx'].count(True)
    # 总的特征数减去daily_indicator 的特征数 因为这个特征在daily中已经包含了
    indicators_count = indicators_count - indicators_idx['daily_indicator_idx'].count(True)
    print('有标记特征数量合计{}个，其他特征{}个'.format(indicators_count, data.shape[1]-indicators_count))

    print('4.数据集包含空值情况统计：')
    print(data.isna().sum().sum())

    print('5.数据集无穷数情况统计：')
    # 获取特征名称的np数组
    feature_name = np.array(data.columns).astype(str)
    # 获取特征取值的np数组
    values = np.array(data.values).astype(float)
    # 获取存在无穷数据的索引
    idx = np.where(np.isinf(values)==True)
    # 将无穷数据的索引zip成坐标
    idx_list = list(zip(idx[0], idx[1]))

    while len(idx_list) > 0:
        print('数据集中共包含{}个无穷数'.format(len(idx_list)))
        print('以下特征出现了无穷数：')
        # 获取出现了无穷数据的特征名称的索引
        feature_idx = set(idx[1])
        # 将每个特征列 除了无穷数据之外的数据的最大值 保存成字典
        feature_max = {}
        for _ in feature_idx:
            print(feature_name[_])
            # 将当前特征列的有效数据取出来 并计算最大值 
            # 在显示特征名称这个循环中顺便处理这个列的无穷数据
            # 首先获得有效数据的索引
            idx_significant = np.where(np.isinf(values[:,_])==False)
            # 特征最大值的字典 key为特征名称 value为最大值
            feature_max[feature_name[_]] = data[feature_name[_]].iloc[idx_significant].max()
        print('无穷数的索引为：')
        print(idx_list)
        if fill_inf:# 对无穷数进行处理 用当前列数据中的 最大值 或者用0 来填充
            for i in idx_list:
                if np.isinf(data.iloc[i]):
                    data.iloc[i] = feature_max[feature_name[i[1]]]
            print('已将无穷数填充为特征列最大值。')
            # 再次检验是否有无穷数
            values = np.array(data.values).astype(float)
            idx = np.where(np.isinf(values)==True)
            idx_list = list(zip(idx[0], idx[1]))

    print('数据集中无无穷数。')
    return data

def read_data(file_name):# 通过文件路径读取文件 并处理数据中的空值和无意义的列
    # 读取已经进行过特征工程的数据
    data_csv = pd.read_csv(file_name)
    data = data_csv.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', ])
    # 将交易日期设置为索引 便于分析和画图
    cal_date = pd.to_datetime(data['cal_date'], format='%Y%m%d').to_list()
    data = pd.DataFrame(data={col:data[col].tolist() for col in data.columns}, index=cal_date)
    # 获取包含有指定名称的列名索引
    cols = list(data.columns.astype(str))
    td_col = [col for col in cols if col.endswith('trade_date') or col.endswith('ann_date') or col.endswith('cal_date') or col.endswith('end_date')]
    # 删除指定列
    data = data.drop(columns=list(td_col))
    # 对nan数据填0
    data = data.fillna(0)

    return data

def parse_args(): # 处理参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--loadfile", default="", help="input the path of the saved model.")
    parser.add_argument("--predict_mode", default="multi", help="input the mode of the prediction? multi/avg")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    configs = json.load(open('bin\models\lstm_config.json', 'r', encoding='utf-8'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    # 为了充分利用特征集的所有特征，不再仅仅使用收盘价和成交量
    file_name = 'dataset\\Feature_engineering_20190624_083438.csv'
    data_csv = read_data(file_name)
    data = get_data_statistics(data_csv, fill_inf=True)
    features = np.array(data.columns).astype(str).tolist()

    # data 为训练数据的实例
    data = DataLoader(
        data,
        configs['data']['train_test_split'],
        features
    )

    # 参数中是否加载已经训练好的模型
    if args.loadfile == "":
        model = Model()
        model.build_model(configs)
        x, y = data.get_train_data(
            seq_len=configs['data']['sequence_length'],
            normalise=configs['data']['normalise']
        )
        # 在内存中进行训练
        model.train(
            x,
            y,
            epochs = configs['training']['epochs'],
            batch_size = configs['training']['batch_size'],
            save_dir = configs['model']['save_dir']
        )
    else:
        model = Model()
        model.load_model(args.loadfile)
    
    # 根据外部参数决定是否使用覆盖模型的测试数据
    overlap = True if args.predict_mode == "avg" else False

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise'],
        overlap=overlap
    )

    # 如果使用覆盖的测试方法，那么测试集会使用滑动窗口进行预测，并在同一个点处进行平均，以消除噪声的影响。
    if overlap:
        predictions = model.predict_sequence_overlap(x_test, configs['data']['sequence_length'])
    else:
        predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    plot_results(predictions, data.y_test, configs['data']['sequence_length'])


if __name__ == '__main__':
    main()