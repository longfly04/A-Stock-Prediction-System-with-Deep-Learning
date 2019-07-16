'''
在LSTM模型中，更改数据组织方式，在一个窗口中，前(1 - predict_window) * seq_len作为训练数据，后predict_window * seq_len收盘价作为股价标签，
使得模型在训练阶段就学习预测未来的能力。
在测试集中，使用同样的方法组织数据，以seq_len * predict_window 为间隔提取测试数据集并预测，最后的结果既能够充分覆盖测试集，
又能够利用测试集数据的连续信息，不至于丢失边缘数据的深层趋势。

by LongFly

2019.7

'''
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

class Timer():
    '''
    定义一个计时器类 stop方法输出用时
    '''
    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))

class Model():
    """
    方法：
        load_model:加载已经训练好的模型，filepath=h5文件路径
        buildmodel：根据configs.json文件构造模型，config=json文件路径
        train：训练模型，x，y=特征和标签，epochs=数据集训练迭代次数, batch_size=批的大小, save_dir=模型保存路径
        predict_sequences_multiple:以预测序列长度为间隔进行预测, x_test=测试集, window_size=测试集序列窗口长度, prediction_len=预测结果长度
        predict_sequence_overlap：以1为间隔进行预测，并将同一时间点预测结果进行平均，x_test=测试集, window_size=测试集序列窗口长度
        rectify_predict:修正预测结果，即按照序列长度间隔预测时，起止点结果差别大于涨跌停板限制时，将整个预测预测序列进行修正，prediction=预测结果, window_size=时序窗口
    """

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
            EarlyStopping(monitor='val_loss', patience=1),
            # 使用early stop防止过拟合 patience是可以忍耐多少个epoch，monitor所监控的变量没有提升
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
            # 监测验证集误差这个变量，当监测值有改进的时候才保存当前模型，不仅保存权重，也保存模型结构
        ]
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_split=0.1,
        )
        self.model.save(save_fname)
        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def predict_sequences_multiple(self, x_test, window_size, prediction_len):
        #预测test_len/(seq_len * predict_window)个时间步的股价 这个预测跟训练的时间步吻合 并且有一定的实际应用价值
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(x_test))):
            curr_slice = x_test[i]
            prediction_seqs.append(self.model.predict(curr_slice[newaxis,:,:]))
        prediction = np.array(prediction_seqs).reshape(prediction_len * len(prediction_seqs))
        prediction = self.rectify_predict(prediction, window_size=window_size)
        return prediction

    def predict_sequence_overlap(self, x_test, window_size):
        # 使用重叠的时间窗口预测股价，并在重叠处使用平均值削弱噪声的影响
        print('[Model] Predicting Sequences Average...')
        # 记录预测结果
        prediction_seqs = []

        for i in range(int(len(x_test))):
            curr_slice = x_test[i, :, :]
            predicted = self.model.predict(curr_slice[newaxis,:,:])
            prediction_seqs.append(predicted)
        prediction_seqs = np.array(prediction_seqs).reshape(int(len(x_test)), window_size)
        prediction_matrix = np.zeros([int(len(x_test)), int(len(x_test))+window_size-1]) 
        # 将窗口数据按照实际的时间起始点平移赋值给0矩阵，下一步进行按列求和取平均
        for i in range(int(len(x_test))):
            prediction_matrix[i, i:i+window_size] = prediction_seqs[i, :]
        # 对矩阵按列相加
        prediction = prediction_matrix.sum(axis=0)
        # 取平均值 前window_size和后window_size个数据的分母是变化的 中间都是除以window_size
        for i in range(int(len(prediction))):
            if i >= window_size and i <= len(prediction)-window_size:
                prediction[i] = prediction[i]/window_size
            elif i < window_size:
                prediction[i] = prediction[i]/(i+1)
            elif i > len(prediction)-window_size:
                prediction[i] = prediction[i]/(len(prediction)-i)
        return prediction

    def rectify_predict(self, prediction, window_size):
        # 对窗口预测的股价进行修正 防止突破涨跌停板限制
        for i in range(int(len(prediction))-1):
            if prediction[i+1] > prediction[i] * 1.1 :
                # 如果预测结果突破了涨停板限制 将后面一个窗口的数据同时赋值 相当于在这个窗口衔接处经历了一个涨停板
                prediction[i+1:i+window_size] = prediction[i+1:i+window_size] - (prediction[i+1] - prediction[i] * 1.1)
            elif prediction[i+1] < prediction[i] * 0.9 :
                # 如果预测结果突破了跌停板限制 将后面一个窗口的数据同时赋值 相当于在这个窗口衔接处经历了一个跌停板
                prediction[i+1:i+window_size] = prediction[i+1:i+window_size] + (prediction[i] * 0.9 - prediction[i+1])
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

def parse_args(): # 处理参数 分别是加载已经保存好的模型的路径 以及预测值的模式：滑动窗口还是多窗口
    parser = argparse.ArgumentParser()
    parser.add_argument("--loadfile", default="", help="input the path of the saved model.")
    parser.add_argument("--predict_mode", default="multi", help="input the mode of the prediction? multi/avg")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    configs = json.load(open('bin\\models\\lstm_modified_config.json', 'r', encoding='utf-8'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    # 为了充分利用特征集的所有特征，不再仅仅使用收盘价和成交量
    file_name = 'dataset\\Feature_engineering_20190624_083438.csv'
    data_csv = read_data(file_name)
    data_raw = get_data_statistics(data_csv, fill_inf=True)
    features = np.array(data_raw.columns).astype(str).tolist()

    # data 为训练数据的实例
    data = DataLoader(
        data=data_raw,
        cols=features,
        split=configs['data']['train_test_split'],
        pred_window=configs['data']['predict_window'],
        seq_len=configs['data']['sequence_length'],
        norm_mode=configs['data']['normalise_mode'],
        y_tag=configs['data']['y_tag']
    )
    x, x_test, y, y_test = data.data_split(data_raw, data.split)

    # 参数中是否加载已经训练好的模型
    if args.loadfile == "":
        model = Model()
        model.build_model(configs)
        # x, y = data.get_train_data()
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
    
    # 根据外部参数决定是否使用滑动窗口的测试数据
    overlap = True if args.predict_mode == "avg" else False
    # x_test, y_test = data.get_test_data(overlap=overlap)

    # 如果使用覆盖的测试方法，那么测试集会使用滑动窗口进行预测，并在同一个点处进行平均，以消除噪声的影响。
    if overlap:
        predictions = model.predict_sequence_overlap(x_test, configs['data']['sequence_length'])
    else:
        predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], int(configs['data']['sequence_length']*configs['data']['predict_window']))
    plot_results(predictions, data.y_test, configs['data']['sequence_length'])

if __name__ == '__main__':
    main()