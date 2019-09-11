
'''
概述：
    使用时序卷积网络训练模型，每日股价指标预测未来股价波动，使用膨胀卷积时，模型自动生成深度网络。

TCN：
    应用扩张卷积、层间残差连接的一维时序卷积网络

Class:
    Preprocessor:数据预处理器
    Datamaker:数据切片、划分器
    Timer:计时器
    TCN_Model:卷积模型
    Visualiser:可视化器
    
Author:
    By LongFly
    2019.09
'''

import sys,os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import datetime as dt
import json

sys.path.append(
    'C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\A-Stock-Prediction-System-with-GAN-and-DRL')

from numpy import newaxis
from tcn import TCN
from sklearn.preprocessing import StandardScaler
from keras import Input, Model
from keras.layers import Dense, Flatten
from keras.callbacks import TensorBoard 
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


class Datapreprocessor():
    '''
    数据预处理类：
        直接从原始文件读取数据，并进行数据质量检测、异常数据填充。
    
    config：
        "filename":"dataset\\Feature_engineering_20190624_083438.csv",
		"describe":2,
		"fill_na":"ffill",
		"fill_inf":true,
        "save":false

    method:
        _fill_exceptions:填充缺失值
        _describe_data:描述数据画像，并填充无穷值
        data_preprocessing:预处理，返回处理后的数据集
    '''
    def __init__(self, config,):
        self.preprocess_config = config['preprocess']
        self.data_config = config['data']
        # 从数据文件中读取，设置统一的index
        data_csv = pd.read_csv(self.data_config['filename'])
        cal_date = pd.to_datetime(data_csv['cal_date'], format='%Y%m%d').to_list()
        self.data_csv = pd.DataFrame(data={col:data_csv[col].tolist() for col in data_csv.columns}, index=cal_date)
        cols = list(self.data_csv.columns.astype(str))
        td_col = [col for col in cols if col.endswith('trade_date') or col.endswith('ann_date') or col.endswith('cal_date') or col.endswith('end_date')]
        # 删除指定列
        self.data_csv = self.data_csv.drop(columns=list(td_col))

    def _fill_exceptions(self,):
        # 填充异常值 , 默认使用配置文件中的方法，如果还剩余nan，则使用bfill方法
        self.data = self.data.fillna(method=self.preprocess_config['fill_na'])
        if self.data.isna().sum().sum() > 0:
            self.data = self.data.fillna(method='bfill')

    def _describe_data(self,):
        # 显示数据统计信息和数据画像
        data = self.data
        if self.preprocess_config['describe'] >= 2:
            print('数据集共有{}个样本，{}个特征。'.format(data.shape[0], data.shape[1]))
            print('数据集基本信息：')
            print(data.describe())
        if self.preprocess_config['describe'] >= 1:
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
            print('数据集特征和数据类型情况：')
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
        if self.preprocess_config['describe'] >= 0:
            print('数据集包含空值情况统计：')
            print(data.isna().sum().sum())
            if isinstance(self.preprocess_config['fill_na'], str):
                self._fill_exceptions()
            print('数据集无穷数情况统计：')
            feature_name = np.array(data.columns).astype(str)
            values = np.array(data.values).astype(float)
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
                if self.preprocess_config['fill_inf']:# 对无穷数进行处理 用当前列数据中的 最大值 来填充
                    for i in idx_list:
                        if np.isinf(data.iloc[i]):
                            data.iloc[i] = feature_max[feature_name[i[1]]]
                    print('已将无穷数填充为特征列最大值。')
                    # 再次检验是否有无穷数
                    values = np.array(data.values).astype(float)
                    idx = np.where(np.isinf(values)==True)
                    idx_list = list(zip(idx[0], idx[1]))
            print('数据集中无无穷数。')

        self.data = data


    def data_preprocessing(self,):
        self.data = self.data_csv.drop(
                    columns=[x for x in self.data_csv.columns if x.startswith('Unnamed: ')]).sort_index(ascending=True)
        self._describe_data()
        if self.preprocess_config['save']:
            save_path = os.path.join(self.preprocess_config['data_save_path'], 
                                  'Featured_data_%s.csv' % (dt.datetime.now().strftime('%Y%m%d_%H%M%S')))
            self.data.to_csv(save_path)
        
        return self.data


class Datamaker():
    '''
    数据加载器：
        通过配置文件，将数据从文件中读入，进行数据清洗和异常值检测，然后切片、划分训练集和测试集。

    config:
        "filename": "dataset\\Feature_engineering_20190624_083438.csv",
		"data_index":"daily_trade_date",
        "load_data":false,

		"feature_name":[],
		"feature_dim":489,
		"feature_length": 45,
		
		"target_name":"daily_close",
		"target_dim":1,
		"target_length": 5,
		"target_mode":"precent",

		"fill_null":null,
		"train_test_split": 0.9,
		"train_window_step":1,
		"predict_step":5,
		"normalise_mode": "global"

    method:

        window_data:
        split_data:
        _data_normalise:
    '''
    def __init__(self, config, data=None):
        self.data_config = config['data']
        # 经过预处理的数据直接读入，不需要经过预处理的数据通过文件加载
        if self.data_config['load_data']:
            data_csv = pd.read_csv(self.data_config['filename'])
            self.data = data_csv.set_index(
                pd.to_datetime(data_csv[self.data_config['data_index']]))
        else:
            self.data = data
        # y
        self.y_data = self.data[self.data_config['target_name']].drop(
                            columns=[x for x in self.data.columns if x.startswith('Unnamed: ')])
        
        # x 根据配置中特征名称选择列
        if len(self.data_config['feature_name']) == 0:
            self.x_data = self.data.drop(
                    columns=[x for x in self.data.columns if x.startswith('Unnamed: ')])
        else:
            self.x_data = self.data[self.data_config['feature_name']].drop(
                    columns=[x for x in self.data.columns if x.startswith('Unnamed: ')])
        # 全局标准化
        if self.data_config['normalise_mode'] == "global":
            self.x_data = self._data_normalise(self.x_data)

        if self.data_config['target_mode'] == 'diff':
            # 默认是使用标签的真实值，如果要使用差分值，那么转化delta(t) = y(t) - y(t-1)
            self.y_data = self.y_data - self.y_data.shift(1)
            self.y_data.fillna(0)

        if self.data_config['target_mode'] == 'percent':
            # 要使用百分比，那么需要转化 p(t) = (y(t) - y(t-1))/y(t-1) * 100% ,由于股价波动在-10%~10%之间，
            # 对百分比数据进行标准化，即 p(t) = (y(t) - y(t-1))/y(t-1) * 100% * 10，使其控制在 -1~1之间，
            # 使用tanh激活函数可以确保激活区间有效性
            self.y_data = (self.y_data - self.y_data.shift(1))/self.y_data.shift(1) * 10
            self.y_data.fillna(0)


    def window_data(self,):
        # 对数据序列窗口化，以序列长度切片
        x_windowed_data = []
        y_windowed_data = []
        x_len = self.data_config['feature_length']
        y_len = self.data_config['target_length']
        step = self.data_config['train_window_step']
        assert len(self.x_data) == len(self.y_data)

        for i in range(0, len(self.x_data) - x_len - y_len, step):
            # 划分特征集，以train_window_step为间隔
            x_window = self.x_data[i : i + x_len]
            if self.data_config['normalise_mode'] == 'window':
                x_window = self._data_normalise(x_window)
            y_window = self.y_data[i + x_len : i + x_len + y_len]
            # 特征集是x_len长度的宽表，标签是y_len长度的短序列，这里实现了任意时间长度预测机制
            x_windowed_data.append(x_window)
            y_windowed_data.append(y_window)
        
        return np.array(x_windowed_data), np.array(y_windowed_data)


    def split_data(self, x_windowed_data, y_windowed_data):
        # 默认先将数据进行窗口化，再进行分割训练集和测试集
        assert len(x_windowed_data) == len(x_windowed_data)
        train_window_step = self.data_config['train_window_step']
        predict_step = self.data_config['predict_step']

        i_split = int(len(x_windowed_data) * self.data_config['train_test_split'])
        x_train = x_windowed_data[:i_split:train_window_step]
        y_train = y_windowed_data[:i_split:train_window_step]
        x_test = x_windowed_data[i_split::predict_step]
        y_test = y_windowed_data[i_split::predict_step]
        
        return x_train, y_train, x_test, y_test
    
    def _data_normalise(self, data):
        scalar = StandardScaler()
        scalar.fit(data)
        # 此处出现2个runtimeWarning：
        data = scalar.transform(data)
        return data
        

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


class TCN_Model(Model):
    '''
    时序卷积模型：
        对高维的时序序列卷积
    '''
    def __init__(self, config, **kwargs):
        super(TCN_Model, self).__init__(**kwargs)
        self.model_config = config['model']
        self.training_config = config['training']
        self.data_config = config['data']

    def load_model(self,):
        filepath = self.training_config['load_model_path']
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)
    
    def build_model(self,):
        timer = Timer()
        timer.start()

        batch_size = self.training_config['batch_size']
        input_timestep = self.data_config['feature_length']
        input_dim = self.data_config['feature_dim']
        output_step = self.data_config['target_length']
        output_dim = self.data_config['target_dim']
        hyperparas = self.model_config['hyperparameters']

        i = Input(shape=(input_timestep, input_dim))
        m = TCN(
            nb_filters=hyperparas['nb_filters'], 
            kernel_size=hyperparas['kernel_size'], 
            nb_stacks=hyperparas['nb_stacks'], 
            dilations=hyperparas['dilations'], 
            padding=hyperparas['padding'], 
            use_skip_connections=hyperparas['use_skip_connections'], 
            dropout_rate=hyperparas['dropout_rate'], 
            return_sequences=hyperparas['return_sequences'], 
            activation=hyperparas['activation'], 
            name=hyperparas['name'], 
            kernel_initializer=hyperparas['kernel_initializer'], 
            use_batch_norm=hyperparas['use_batch_norm']
            )(i)
        m = Flatten()(m)
        m = Dense(output_step, activation='linear')(m)
        self.model = Model(inputs=[i], outputs=[m])
        self.model.summary()
        self.model.compile(loss=self.model_config['loss'], optimizer=self.model_config['optimizer'])
        
        if self.model_config['plot_model']:
            from keras.utils import plot_model
            plot_model(self.model, to_file=self.model_config['plot_model_path'])

        print('[Model] Model Compiled')
        timer.stop()

    def train_model(self, x, y):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (self.training_config['epochs'], self.training_config['batch_size']))
        
        save_fname = os.path.join(self.training_config['save_dir'], 
                                  '%s-epochs%s.h5' % (dt.datetime.now().strftime('%Y%m%d-%H%M%S'), 
                                  str(self.training_config['epochs'])))
        callbacks = [
            #EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True),
            TensorBoard(log_dir=self.training_config['tensorboard_dir'])
        ]
        self.history = self.model.fit(
                                      x,
                                      y,
                                      epochs=self.training_config['epochs'],
                                      batch_size=self.training_config['batch_size'],
                                      callbacks=callbacks,
                                      validation_split=self.training_config['validation_split'],
                                      )
        if self.training_config['save']:
            if not os.path.exists(self.training_config['save_dir']): 
                os.makedirs(self.training_config['save_dir'])
            self.model.save(save_fname)
            print('[Saving] Model saved as %s' % save_fname)
        print('[Model] Training Completed.')
        timer.stop()

    def train_generate_model(self,):
        pass

class Visualiser():
    '''
    可视化器：
        对模型结果进行处理，并且可视化

    methed:
        preprocess:对预测结果进行处理，便于进行可视化
        plot_predict:可视化预测结果
        plot_history:可视化训练过程

    config:
        "moving_avg":false,
		"title":"Stock price prediction with tcn"
        "plot_loss":true,
		"plot_prediction":true
    '''
    def __init__(self, config, prediction, actuality, history=None):
        self.visual_config = config['visualise']
        self.data_config = config['data']
        self.prediction = prediction
        self.actuality = actuality
        self.history = history

    def preprocess(self,):
        self.plot_actual = []
        self.plot_predict = []
        pred_len = self.data_config['target_length']
        pred_step = self.data_config['predict_step']

        assert pred_len >= pred_step

        if pred_len == pred_step:
            # 预测结果的间隔就是预测窗口的长度，可以直接将预测结果拼接起来
            self.plot_predict = np.array(self.prediction).reshape(len(self.prediction) * pred_len)
            self.plot_actual = np.array(self.actuality).reshape(len(self.actuality) * pred_len)
        elif self.visual_config['moving_avg']:
            # 预测结果有重叠，如果需要绘制预测结果的移动平均线
            pass
        else:
            # 预测结果有重叠，不需要绘制移动均线，则自动按照窗口间隔筛选预测值，拼接为预测结果
            pass

        if self.data_config['target_mode'] == 'real':
            # 预测标签是真实值，直接可视化
            pass
        elif self.data_config['target_mode'] == 'diff':
            # 预测标签是差分值，需要进行累加计算
            true_value = self.actuality[0][0]
            sum_list = [true_value, ]
            sum_i = true_value
            for i in range(len(self.plot_predict) - 1):
                sum_i = sum_i + self.plot_predict[i+1]
                sum_list.append(sum_i)
            self.plot_predict = [x + true_value for x in sum_list]

        elif self.data_config['target_mode'] == 'percent':
            # 预测标签是百分比，需要进行累乘计算,percent的取值为-1~1，进行了标准化
            true_value = self.actuality[0][0]
            multi_list = [true_value, ]
            multi_i = true_value
            for i in range(len(self.plot_predict) - 1):
                multi_i = multi_i * (1 + self.plot_predict[i+1] / 10)
                multi_list.append(multi_i)
            self.plot_predict = multi_list


    def plot_predictions(self,):
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        fig.title(self.visual_config['title'])
        ax.plot(self.plot_actual, label='True Data')
        plt.plot(self.plot_predict, label='Prediction')
        plt.legend()
        plt.show()

    def plot_history(self,):
        # 绘制训练 & 验证的损失值
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


def main():
    config = json.load(open('bin\\tcn_model\\tcn_config.json', 'r', encoding='utf-8'))

    # 定义一个预处理器
    preprocessor = Datapreprocessor(config=config)
    data = preprocessor.data_preprocessing()

    # 预处理之后的数据集送到maker中制作训练、测试数据
    datamaker = Datamaker(config=config, data=data)
    [x_windowed_data, y_windowed_data] = datamaker.window_data()
    [x_train, y_train, x_test, y_test] = datamaker.split_data(x_windowed_data, y_windowed_data)

    # 加载已有模型或者训练模型
    if config['training']['load']:
        # 如果从已保存的模型中加载
        model = TCN_Model(config=config)
        model.load_model()
    else:
        model = TCN_Model(config=config)
        model.build_model()
        model.train_model(x_train, y_train)

    predictions = []
    for x_test_i in x_test:
        predictions.append(model.predict(x_test_i[newaxis, :, :]))

    # 定义一个可视化实例
    visualiser = Visualiser(config=config, 
                            prediction=predictions, 
                            actuality=y_test, 
                            history=model.history
                            )

    # 绘制预测结果和训练过程示意图
    visualiser.plot_predict()
    visualiser.plot_history()


if __name__=='__main__':
    main()





