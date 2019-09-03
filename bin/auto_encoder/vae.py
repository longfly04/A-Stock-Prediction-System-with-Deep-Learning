import numpy as np 
import pandas as pd 
import datetime,time
import math
import matplotlib.pyplot as plt

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

def pca(data):# 降维 
    import sklearn.preprocessing as pp
    from sklearn.decomposition import PCA

    # sklearn的一系列标准化 归一化函数 想用哪里点哪里
    # 标准化 均值为0 方差为1
    std_scalar = pp.StandardScaler()
    data_std = std_scalar.fit_transform(data)
    # 规范化 默认使用L2正则
    normer = pp.Normalizer()
    data_norm = normer.fit_transform(data)
    # 最大最小归一化 所有值都缩放到（0，1）
    minmax_scalar = pp.MinMaxScaler()
    data_minmax = minmax_scalar.fit_transform(data)
    # PCA类 用MLE算法根据特征的方差分布情况自己去选择一定数量的主成分特征来降维
    pca = PCA(n_components=200)
    data_pca = pca.fit_transform(data_std)
    print(pca.explained_variance_ratio_)
    return data_pca

def vae(X_data, y): #通过VAE提取分布信息
    # 对时间序列数据使用自编码器 没有充分的挖掘时间序列的特性 模型极容易发散
    from scipy.stats import norm
    from keras.layers import Input, Dense, Lambda
    from keras.models import Model
    from keras import backend as K
    from keras.datasets import mnist
    from keras.utils import np_utils
    import keras

    batch_size = 100
    original_dim = 200
    latent_dim = 128 # 隐变量 需要确保模型有足够的容量
    intermediate_dim = 200
    epochs = 50

    X = pd.DataFrame(X_data)
    train_samples = int(X.shape[0] * 0.90)
    # 训练集，但是标签数据没有进入模型
    X_train = X.iloc[:train_samples]
    y_train = y.iloc[:train_samples]
    # 验证集，是验证还是测试呢？
    X_test = X.iloc[train_samples:]
    y_test = y.iloc[train_samples:]

    #LossHistory类，保存loss和acc 并且plot
    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = {'batch':[], 'epoch':[]}
            self.accuracy = {'batch':[], 'epoch':[]}
            self.val_loss = {'batch':[], 'epoch':[]}
            self.val_acc = {'batch':[], 'epoch':[]}

        def on_batch_end(self, batch, logs={}):
            self.losses['batch'].append(logs.get('loss'))
            self.accuracy['batch'].append(logs.get('acc'))
            self.val_loss['batch'].append(logs.get('val_loss'))
            self.val_acc['batch'].append(logs.get('val_acc'))

        def on_epoch_end(self, batch, logs={}):
            self.losses['epoch'].append(logs.get('loss'))
            self.accuracy['epoch'].append(logs.get('acc'))
            self.val_loss['epoch'].append(logs.get('val_loss'))
            self.val_acc['epoch'].append(logs.get('val_acc'))

        def loss_plot(self, loss_type):
            iters = range(len(self.losses[loss_type]))
            plt.figure()
            # acc
            plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
            # loss
            plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
            if loss_type == 'epoch':
                # val_acc
                plt.scatter(iters, self.val_acc[loss_type], c='b', label='val acc')
                # val_loss
                plt.scatter(iters, self.val_loss[loss_type], c='k', label='val loss')
            plt.grid(True)
            plt.xlabel(loss_type)
            plt.ylabel('acc-loss')
            plt.legend(loc="upper right")
            plt.show()

    # 构建模型输入，时间序列数据应该从二维展平成一维，或者通过其他方式将时序体现出来
    x = Input(shape=(original_dim,))
    # 输入的隐层，由于是时间序列，考虑应该替换成堆叠的LSTM，并尝试捕获全局信息
    # TODO
    h = Dense(intermediate_dim, activation='relu')(x)

    # 算p(Z|X)的均值和方差，
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    # 重参数
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # 重参数层，相当于给输入加入噪声
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # 解码层，也就是生成器部分
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # 训练历史记录
    history = LossHistory()
    # 建立模型
    vae = Model(x, x_decoded_mean)

    # xent_loss是重构loss，kl_loss是KL loss
    xent_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    # add_loss是新增的方法，用于更灵活地添加各种loss
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()

    vae.fit(X_train,
            # y=y_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, None),
            callbacks=[history])

    # 绘制训练误差 精度
    history.loss_plot('epoch')

    # 构建encoder，然后观察各个数字在隐空间的分布
    encoder = Model(x, z_mean)

    x_test_encoded = encoder.predict(X_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.show()

    # 构建生成器
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

def get_data_statistics(data, fill_inf=False):# 获取数据的统计信息 默认不对无穷数进行处理
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

if __name__ == "__main__":
    # 读取已经进行过特征工程的数据
    data_csv = pd.read_csv('dataset\Feature_engineering_20190624_083438.csv')
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
    data = get_data_statistics(data, fill_inf=True)

    # 降维
    data_pca = pca(data)
    
    # 训练自编码器
    y = data['daily_close'].astype(float)
    vae(data_pca, y)

    



    





    