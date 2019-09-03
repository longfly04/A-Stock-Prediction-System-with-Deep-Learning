import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Input, Model
from keras.layers import Dense
from keras.callbacks import TensorBoard 

from tcn import TCN

'''
概述：
    使用时序卷积网络训练模型，每日股价指标预测未来股价波动，使用膨胀卷积时，模型自动生成深度网络。
TCN：
    该层创建了一个卷积核，该卷积核以 单个空间（或时间）维上的层输入进行卷积， 以生成输出张量。 
    如果 use_bias 为 True， 则会创建一个偏置向量并将其添加到输出中。 最后，如果 activation 不是 None，
    它也会应用于输出。

    当使用该层作为模型第一层时，需要提供 input_shape 参数（整数元组或 None），例如， 
    (10, 128) 表示 10 个 128 维的向量组成的向量序列， (None, 128) 表示 128 维的向量组成的变长序列。

    参数：
        filters: 整数，输出空间的维度 （即卷积中滤波器的输出数量）。
        kernel_size: 一个整数，或者单个整数表示的元组或列表， 指明 1D 卷积窗口的长度。
        strides: 一个整数，或者单个整数表示的元组或列表， 指明卷积的步长。 指定任何 stride 值 != 1 与指定 dilation_rate 值 != 1 两者不兼容。
        padding: "valid", "causal" 或 "same" 之一 (大小写敏感)  "valid" 表示「不填充」。  "same" 表示填充输入以使输出具有与原始输入相同的长度。 "causal" 表示因果（膨胀）卷积， 例如，output[t] 不依赖于 input[t+1:]， 在模型不应违反时间顺序的时间数据建模时非常有用。 详见 WaveNet: A Generative Model for Raw Audio, section 2.1。
        data_format: 字符串,  "channels_last" (默认) 或 "channels_first" 之一。输入的各个维度顺序。  "channels_last" 对应输入尺寸为 (batch, steps, channels) (Keras 中时序数据的默认格式) 而 "channels_first" 对应输入尺寸为 (batch, channels, steps)。
        dilation_rate: 一个整数，或者单个整数表示的元组或列表，指定用于膨胀卷积的膨胀率。 当前，指定任何 dilation_rate 值 != 1 与指定 stride 值 != 1 两者不兼容。
        activation: 要使用的激活函数 (详见 activations)。 如未指定，则不使用激活函数 (即线性激活： a(x) = x)。
        use_bias: 布尔值，该层是否使用偏置向量。
        kernel_initializer: kernel 权值矩阵的初始化器 (详见 initializers)。
        bias_initializer: 偏置向量的初始化器 (详见 initializers)。
        kernel_regularizer: 运用到 kernel 权值矩阵的正则化函数 (详见 regularizer)。
        bias_regularizer: 运用到偏置向量的正则化函数 (详见 regularizer)。
        activity_regularizer: 运用到层输出（它的激活值）的正则化函数 (详见 regularizer)。
        kernel_constraint: 运用到 kernel 权值矩阵的约束函数 (详见 constraints)。
        bias_constraint: 运用到偏置向量的约束函数 (详见 constraints)。
'''
filepath = "C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\keras-tcn\\tasks\\monthly-milk-production-pounds-p.csv"

milk = pd.read_csv(filepath, index_col=0, parse_dates=True)

print(milk.head())

lookback_window = 12  # months.

milk = milk.values  # just keep np array here for simplicity.

x, y = [], []
for i in range(lookback_window, len(milk)):
    x.append(milk[i - lookback_window:i])
    y.append(milk[i])
x = np.array(x)
y = np.array(y)

print(x.shape)
print(y.shape)

i = Input(shape=(lookback_window, 1))
m = TCN()(i)
m = Dense(1, activation='linear')(m)

model = Model(inputs=[i], outputs=[m])

model.summary()

model.compile('adam', 'mae')

print('Train...')
history = model.fit(x, y, 
                    epochs=100, 
                    verbose=1, 
                    validation_split=0.25,
                    callbacks=[TensorBoard(log_dir='log/')]
                    )

p = model.predict(x)

from keras.utils import plot_model
plot_model(model, to_file='model.png')

plt.plot(p)
plt.plot(y)
plt.title('Monthly Milk Production (in pounds)')
plt.legend(['predicted', 'actual'])
plt.show()


# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()