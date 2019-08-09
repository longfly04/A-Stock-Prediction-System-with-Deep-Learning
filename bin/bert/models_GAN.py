'''
GAN:
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


def wgan_div_model():
	'''
	div:
		以W散度为L约束的WGAN-div，在图像生成任务中表现突出。	
		https://kexue.fm/archives/6139
	'''
	
	from scipy import misc
	import glob
	# import imageio
	from keras.models import Model
	from keras.layers import *
	from keras import backend as K
	from keras.initializers import RandomNormal
	from keras.optimizers import Adam
	
	if not os.path.exists('samples'):
	    os.mkdir('samples')

	imgs = glob.glob('../CelebA-HQ/train/*.png')
	np.random.shuffle(imgs)
	img_dim = 128
	z_dim = 128
	num_layers = int(np.log2(img_dim)) - 3
	max_num_channels = img_dim * 8
	f_size = img_dim // 2**(num_layers + 1)
	batch_size = 64
	'''
	模型的层数是输入维度的对数-3，每层模型的输入维度最大是输入维度的8倍，假样本的维度是
	'''

	def imread(f):
	    x = misc.imread(f, mode='RGB')
	    x = misc.imresize(x, (img_dim, img_dim))
	    return x.astype(np.float32) / 255 * 2 - 1


	def data_generator(batch_size=64):
	    X = []
	    while True:
	        np.random.shuffle(imgs)
	        for f in imgs:
	            X.append(imread(f))
	            if len(X) == batch_size:
	                X = np.array(X)
	                yield X
	                X = []


	# 判别器
	x_in = Input(shape=(img_dim, img_dim, 3))
	x = x_in

	for i in range(num_layers + 1):
	    num_channels = max_num_channels // 2**(num_layers - i)
	    x = Conv2D(num_channels,
	               (5, 5),
	               strides=(2, 2),
	               use_bias=False,
	               padding='same',
	               kernel_initializer=RandomNormal(stddev=0.02))(x)
	    if i > 0:
	        x = BatchNormalization()(x)
	    x = LeakyReLU(0.2)(x)

	x = GlobalAveragePooling2D()(x)
	x = Dense(1, use_bias=False)(x)

	d_model = Model(x_in, x)
	d_model.summary()


	# 生成器
	z_in = Input(shape=(z_dim, ))
	z = z_in

	z = Dense(f_size**2 * max_num_channels,
	          kernel_initializer=RandomNormal(stddev=0.02))(z)
	z = BatchNormalization()(z)
	z = Activation('relu')(z)
	z = Reshape((f_size, f_size, max_num_channels))(z)

	for i in range(num_layers):
	    num_channels = max_num_channels // 2**(i + 1)
	    z = Conv2DTranspose(num_channels,
	                        (5, 5),
	                        strides=(2, 2),
	                        padding='same',
	                        kernel_initializer=RandomNormal(stddev=0.02))(z)
	    z = BatchNormalization()(z)
	    z = Activation('relu')(z)

	z = Conv2DTranspose(3,
	                    (5, 5),
	                    strides=(2, 2),
	                    padding='same',
	                    kernel_initializer=RandomNormal(stddev=0.02))(z)
	z = Activation('tanh')(z)

	g_model = Model(z_in, z)
	g_model.summary()


	# 整合模型（训练判别器）
	x_in = Input(shape=(img_dim, img_dim, 3))
	z_in = Input(shape=(z_dim, ))
	g_model.trainable = False

	x_real = x_in
	x_fake = g_model(z_in)

	x_real_score = d_model(x_real)
	x_fake_score = d_model(x_fake)

	d_train_model = Model([x_in, z_in],
	                      [x_real_score, x_fake_score])

	# 从论文中搜索实验中找到的k=2和p=6的参数组合
	k = 2
	p = 6
	d_loss = K.mean(x_real_score - x_fake_score)

	real_grad = K.gradients(x_real_score, [x_real])[0]
	fake_grad = K.gradients(x_fake_score, [x_fake])[0]

	real_grad_norm = K.sum(real_grad**2, axis=[1, 2, 3])**(p / 2)
	fake_grad_norm = K.sum(fake_grad**2, axis=[1, 2, 3])**(p / 2)
	grad_loss = K.mean(real_grad_norm + fake_grad_norm) * k / 2
	# 定义W散度
	w_dist = K.mean(x_fake_score - x_real_score)

	d_train_model.add_loss(d_loss + grad_loss)
	d_train_model.compile(optimizer=Adam(2e-4, 0.5))
	d_train_model.metrics_names.append('w_dist')
	d_train_model.metrics_tensors.append(w_dist)


	# 整合模型（训练生成器）
	g_model.trainable = True
	d_model.trainable = False

	x_fake = g_model(z_in)
	x_fake_score = d_model(x_fake)

	g_train_model = Model(z_in, x_fake_score)

	g_loss = K.mean(x_fake_score)
	g_train_model.add_loss(g_loss)
	g_train_model.compile(optimizer=Adam(2e-4, 0.5))


	# 检查模型结构
	d_train_model.summary()
	g_train_model.summary()


	# 采样函数
	def sample(path):
	    n = 9
	    figure = np.zeros((img_dim * n, img_dim * n, 3))
	    for i in range(n):
	        for j in range(n):
	            z_sample = np.random.randn(1, z_dim)
	            x_sample = g_model.predict(z_sample)
	            digit = x_sample[0]
	            figure[i * img_dim:(i + 1) * img_dim,
	                   j * img_dim:(j + 1) * img_dim] = digit
	    figure = (figure + 1) / 2 * 255
	    figure = np.round(figure, 0).astype(int)
	    # imageio.imwrite(path, figure)


	iters_per_sample = 100
	total_iter = 1000000
	img_generator = data_generator(batch_size)

	for i in range(total_iter):
	    for j in range(1):
	        z_sample = np.random.randn(batch_size, z_dim)
	        d_loss = d_train_model.train_on_batch(
	            [img_generator.next(), z_sample], None) 
	    for j in range(1):
	        z_sample = np.random.randn(batch_size, z_dim)
	        g_loss = g_train_model.train_on_batch(z_sample, None)
	    if i % 10 == 0:
	        print ('iter: %s, d_loss: %s, g_loss: %s' % (i, d_loss, g_loss))
	    if i % iters_per_sample == 0:
	        sample('samples/test_%s.png' % i)
	        g_train_model.save_weights('./g_train_model.weights')
