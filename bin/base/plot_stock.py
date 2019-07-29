'''
概述：
	绘制股价日线、分钟线，绘制成交量直方图
'''

import numpy as np 
import pandas as pd 
import os
import sys

from matplotlib import pyplot as plt 

sys.path.append('C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\A-Stock-Prediction-System-with-GAN-and-DRL')

def plot_minutes_stock(data):
	pass

def plot_daily_stock(data):
	pass

def main():
	filename = "dataset\\MinutesStock-600690.SH-6 months from 2019-07-20 .csv"
	data = pd.read_csv(filename)

if __name__ == '__main__':
	main()
