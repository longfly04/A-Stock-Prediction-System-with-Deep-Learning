A Stock Prediction System with Deep Learning 
====
基于深度学习的股市预测系统
----

本系统参考了 https://github.com/borisbanushev/stockpredictionai 并尝试在A股市场中预测股价。

先写下实现的思路，以后慢慢改。

数据来源：https://tushare.pro/

For more information ： https://longfly04.github.io/A-Stock-Prediction-System-with-Deep-Learning/

-----
本系统使用了时间序列模型对股票价格特征进行建模，并利用强化学习算法DQN控制模型训练过程，对超参数空间进行探索，期望找到针对每支股票最合理的模型。

原版本系统目前正在进行架构调整，预计十月底就可以完成，届时整个系统将会打包并开源。

作者同时正在探索一种新的基于强化学习的量化投资系统，以本系统预测结果为依据，结合策略库进行最优化选择，实现股价预测结合指标分析的量化投资系统。

系统框架图如下：

![模型框架](doc/模型框架v2.0.jpg)

目录

[1. 概述](doc\0.概述.md)

[2. 研究现状](doc\1.研究现状.md)

[3. 系统框架和主要功能](doc\2.系统框架和主要功能.md)

[3.1 数据获取](doc\2.1数据获取.md)

[3.2 特征工程](doc\2.2特征工程.md)

[3.3 模型](doc\2.3模型搭建.md)

[3.4 训练和控制](doc\2.4模型训练和调参.md)

[4. 创新点](doc\3.算法改进和创新点.md)

[5. 总结](doc\4.结果分析和总结.md)

-----

感谢支持，赚个积分。

https://tushare.pro/register?reg=266868 



<left><img src='doc\weixin1.jpg' width=180></img></left>
