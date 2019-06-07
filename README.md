# A Stock Prediction System with GAN and DRL

# 基于机器学习的股市预测系统


本系统参考了 https://github.com/borisbanushev/stockpredictionai 并尝试在A股市场中预测股价。

先写下实现的思路，以后慢慢改。

数据来源：https://tushare.pro/

For more information ： https://longfly04.github.io/A-Stock-Prediction-System-with-GAN-and-DRL/


## 系统框架

<center><img src='doc\模型框架.jpg' width=1060></img></center>

1.系统需要获取股价信息，包括基本面、技术指标、财经新闻、上市公司财报和金融银行拆借利率等信息。训练模型使用90%训练集，9%交叉验证，1%测试。应用模型使用95%训练，5%验证，生成5%预测数据。


2.个股基本面、技术指标特征进行归一化。消息面特征经过BERT，取300维特征，归一化。


3.向量拼接后分为两流：

stream1 输入VAE，输出结果经PCA，降到80%，再进入到XGBoost，90%训练，9%验证，1%测试。

stream2 直接进到boost的数据，90%训练，9%验证，1%测试。

两个stream的output进入可视化，相当于对比特征工程对数据压缩后，数据意义表达层面有没有问题。

（但是为什么要这么做还是没搞懂。。。）

两个训练结果进本地保存，并附加时间戳。

4.stream1的输出进入GAN模型，Generator是LSTM，400维，Discriminator是CNN，padding是1维的么？主模型每50次迭代输入1次日志，每500次迭代为一次训练。

训练结果进可视化。

训练完成的模型参数进本地保存，并附加时间戳。


5.计算的D-loss和G-loss需要保存，predic-real-loss也需要保存，作为RL的输入。


6.将GAN的超参数集封装为一个对象HyperParameters，交给RL进行调优。


7.第五步的loss之和作为奖励，输入到Q值中，用PPO这种直接采取行动的方式调整HyperParameters。


每50次迭代记录一次日志，并同时将RL的参数保存到本地。

（不知道会调成什么样子。。。）


8.最后，让它跑起来。



## 框架功能

### 数据获取
stock.py:

StockData类：
1.getStockList():获取股票基本信息：股票列表、上市公司基本信息

2.getStockMarket():获取股票行情：代码、交易日、开盘价、收盘价、最高价、最低价、昨收价、涨跌幅、成交量、成交额

3.getDailyIndicator():获取每日指标：股票代码、交易日期、收盘价、换手率、量比、市盈率、市净率、市销率、总股本、流通股本、市值、流通市值。

4.getStockCurrent():个股资金流向:股票代码、交易日期、小单买入量（手）、小单买入金额（万元）、小单卖出量（手）、小单卖出金额（万元）、中单买入量（手）、中单买入金额（万元）、中单卖出量（手）、中单卖出金额（万元）、大单买入量（手）、大单买入金额（万元）、大单卖出量（手）、大单卖出金额（万元）、特大单买入量（手）、特大单买入金额（万元）、特大单卖出量（手）、特大单卖出金额（万元）、净流入量（手）、净流入额（万元）

StockFinance类：获取财务数据 API参考tushare

1.getProfit():获取上市公司财务利润

2.getBalanceSheet():获取上市公司资产负债表

3.getCashflow():获取上市公司现金流量表

4.getForecast():获取业绩预告数据

5.getExpress():获取上市公司业绩快报

6.getDividend():获取分红送股

7.getFinacialIndicator():获取上市公司财务指标数据 

8.getFinacialAudit():获取上市公司定期财务审计意见数据 

9.getFinacialMain():获得上市公司主营业务构成，分地区和产品两种方式

Market类:获取市场参考数据

1.getMoneyflow_HSGT():获取沪股通、深股通、港股通每日资金流向数据，每次最多返回300条记录，总量不限制 

2.getSecuritiesMarginTrading():获取融资融券每日交易汇总数据 

3.getPledgeState():获取股权质押统计数据 

4.getRepurchase():获取上市公司回购股票数据

5.getDesterilization():获取限售股解禁 

6.getBlockTrade():获取大宗交易 

7.getStockHolder():获取上市公司增减持数据，了解重要股东近期及历史上的股份增减变化 

Index类:指数

1.getIndexBasic():获取指数基础信息。

2.getIndexDaily():获取指数每日行情，还可以通过bar接口获取。  

3.getIndexWeight():获取各类指数成分和权重，月度数据  

4.getStockMarketIndex():上证综指指标数据

Futures类：期货

1.getFuturesDaily():获取期货日线

2.getFururesHolding():获取每日成交量

3.getFuturesWSR():获取仓单日报


Interes类：利率

1.getShibor():上海银行间同业拆放利率（Shanghai Interbank Offered Rate，简称Shibor），以位于上海的全国银行间同业拆借中心为技术平台计算、发布并命名，是由信用等级较高的银行组成报价团自主报出的人民币同业拆出利率计算确定的算术平均利率，是单利、无担保、批发性利率。

2.getShiborQuote():Shibor报价数据 

3.getShibor_LPR():贷款基础利率（Loan Prime Rate，简称LPR），是基于报价行自主报出的最优贷款利率计算并发布的贷款市场参考利率。目前，对社会公布1年期贷款基础利率。 

4.getLibor():Libor（London Interbank Offered Rate ），即伦敦同业拆借利率，是指伦敦的第一流银行之间短期资金借贷的利率，是国际金融市场中大多数浮动利率的基础利率。 

5.getHibor():HIBOR (Hongkong InterBank Offered Rate)，是香港银行同行业拆借利率。指香港货币市场上，银行与银行之间的一年期以下的短期资金借贷利率，从伦敦同业拆借利率（LIBOR）变化出来的。 

6.getWenZhouIndex():温州指数 ，即温州民间融资综合利率指数，该指数及时反映民间金融交易活跃度和交易价格。该指数样板数据主要采集于四个方面：由温州市设立的几百家企业测报点，把各自借入的民间资本利率通过各地方金融办不记名申报收集起来；对各小额贷款公司借出的利率进行加权平均；融资性担保公司如典当行在融资过程中的利率，由温州经信委和商务局负责测报；民间借贷服务中心的实时利率。(可选，2012年开放数据)

7.getGuangZhouIndex():广州民间借贷利率(可选)

News类：获取文本
get_news.py：获取新闻，获取主流新闻网站的快讯新闻数据 

1.getNews():新闻资讯数据从2018年10月7日开始有数据 之前没有数据

2.getCompanyPublic():上市公司公告 几乎都是空的

3.getCCTVNews():CCTV 接口限制每分钟100次




## 数据特征工程

### 思路

股价特征作为标签y，指的是：开盘价、收盘价、最高价、最低价，这几个价格的组合。这样定义标签可以么？

1.数据存储为csv文件，由于指标特征类数据需要做归一化，则指标类数据中的文本数据需要做枚举并映射。

2.文本数据需要通过BERT做特征映射，所以文本信息单独使用，数据指标单独使用。

3.特征信息中有一些统计类信息，如季度、月、周信息，而训练数据都是日数据，所以这部分统计数据怎么处理呢？将统计数据按统计周期平均还是单独作为一个指标？

4.将按日数据特征横向拼接，送VAE和PCA训练。

5.训练结果送XGBoost回归，出预测结果比对真实数据，送可视化。

6.降维之后的特征，送GAN模型。

## 技术指标

1.技术指标 technical indicator 需要通过技术指标反映出股价的相关统计信息，并作为股价的参数

2.统计数据 statistical data 股价的统计数据也作为可视化和股价参数

3.通过自编码器之后的向量



### 训练模型




### 评估和调参

TODO

### 可视化

TODO

## 参考资料

感谢支持，赚个积分。

https://tushare.pro/register?reg=266868 