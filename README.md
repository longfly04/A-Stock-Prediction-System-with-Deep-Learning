# A Stock Prediction System with GAN and DRL

# 基于机器学习的股市预测系统


本系统参考了 https://github.com/borisbanushev/stockpredictionai 并尝试在A股市场中预测股价。

先写下实现的思路，以后慢慢改。

数据来源：https://tushare.pro/


## 系统框架

<center><img src='模型框架.jpg' width=1060></img></center>

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
get_stock_data.py:获取股票数据

1.GetStockList():获取股票基本信息：股票列表、上市公司基本信息

2.GetStockMarket():获取股票行情：代码、交易日、开盘价、收盘价、最高价、最低价、昨收价、涨跌幅、成交量、成交额

3.GetDailyIndicator():获取每日指标：股票代码、交易日期、收盘价、换手率、量比、市盈率、市净率、市销率、总股本、流通股本、市值、流通市值。

4.GetStockCurrent():个股资金流向:股票代码、交易日期、小单买入量（手）、小单买入金额（万元）、小单卖出量（手）、小单卖出金额（万元）、中单买入量（手）、中单买入金额（万元）、中单卖出量（手）、中单卖出金额（万元）、大单买入量（手）、大单买入金额（万元）、大单卖出量（手）、大单卖出金额（万元）、特大单买入量（手）、特大单买入金额（万元）、特大单卖出量（手）、特大单卖出金额（万元）、净流入量（手）、净流入额（万元）

get_stock_finance.py：获取财务数据 API参考tushare

1.GetProfit():获取上市公司财务利润 https://tushare.pro/document/2?doc_id=33

2.GetBalanceSheet():获取上市公司资产负债表 https://tushare.pro/document/2?doc_id=36

3.GetCashflow():获取上市公司现金流量表 https://tushare.pro/document/2?doc_id=44

4.GetForecast():获取业绩预告数据 https://tushare.pro/document/2?doc_id=45

5.GetExpress():获取上市公司业绩快报 https://tushare.pro/document/2?doc_id=46

6.GetDividend():获取分红送股：https://tushare.pro/document/2?doc_id=103

7.GetFinacialIndicator():获取上市公司财务指标数据 https://tushare.pro/document/2?doc_id=79

8.GetFinacialAudit():获取上市公司定期财务审计意见数据 https://tushare.pro/document/2?doc_id=80

9.GetFinacialMain():获得上市公司主营业务构成，分地区和产品两种方式 https://tushare.pro/document/2?doc_id=81

get_market_reference.py:获取市场参考数据

1.GetMoneyflow_HSGT():获取沪股通、深股通、港股通每日资金流向数据，每次最多返回300条记录，总量不限制 https://tushare.pro/document/2?doc_id=47

2.GetSecuritiesMarginTrading():获取融资融券每日交易汇总数据 https://tushare.pro/document/2?doc_id=58

3.GetPledgeState():获取股权质押统计数据 https://tushare.pro/document/2?doc_id=110

4.GetRepurchase():获取上市公司回购股票数据 https://tushare.pro/document/2?doc_id=124

5.GetDesterilization():获取限售股解禁 https://tushare.pro/document/2?doc_id=160

6.GetBlockTrade():获取大宗交易 https://tushare.pro/document/2?doc_id=161

7.GetStockHolder():获取上市公司增减持数据，了解重要股东近期及历史上的股份增减变化 https://tushare.pro/document/2?doc_id=175

get_index.py:获取指数

1.GetIndexBasic():获取指数基础信息。 https://tushare.pro/document/2?doc_id=94

2.GetIndexDaily():获取指数每日行情，还可以通过bar接口获取。  https://tushare.pro/document/2?doc_id=95

3.GetIndexWeight():获取各类指数成分和权重，月度数据  https://tushare.pro/document/2?doc_id=96

4.GetStockMarketIndex():上证综指指标数据 https://tushare.pro/document/2?doc_id=128

get_interest_indicator.py

1.GetShibor():上海银行间同业拆放利率（Shanghai Interbank Offered Rate，简称Shibor），以位于上海的全国银行间同业拆借中心为技术平台计算、发布并命名，是由信用等级较高的银行组成报价团自主报出的人民币同业拆出利率计算确定的算术平均利率，是单利、无担保、批发性利率。https://tushare.pro/document/2?doc_id=149

2.GetShiborQuote():Shibor报价数据 https://tushare.pro/document/2?doc_id=150

3.GetShibor_LPR():贷款基础利率（Loan Prime Rate，简称LPR），是基于报价行自主报出的最优贷款利率计算并发布的贷款市场参考利率。目前，对社会公布1年期贷款基础利率。 https://tushare.pro/document/2?doc_id=151

4.GetLibor():Libor（London Interbank Offered Rate ），即伦敦同业拆借利率，是指伦敦的第一流银行之间短期资金借贷的利率，是国际金融市场中大多数浮动利率的基础利率。 https://tushare.pro/document/2?doc_id=152

5.GetHibor():HIBOR (Hongkong InterBank Offered Rate)，是香港银行同行业拆借利率。指香港货币市场上，银行与银行之间的一年期以下的短期资金借贷利率，从伦敦同业拆借利率（LIBOR）变化出来的。 https://tushare.pro/document/2?doc_id=153

6.GetWenZhouIndex():温州指数 ，即温州民间融资综合利率指数，该指数及时反映民间金融交易活跃度和交易价格。该指数样板数据主要采集于四个方面：由温州市设立的几百家企业测报点，把各自借入的民间资本利率通过各地方金融办不记名申报收集起来；对各小额贷款公司借出的利率进行加权平均；融资性担保公司如典当行在融资过程中的利率，由温州经信委和商务局负责测报；民间借贷服务中心的实时利率。(可选，2012年开放数据)

7.GetGuangZhouIndex():广州民间借贷利率(可选)

get_news.py：获取新闻，获取主流新闻网站的快讯新闻数据 https://tushare.pro/document/2?doc_id=143

get_cctv_news.py:获取新闻联播文字稿数据，数据开始于2006年6月，超过12年历史 https://tushare.pro/document/2?doc_id=154

get_company_public.py:获取上市公司公告数据及原文文本，数据从2000年开始，内容很大，请注意数据调取节奏。https://tushare.pro/document/2?doc_id=176


## 数据特征工程

股价特征作为标签y，指的是：开盘价、收盘价、最高价、最低价，这几个价格的组合。这样定义标签可以么？

1.数据存储为csv文件，由于指标特征类数据需要做归一化，则指标类数据中的文本数据需要做枚举并映射。

2.文本数据需要通过BERT做特征映射，所以文本信息单独使用，数据指标单独使用。

3.特征信息中有一些统计类信息，如季度、月、周信息，而训练数据都是日数据，所以这部分统计数据怎么处理呢？将统计数据按统计周期平均还是单独作为一个指标？

4.将按日数据特征横向拼接，送VAE和PCA训练。

5.训练结果送XGBoost回归，出预测结果比对真实数据，送可视化。

6.降维之后的特征，送GAN模型。



### 训练模型

TODO

### 评估和调参

TODO

### 可视化

TODO

## 参考资料

感谢支持，赚个积分。

https://tushare.pro/register?reg=266868 