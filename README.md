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



## 数据获取

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


Interest类：利率

1.getShibor():上海银行间同业拆放利率（Shanghai Interbank Offered Rate，简称Shibor），以位于上海的全国银行间同业拆借中心为技术平台计算、发布并命名，是由信用等级较高的银行组成报价团自主报出的人民币同业拆出利率计算确定的算术平均利率，是单利、无担保、批发性利率。

2.getShiborQuote():Shibor报价数据 

3.getShibor_LPR():贷款基础利率（Loan Prime Rate，简称LPR），是基于报价行自主报出的最优贷款利率计算并发布的贷款市场参考利率。目前，对社会公布1年期贷款基础利率。 

4.getLibor():Libor（London Interbank Offered Rate ），即伦敦同业拆借利率，是指伦敦的第一流银行之间短期资金借贷的利率，是国际金融市场中大多数浮动利率的基础利率。 

5.getHibor():HIBOR (Hongkong InterBank Offered Rate)，是香港银行同行业拆借利率。指香港货币市场上，银行与银行之间的一年期以下的短期资金借贷利率，从伦敦同业拆借利率（LIBOR）变化出来的。 

6.getWenZhouIndex():温州指数 ，即温州民间融资综合利率指数，该指数及时反映民间金融交易活跃度和交易价格。该指数样板数据主要采集于四个方面：由温州市设立的几百家企业测报点，把各自借入的民间资本利率通过各地方金融办不记名申报收集起来；对各小额贷款公司借出的利率进行加权平均；融资性担保公司如典当行在融资过程中的利率，由温州经信委和商务局负责测报；民间借贷服务中心的实时利率。(可选，2012年开放数据)

7.getGuangZhouIndex():广州民间借贷利率(可选)

News类：获取经济和金融新闻

get_news.py：获取新闻，获取主流新闻网站的快讯新闻数据 

1.getNews():新闻资讯数据从2018年10月7日开始有数据 之前没有数据

2.getCompanyPublic():上市公司公告 几乎都是空的

3.getCCTVNews():CCTV 接口限制每分钟100次



## 数据特征工程

### 概述

特征工程的好坏决定了模型的上限，调参只是逼近这个上限而已。

这个阶段，我计划将收集的数据以每日股价收盘价作为标签，获取技术指标，并作为数据特征。传统技术指标分析主要是针对股价，包括了7日均线，21日均线，MACD平滑异同移动均线，BOLLING布林线等，实际上，股价的波动在影响因素上，取决于非常多的客观交易行为，传统技术指标只是对股价的时间统计特征进行体现，并没有将市场的细节展现出来。本系统我将尝试增加自定义的技术指标，试图追踪交易量、大单买入卖出、换手率等指标的统计特征，并对以上技术指标进行频率分析，以期待获得更多市场行为信息。

基本面分析主要参考上市公司财务数据和公告，但是由于公告涉及到文本分析且数据量较少，这部分仅保留财务数据。另外，宏观经济数据也作为基本面分析的一部分。

特征组合，删除重复特征以及空值处理。由于特征存在大量空值，所以要分情况对空值进行处理，主要的方式是以空值前一日的value填充。即便如此，股价数据集仍然是一个庞大的稀疏矩阵。

在股价趋势预测任务中，暂时不适用于将每日财经新闻进行情感分析并作为事件特征，因为每日新闻数据量较大，对于个股的影响需要提取相关系数矩阵，计算量惊人。考虑到新闻的综合影响会直接体现在交易数据中，所以在技术指标中增加交易数据的统计特征，平衡特征的数量。



### 基本面分析

概括起来，基本分析主要包括以下三个方面内容：

（1）宏观经济分析。研究经济政策（货币政策、财政政策、税收政策、产业政策等等）、经济指标（国内生产总值、失业率、通胀率、利率、汇率等等）对股票市场的影响。

（2）行业分析。分析产业前景、区域经济发展对上市公司的影响。

（3）公司分析。具体分析上市公司行业地位、市场前景、财务状况。

这里，基本面分析参考的指标主要包括：

|指标|格式|含义|
| --| -- | -- |
|basic_eps|float|基本每股收益
|diluted_eps|float|稀释每股收益
|total_revenue|float|营业总收入
|revenue|float|营业收入
|int_income|float|利息收入
total_share|float|期末总股本
cap_rese|float|资本公积金
undistr_porfit|float|未分配利润
surplus_rese|float|盈余公积金
special_rese|float|专项储备
money_cap|float|货币资金
trad_asset|float|交易性金融资产
notes_receiv|float|应收票据
accounts_receiv|float|应收账款
oth_receiv|float|其他应收款
prepayment|float|预付款项
div_receiv|float|应收股利
int_receiv|float|应收利息
inventories|float|存货
amor_exp|float|长期待摊费用
nca_within_1y|float|一年内到期的非流动资产
sett_rsrv|float|结算备付金
loanto_oth_bank_fi|float|拆出资金
premium_receiv|float|应收保费
reinsur_receiv|float|应收分保账款
reinsur_res_receiv|float|应收分保合同准备金
pur_resale_fa|float|买入返售金融资产
oth_cur_assets|float|其他流动资产
total_cur_assets|float|流动资产合计
fa_avail_for_sale|float|可供出售金融资产
htm_invest|float|持有至到期投资
lt_eqt_invest|float|长期股权投资
invest_real_estate|float|投资性房地产
time_deposits|float|定期存款
oth_assets|float|其他资产
lt_rec|float|长期应收款
fix_assets|float|固定资产
cip|float|在建工程
const_materials|float|工程物资
fixed_assets_disp|float|固定资产清理
produc_bio_assets|float|生产性生物资产
oil_and_gas_assets|float|油气资产
intan_assets|float|无形资产
r_and_d|float|研发支出
goodwill|float|商誉
lt_amor_exp|float|长期待摊费用
defer_tax_assets|float|递延所得税资产
decr_in_disbur|float|发放贷款及垫款
oth_nca|float|其他非流动资产
total_nca|float|非流动资产合计
cash_reser_cb|float|现金及存放中央银行款项
depos_in_oth_bfi|float|存放同业和其它金融机构款项
prec_metals|float|贵金属
deriv_assets|float|衍生金融资产
rr_reins_une_prem|float|应收分保未到期责任准备金
rr_reins_outstd_cla|float|应收分保未决赔款准备金
rr_reins_lins_liab|float|应收分保寿险责任准备金
rr_reins_lthins_liab|float|应收分保长期健康险责任准备金
refund_depos|float|存出保证金
ph_pledge_loans|float|保户质押贷款
refund_cap_depos|float|存出资本保证金
indep_acct_assets|float|独立账户资产
client_depos|float|其中：客户资金存款
client_prov|float|其中：客户备付金
transac_seat_fee|float|其中:交易席位费
invest_as_receiv|float|应收款项类投资
total_assets|float|资产总计
lt_borr|float|长期借款
st_borr|float|短期借款
cb_borr|float|向中央银行借款
depos_ib_deposits|float|吸收存款及同业存放
loan_oth_bank|float|拆入资金
trading_fl|float|交易性金融负债
notes_payable|float|应付票据
acct_payable|float|应付账款
adv_receipts|float|预收款项
sold_for_repur_fa|float|卖出回购金融资产款
comm_payable|float|应付手续费及佣金
payroll_payable|float|应付职工薪酬
taxes_payable|float|应交税费
int_payable|float|应付利息
div_payable|float|应付股利
oth_payable|float|其他应付款
acc_exp|float|预提费用
deferred_inc|float|递延收益
st_bonds_payable|float|应付短期债券
payable_to_reinsurer|float|应付分保账款
rsrv_insur_cont|float|保险合同准备金
acting_trading_sec|float|代理买卖证券款
acting_uw_sec|float|代理承销证券款
non_cur_liab_due_1y|float|一年内到期的非流动负债
oth_cur_liab|float|其他流动负债
total_cur_liab|float|流动负债合计
bond_payable|float|应付债券
lt_payable|float|长期应付款
specific_payables|float|专项应付款
estimated_liab|float|预计负债
defer_tax_liab|float|递延所得税负债
defer_inc_non_cur_liab|float|递延收益-非流动负债
oth_ncl|float|其他非流动负债
total_ncl|float|非流动负债合计
depos_oth_bfi|float|同业和其它金融机构存放款项
deriv_liab|float|衍生金融负债
depos|float|吸收存款
agency_bus_liab|float|代理业务负债
oth_liab|float|其他负债
prem_receiv_adva|float|预收保费
depos_received|float|存入保证金
ph_invest|float|保户储金及投资款
reser_une_prem|float|未到期责任准备金
reser_outstd_claims|float|未决赔款准备金
reser_lins_liab|float|寿险责任准备金
reser_lthins_liab|float|长期健康险责任准备金
indept_acc_liab|float|独立账户负债
pledge_borr|float|其中:质押借款
indem_payable|float|应付赔付款
policy_div_payable|float|应付保单红利
total_liab|float|负债合计
treasury_share|float|减:库存股
ordin_risk_reser|float|一般风险准备
forex_differ|float|外币报表折算差额
invest_loss_unconf|float|未确认的投资损失
minority_int|float|少数股东权益
total_hldr_eqy_exc_min_int|float|股东权益合计(不含少数股东权益)
total_hldr_eqy_inc_min_int|float|股东权益合计(含少数股东权益)
total_liab_hldr_eqy|float|负债及股东权益总计
lt_payroll_payable|float|长期应付职工薪酬
oth_comp_income|float|其他综合收益
oth_eqt_tools|float|其他权益工具
oth_eqt_tools_p_shr|float|其他权益工具(优先股)
lending_funds|float|融出资金
acc_receivable|float|应收款项
st_fin_payable|float|应付短期融资款
payables|float|应付款项
hfs_assets|float|持有待售的资产
hfs_sales|float|持有待售的负债
net_profit|float|净利润
finan_exp|float|财务费用
c_fr_sale_sg|float|销售商品、提供劳务收到的现金
recp_tax_rends|float|收到的税费返还
n_depos_incr_fi|float|客户存款和同业存放款项净增加额
n_incr_loans_cb|float|向中央银行借款净增加额
n_inc_borr_oth_fi|float|向其他金融机构拆入资金净增加额
prem_fr_orig_contr|float|收到原保险合同保费取得的现金
n_incr_insured_dep|float|保户储金净增加额
n_reinsur_prem|float|收到再保业务现金净额
n_incr_disp_tfa|float|处置交易性金融资产净增加额
ifc_cash_incr|float|收取利息和手续费净增加额
n_incr_disp_faas|float|处置可供出售金融资产净增加额
n_incr_loans_oth_bank|float|拆入资金净增加额
n_cap_incr_repur|float|回购业务资金净增加额
c_fr_oth_operate_a|float|收到其他与经营活动有关的现金
c_inf_fr_operate_a|float|经营活动现金流入小计
c_paid_goods_s|float|购买商品、接受劳务支付的现金
c_paid_to_for_empl|float|支付给职工以及为职工支付的现金
c_paid_for_taxes|float|支付的各项税费
n_incr_clt_loan_adv|float|客户贷款及垫款净增加额
n_incr_dep_cbob|float|存放央行和同业款项净增加额
c_pay_claims_orig_inco|float|支付原保险合同赔付款项的现金
pay_handling_chrg|float|支付手续费的现金
pay_comm_insur_plcy|float|支付保单红利的现金
oth_cash_pay_oper_act|float|支付其他与经营活动有关的现金
st_cash_out_act|float|经营活动现金流出小计
n_cashflow_act|float|经营活动产生的现金流量净额
oth_recp_ral_inv_act|float|收到其他与投资活动有关的现金
c_disp_withdrwl_invest|float|收回投资收到的现金
c_recp_return_invest|float|取得投资收益收到的现金
n_recp_disp_fiolta|float|处置固定资产、无形资产和其他长期资产收回的现金净额
n_recp_disp_sobu|float|处置子公司及其他营业单位收到的现金净额
stot_inflows_inv_act|float|投资活动现金流入小计
c_pay_acq_const_fiolta|float|购建固定资产、无形资产和其他长期资产支付的现金
c_paid_invest|float|投资支付的现金
n_disp_subs_oth_biz|float|取得子公司及其他营业单位支付的现金净额
oth_pay_ral_inv_act|float|支付其他与投资活动有关的现金
n_incr_pledge_loan|float|质押贷款净增加额
stot_out_inv_act|float|投资活动现金流出小计
n_cashflow_inv_act|float|投资活动产生的现金流量净额
c_recp_borrow|float|取得借款收到的现金
proc_issue_bonds|float|发行债券收到的现金
oth_cash_recp_ral_fnc_act|float|收到其他与筹资活动有关的现金
stot_cash_in_fnc_act|float|筹资活动现金流入小计
free_cashflow|float|企业自由现金流量
c_prepay_amt_borr|float|偿还债务支付的现金
c_pay_dist_dpcp_int_exp|float|分配股利、利润或偿付利息支付的现金
incl_dvd_profit_paid_sc_ms|float|其中:子公司支付给少数股东的股利、利润
oth_cashpay_ral_fnc_act|float|支付其他与筹资活动有关的现金
stot_cashout_fnc_act|float|筹资活动现金流出小计
n_cash_flows_fnc_act|float|筹资活动产生的现金流量净额
eff_fx_flu_cash|float|汇率变动对现金的影响
n_incr_cash_cash_equ|float|现金及现金等价物净增加额
c_cash_equ_beg_period|float|期初现金及现金等价物余额
c_cash_equ_end_period|float|期末现金及现金等价物余额
c_recp_cap_contrib|float|吸收投资收到的现金
incl_cash_rec_saims|float|其中:子公司吸收少数股东投资收到的现金
uncon_invest_loss|float|未确认投资损失
prov_depr_assets|float|加:资产减值准备
depr_fa_coga_dpba|float|固定资产折旧、油气资产折耗、生产性生物资产折旧
amort_intang_assets|float|无形资产摊销
lt_amort_deferred_exp|float|长期待摊费用摊销
decr_deferred_exp|float|待摊费用减少
incr_acc_exp|float|预提费用增加
loss_disp_fiolta|float|处置固定、无形资产和其他长期资产的损失
loss_scr_fa|float|固定资产报废损失
loss_fv_chg|float|公允价值变动损失
invest_loss|float|投资损失
decr_def_inc_tax_assets|float|递延所得税资产减少
incr_def_inc_tax_liab|float|递延所得税负债增加
decr_inventories|float|存货的减少
decr_oper_payable|float|经营性应收项目的减少
incr_oper_payable|float|经营性应付项目的增加
others|float|其他
im_net_cashflow_oper_act|float|经营活动产生的现金流量净额(间接法)
conv_debt_into_cap|float|债务转为资本
conv_copbonds_due_within_1y|float|一年内到期的可转换公司债券
fa_fnc_leases|float|融资租入固定资产
end_bal_cash|float|现金的期末余额
beg_bal_cash|float|减:现金的期初余额
end_bal_cash_equ|float|加:现金等价物的期末余额
beg_bal_cash_equ|float|减:现金等价物的期初余额
im_n_incr_cash_equ|float|现金及现金等价物净增加额(间接法)
eps|float|基本每股收益
dt_eps|float|稀释每股收益
total_revenue_ps|float|每股营业总收入
revenue_ps|float|每股营业收入
capital_rese_ps|float|每股资本公积
surplus_rese_ps|float|每股盈余公积
undist_profit_ps|float|每股未分配利润
extra_item|float|非经常性损益
profit_dedt|float|扣除非经常性损益后的净利润
gross_margin|float|毛利
current_ratio|float|流动比率
quick_ratio|float|速动比率
cash_ratio|float|保守速动比率
invturn_days|float|存货周转天数
arturn_days|float|应收账款周转天数
inv_turn|float|存货周转率
ar_turn|float|应收账款周转率
ca_turn|float|流动资产周转率
fa_turn|float|固定资产周转率
assets_turn|float|总资产周转率
op_income|float|经营活动净收益
valuechange_income|float|价值变动净收益
interst_income|float|利息费用
daa|float|折旧与摊销
ebit|float|息税前利润
ebitda|float|息税折旧摊销前利润
fcff|float|企业自由现金流量
fcfe|float|股权自由现金流量
current_exint|float|无息流动负债
noncurrent_exint|float|无息非流动负债
interestdebt|float|带息债务
netdebt|float|净债务
tangible_asset|float|有形资产
working_capital|float|营运资金
networking_capital|float|营运流动资本
invest_capital|float|全部投入资本
retained_earnings|float|留存收益
diluted2_eps|float|期末摊薄每股收益
bps|float|每股净资产
ocfps|float|每股经营活动产生的现金流量净额
retainedps|float|每股留存收益
cfps|float|每股现金流量净额
ebit_ps|float|每股息税前利润
fcff_ps|float|每股企业自由现金流量
fcfe_ps|float|每股股东自由现金流量
netprofit_margin|float|销售净利率
grossprofit_margin|float|销售毛利率
cogs_of_sales|float|销售成本率
expense_of_sales|float|销售期间费用率
profit_to_gr|float|净利润/营业总收入
saleexp_to_gr|float|销售费用/营业总收入
adminexp_of_gr|float|管理费用/营业总收入
finaexp_of_gr|float|财务费用/营业总收入
impai_ttm|float|资产减值损失/营业总收入
gc_of_gr|float|营业总成本/营业总收入
op_of_gr|float|营业利润/营业总收入
ebit_of_gr|float|息税前利润/营业总收入
roe|float|净资产收益率
roe_waa|float|加权平均净资产收益率
roe_dt|float|净资产收益率(扣除非经常损益)
roa|float|总资产报酬率
npta|float|总资产净利润
roic|float|投入资本回报率
roe_yearly|float|年化净资产收益率
roa2_yearly|float|年化总资产报酬率
roe_avg|float|平均净资产收益率(增发条件)
opincome_of_ebt|float|经营活动净收益/利润总额
investincome_of_ebt|float|价值变动净收益/利润总额
n_op_profit_of_ebt|float|营业外收支净额/利润总额
tax_to_ebt|float|所得税/利润总额
dtprofit_to_profit|float|扣除非经常损益后的净利润/净利润
salescash_to_or|float|销售商品提供劳务收到的现金/营业收入
ocf_to_or|float|经营活动产生的现金流量净额/营业收入
ocf_to_opincome|float|经营活动产生的现金流量净额/经营活动净收益
capitalized_to_da|float|资本支出/折旧和摊销
debt_to_assets|float|资产负债率
assets_to_eqt|float|权益乘数
dp_assets_to_eqt|float|权益乘数(杜邦分析)
ca_to_assets|float|流动资产/总资产
nca_to_assets|float|非流动资产/总资产
tbassets_to_totalassets|float|有形资产/总资产
int_to_talcap|float|带息债务/全部投入资本
eqt_to_talcapital|float|归属于母公司的股东权益/全部投入资本
currentdebt_to_debt|float|流动负债/负债合计
longdeb_to_debt|float|非流动负债/负债合计
ocf_to_shortdebt|float|经营活动产生的现金流量净额/流动负债
debt_to_eqt|float|产权比率
eqt_to_debt|float|归属于母公司的股东权益/负债合计
eqt_to_interestdebt|float|归属于母公司的股东权益/带息债务
tangibleasset_to_debt|float|有形资产/负债合计
tangasset_to_intdebt|float|有形资产/带息债务
tangibleasset_to_netdebt|float|有形资产/净债务
ocf_to_debt|float|经营活动产生的现金流量净额/负债合计
ocf_to_interestdebt|float|经营活动产生的现金流量净额/带息债务
ocf_to_netdebt|float|经营活动产生的现金流量净额/净债务
ebit_to_interest|float|已获利息倍数(EBIT/利息费用)
longdebt_to_workingcapital|float|长期债务与营运资金比率
ebitda_to_debt|float|息税折旧摊销前利润/负债合计
turn_days|float|营业周期
roa_yearly|float|年化总资产净利率
roa_dp|float|总资产净利率(杜邦分析)
fixed_assets|float|固定资产合计
profit_prefin_exp|float|扣除财务费用前营业利润
non_op_profit|float|非营业利润
op_to_ebt|float|营业利润／利润总额
nop_to_ebt|float|非营业利润／利润总额
ocf_to_profit|float|经营活动产生的现金流量净额／营业利润
cash_to_liqdebt|float|货币资金／流动负债
cash_to_liqdebt_withinterest|float|货币资金／带息流动负债
op_to_liqdebt|float|营业利润／流动负债
op_to_debt|float|营业利润／负债合计
roic_yearly|float|年化投入资本回报率
total_fa_trun|float|固定资产合计周转率
profit_to_op|float|利润总额／营业收入
q_opincome|float|经营活动单季度净收益
q_investincome|float|价值变动单季度净收益
q_dtprofit|float|扣除非经常损益后的单季度净利润
q_eps|float|每股收益(单季度)
q_netprofit_margin|float|销售净利率(单季度)
q_gsprofit_margin|float|销售毛利率(单季度)
q_exp_to_sales|float|销售期间费用率(单季度)
q_profit_to_gr|float|净利润／营业总收入(单季度)
q_saleexp_to_gr|float|销售费用／营业总收入 (单季度)
q_adminexp_to_gr|float|管理费用／营业总收入 (单季度)
q_finaexp_to_gr|float|财务费用／营业总收入 (单季度)
q_impair_to_gr_ttm|float|资产减值损失／营业总收入(单季度)
q_gc_to_gr|float|营业总成本／营业总收入 (单季度)
q_op_to_gr|float|营业利润／营业总收入(单季度)
q_roe|float|净资产收益率(单季度)
q_dt_roe|float|净资产单季度收益率(扣除非经常损益)
q_npta|float|总资产净利润(单季度)
q_opincome_to_ebt|float|经营活动净收益／利润总额(单季度)
q_investincome_to_ebt|float|价值变动净收益／利润总额(单季度)
q_dtprofit_to_profit|float|扣除非经常损益后的净利润／净利润(单季度)
q_salescash_to_or|float|销售商品提供劳务收到的现金／营业收入(单季度)
q_ocf_to_sales|float|经营活动产生的现金流量净额／营业收入(单季度)
q_ocf_to_or|float|经营活动产生的现金流量净额／经营活动净收益(单季度)
basic_eps_yoy|float|基本每股收益同比增长率(%)
dt_eps_yoy|float|稀释每股收益同比增长率(%)
cfps_yoy|float|每股经营活动产生的现金流量净额同比增长率(%)
op_yoy|float|营业利润同比增长率(%)
ebt_yoy|float|利润总额同比增长率(%)
netprofit_yoy|float|归属母公司股东的净利润同比增长率(%)
dt_netprofit_yoy|float|归属母公司股东的净利润-扣除非经常损益同比增长率(%)
ocf_yoy|float|经营活动产生的现金流量净额同比增长率(%)
roe_yoy|float|净资产收益率(摊薄)同比增长率(%)
bps_yoy|float|每股净资产相对年初增长率(%)
assets_yoy|float|资产总计相对年初增长率(%)
eqt_yoy|float|归属母公司的股东权益相对年初增长率(%)
tr_yoy|float|营业总收入同比增长率(%)
or_yoy|float|营业收入同比增长率(%)
q_gr_yoy|float|营业总收入同比增长率(%)(单季度)
q_gr_qoq|float|营业总收入环比增长率(%)(单季度)
q_sales_yoy|float|营业收入同比增长率(%)(单季度)
q_sales_qoq|float|营业收入环比增长率(%)(单季度)
q_op_yoy|float|营业利润同比增长率(%)(单季度)
q_op_qoq|float|营业利润环比增长率(%)(单季度)
q_profit_yoy|float|净利润同比增长率(%)(单季度)
q_profit_qoq|float|净利润环比增长率(%)(单季度)
q_netprofit_yoy|float|归属母公司股东的净利润同比增长率(%)(单季度)
q_netprofit_qoq|float|归属母公司股东的净利润环比增长率(%)(单季度)
equity_yoy|float|净资产同比增长率

### 技术指标分析

技术指标分析是主要的分析手段，技术指标主要包括：

| 指标 | 格式 | 含义 |
| --- | --- | --- |
| open | float | 开盘价 |
|high|float|最高价
|low|float|最低价
|close|float|收盘价
|pre_close|float|昨收价
|change|float|涨跌额
|pct_chg|float|涨跌幅 （未复权，如果是复权请用 通用行情接口 ）
|vol|float|成交量 （手）
|amount|float|成交额 （千元）
|close|float|当日收盘价
|turnover_rate|float|换手率（%）
|turnover_rate_f|float|换手率（自由流通股）
|volume_ratio|float|量比
|pe|float|市盈率（总市值/净利润）
|pe_ttm|float|市盈率（TTM）
|pb|float|市净率（总市值/净资产）
|ps|float|市销率
|ps_ttm|float|市销率（TTM）
|total_share|float|总股本 （万股）
|float_share|float|流通股本 （万股）
|free_share|float|自由流通股本 （万）
|total_mv|float|总市值 （万元）
|circ_mv|float|流通市值（万元）
|buy_sm_vol|int|小单买入量（手）
|buy_sm_amount|float|小单买入金额（万元）
|sell_sm_vol|int|小单卖出量（手）
|sell_sm_amount|float|小单卖出金额（万元）
|buy_md_vol|int|中单买入量（手）
|buy_md_amount|float|中单买入金额（万元）
|sell_md_vol|int|中单卖出量（手）
|sell_md_amount|float|中单卖出金额（万元）
|buy_lg_vol|int|大单买入量（手）
|buy_lg_amount|float|大单买入金额（万元）
|sell_lg_vol|int|大单卖出量（手）
|sell_lg_amount|float|大单卖出金额（万元）
|buy_elg_vol|int|特大单买入量（手）
|buy_elg_amount|float|特大单买入金额（万元）
|sell_elg_vol|int|特大单卖出量（手）
|sell_elg_amount|float|特大单卖出金额（万元）
|net_mf_vol|int|净流入量（手）
|net_mf_amount|float|净流入额（万元）


### 特征重要性分析和降维

分析股价数据特征，主要对开盘价、收盘价、最高价、最低价、涨跌幅、成交额、换手率、量比、市盈率、市净率、市销率、流通股本、总股本、小、中、大、特大单买入卖出额等特征进行分析，并且对以上特征应用了指数移动平均、差分自相关等数据处理手段。


<center><img src='project\feature_engineering\1_open_tech.png' width=1060></img></center>

<center><img src='project\feature_engineering\2_high_tech.png' width=1060></img></center>

<center><img src='project\feature_engineering\3_low_tech.png' width=1060></img></center>

<center><img src='project\feature_engineering\4_close_tech.png' width=1060></img></center>

<center><img src='project\feature_engineering\5_change_percentage_tech.png' width=1060></img></center>

<center><img src='project\feature_engineering\7_turnover_rate_tech.png' width=1060></img></center>

<center><img src='project\feature_engineering\8_volume_ratio_tech.png' width=1060></img></center>

<center><img src='project\feature_engineering\9_pe_tech.png' width=1060></img></center>

<center><img src='project\feature_engineering\20_buy_extra_tech.png' width=1060></img></center>

<center><img src='project\feature_engineering\22_fourier_transforms.png' width=1060></img></center>

<center><img src='project\feature_engineering\23_close_price_correlations.png' width=1060></img></center>

<center><img src='project\feature_engineering\24_ARIMA.png' width=1060></img></center>

<center><img src='project\feature_engineering\25_xgboost_training.png' width=1060></img></center>

<center><img src='project\feature_engineering\26_feature_importance.png' width=1060></img></center>



## 训练模型




## 评估和调参

TODO

## 可视化

TODO

## 参考资料

感谢支持，赚个积分。

https://tushare.pro/register?reg=266868 