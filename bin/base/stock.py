import tushare as ts 
import pandas as pd 

'''
Stock类 定义股票基本参数
Trade类 定义交易的基本参数
Company类 定义上市公司基本参数


'''

class Stock(object): # 股票类
    def __init__(self, 
                ts_code=None, # TS代码
                symbol=None, # 股票代码
                name=None, # 股票名称
                area=None, # 所在地域
                industry=None, # 所属行业
                fullname=None, # 股票全称
                enname=None,# 英文全称
                market=None, # 市场类型：主板、中小板、创业板
                exchange=None, # 交易所代码
                curr_type=None, # 交易货币
                list_status=None,# 上市状态：L上市、D退市、P暂停上市
                list_date=None, # 上市日期
                delist_date=None, # 退市日期
                is_HS=None):# 是否沪深港通标的：N否、H沪港通、S深港通
        self.ts_code = ts_code
        self.symbol = symbol
        self.name = name
        self.area = area
        self.industry = industry
        self.fullname = fullname
        self.enname = enname
        self.market = market
        self.exchange = exchange
        self.curr_type = curr_type
        self.list_status = list_status
        self.list_date = list_date
        self.delist_date = delist_date
        self.is_HS = is_HS 


class Trade(object): # 交易类
    def __init__(self, 
                market=None, #市场名称、交易所名称（指数和期货）
                exchange=None, # 交易所名称
                start_date=None, # 开始日期
                end_date=None, # 结束日期
                trade_date=None, # 交易日期
                date=None # 新闻发布日期
                ):
        self.exchange = exchange
        self.start_date = start_date
        self.end_date = end_date
        self.trade_date = trade_date
        self.market = market
        self.date = date


class Company(object): # 上市公司类
    def __init__(self,
                ts_code=None,# 股票代码
                exchange=None,# 交易所
                chairman=None,# 法人代表
                manager=None,# 总经理
                secretary=None,# 董秘
                reg_capital=None,# 注册资本
                setup_date=None,# 注册日期
                province=None,# 所在省份
                city=None,# 所在城市
                introduction=None,# 介绍
                website=None,# 主页
                email=None,# 电邮
                office=None,# 办公室
                employees=None,# 员工人数
                main_business=None,# 主营业务
                business_scope=None):# 业务范围
        self.ts_code = ts_code
        self.exchange = exchange
        self.chairman = chairman
        self.manager = manager
        self.secretary = secretary
        self.reg_capital = reg_capital
        self.setup_date = setup_date
        self.province = province
        self.city = city
        self.introduction = introduction
        self.website = website
        self.email = email
        self.office = office
        self.employees = employees
        self.main_business = main_business
        self.business_scope = business_scope


class Parameters(Stock, Trade, Company):
    '''# 需要传递给函数的参数 包括了指数、期货所需参数'''
    def __init__(self, 
                    ts_code=None, 
                    start_date=None, 
                    end_date=None, 
                    trade_date=None,
                    market=None,
                    exchange=None,
                    date=None,
                    year=None
                    ):
        Stock.__init__(self, ts_code=ts_code)
        Trade.__init__(self, 
                        start_date=start_date, 
                        end_date=end_date, 
                        trade_date=trade_date,
                        market=market,
                        exchange=exchange,
                        date=date)
        self.year=year
        with open('bin\\base\\token.tkn','r') as token:
            mytoken = token.readline().rstrip('\n')
            # print('Your token is ' + mytoken)
        ts.set_token(mytoken)
        self.pro = ts.pro_api()

    def getToken(self):# 获取token
        return self.pro


class StockData(Stock, Trade):
    '''# 股票基本数据类'''
    def __init__(self, para):
        self.api = para.pro
        Stock.__init__(self, ts_code=para.ts_code)
        Trade.__init__(self,
                        start_date=para.start_date, 
                        end_date=para.end_date, 
                        trade_date=para.trade_date)
        
    def getStockList(self):
        '''
        获取股票列表

        输入参数

        名称	类型	必选	描述

        is_hs	str	N	是否沪深港通标的，N否 H沪股通 S深股通
        list_status	str	N	上市状态： L上市 D退市 P暂停上市
        exchange	str	N	交易所 SSE上交所 SZSE深交所 HKEX港交所(未上线)
        
        输出参数

        名称	类型	描述

        ts_code	str	TS代码
        symbol	str	股票代码
        name	str	股票名称
        area	str	所在地域
        industry	str	所属行业
        fullname	str	股票全称
        enname	str	英文全称
        market	str	市场类型 （主板/中小板/创业板）
        exchange	str	交易所代码
        curr_type	str	交易货币
        list_status	str	上市状态： L上市 D退市 P暂停上市
        list_date	str	上市日期
        delist_date	str	退市日期
        is_hs	str	是否沪深港通标的，N否 H沪股通 S深股通
        '''
        data = self.api.stock_basic(list_status='L',
                                    field='ts_code,symbol,name,area,industry,market,exchange,list_date,delist_date,is_hs')
        '''# 获取股票代码 名称、行业、市场等信息'''
        save = pd.DataFrame(data)
        return save 
        
    def getTradeCalender(self):
        '''
        获取交易日信息并存本地 参数：开始结束日期
        
        输入参数

        名称	类型	必选	描述

        exchange	str	N	交易所 SSE上交所 SZSE深交所
        start_date	str	N	开始日期
        end_date	str	N	结束日期
        is_open	str	N	是否交易 '0'休市 '1'交易

        输出参数

        名称	类型	默认显示	描述

        exchange	str	Y	交易所 SSE上交所 SZSE深交所
        cal_date	str	Y	日历日期
        is_open	str	Y	是否交易 0休市 1交易
        pretrade_date	str	N	上一个交易日
        '''
        data = self.api.trade_cal(exchange='',
        start_date=self.start_date,
        end_date=self.end_date,
        is_open='1')
        save = pd.DataFrame(data)
        return save

    def getDaily(self): 
        '''
        获取个股日线行情 参数：股票代码 开始时间 结束时间
        
        输入参数

        名称	类型	必选	描述

        ts_code	str	N	股票代码（二选一）
        trade_date	str	N	交易日期（二选一）
        start_date	str	N	开始日期(YYYYMMDD)
        end_date	str	N	结束日期(YYYYMMDD)
        注：日期都填YYYYMMDD格式，比如20181010

        输出参数

        名称	类型	描述

        ts_code	str	股票代码
        trade_date	str	交易日期
        open	float	开盘价
        high	float	最高价
        low	float	最低价
        close	float	收盘价
        pre_close	float	昨收价
        change	float	涨跌额
        pct_chg	float	涨跌幅 （未复权，如果是复权请用 通用行情接口 ）
        vol	float	成交量 （手）
        amount	float	成交额 （千元）
        '''
        data = self.api.daily(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date)
        return data

    def getDailyIndicator(self):
        '''获取每日指标
        
        输入参数

        名称	类型	必选	描述

        ts_code	str	Y	股票代码（二选一）
        trade_date	str	N	交易日期 （二选一）
        start_date	str	N	开始日期(YYYYMMDD)
        end_date	str	N	结束日期(YYYYMMDD)
        注：日期都填YYYYMMDD格式，比如20181010

        输出参数

        名称	类型	描述

        ts_code	str	TS股票代码
        trade_date	str	交易日期
        close	float	当日收盘价
        turnover_rate	float	换手率（%）
        turnover_rate_f	float	换手率（自由流通股）
        volume_ratio	float	量比
        pe	float	市盈率（总市值/净利润）
        pe_ttm	float	市盈率（TTM）
        pb	float	市净率（总市值/净资产）
        ps	float	市销率
        ps_ttm	float	市销率（TTM）
        total_share	float	总股本 （万股）
        float_share	float	流通股本 （万股）
        free_share	float	自由流通股本 （万）
        total_mv	float	总市值 （万元）
        circ_mv	float	流通市值（万元）
        
        '''
        data = self.api.daily_basic(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date)
        return data

    def getMoneyflow(self):
        '''# 获取个股资金流向
        
        输入参数

        名称	类型	必选	描述

        ts_code	str	N	股票代码 （股票和时间参数至少输入一个）
        trade_date	str	N	交易日期
        start_date	str	N	开始日期
        end_date	str	N	结束日期


        输出参数

        名称	类型	默认显示	描述

        ts_code	str	Y	TS代码
        trade_date	str	Y	交易日期
        buy_sm_vol	int	Y	小单买入量（手）
        buy_sm_amount	float	Y	小单买入金额（万元）
        sell_sm_vol	int	Y	小单卖出量（手）
        sell_sm_amount	float	Y	小单卖出金额（万元）
        buy_md_vol	int	Y	中单买入量（手）
        buy_md_amount	float	Y	中单买入金额（万元）
        sell_md_vol	int	Y	中单卖出量（手）
        sell_md_amount	float	Y	中单卖出金额（万元）
        buy_lg_vol	int	Y	大单买入量（手）
        buy_lg_amount	float	Y	大单买入金额（万元）
        sell_lg_vol	int	Y	大单卖出量（手）
        sell_lg_amount	float	Y	大单卖出金额（万元）
        buy_elg_vol	int	Y	特大单买入量（手）
        buy_elg_amount	float	Y	特大单买入金额（万元）
        sell_elg_vol	int	Y	特大单卖出量（手）
        sell_elg_amount	float	Y	特大单卖出金额（万元）
        net_mf_vol	int	Y	净流入量（手）
        net_mf_amount	float	Y	净流入额（万元）

        各类别统计规则如下：
        小单：5万以下 中单：5万～20万 大单：20万～100万 特大单：成交额>=100万
        '''
        data = self.api.moneyflow(ts_code=self.ts_code,
                                    start_date=self.start_date,
                                    end_date=self.end_date)
        return data

    def getRestoration(self, asset='E', freq='D', adj=None):
        '''
        获取复权行情

        复权说明

        类型	算法	参数标识

        不复权	无	空或None
        前复权	当日收盘价 × 当日复权因子 / 最新复权因子	qfq
        后复权	当日收盘价 × 当日复权因子	hfq
        注：目前支持A股的日线/周线/月线复权，分钟复权稍后支持



        接口参数

        名称	类型	必选	描述

        ts_code	str	Y	证券代码
        pro_api	str	N	pro版api对象
        start_date	str	N	开始日期 (格式：YYYYMMDD)
        end_date	str	N	结束日期 (格式：YYYYMMDD)
        asset	str	Y	资产类别：E股票 I沪深指数 C数字货币 F期货 FD基金 O期权，默认E
        adj	str	N	复权类型(只针对股票)：None未复权 qfq前复权 hfq后复权 , 默认None
        freq	str	Y	数据频度 ：1MIN表示1分钟（1/5/15/30/60分钟） D日线 ，默认D
        ma	list	N	均线，支持任意周期的均价和均量，输入任意合理int数值  
        '''
        data = ts.pro_bar(ts_code=self.ts_code,
                                start_date=self.start_date,
                                end_date=self.end_date,
                                asset=asset,
                                freq=freq,
                                adj=adj)
        return data

    def getSuspend(self):
        '''
        获取股票每日停复牌信息

        输入参数

        名称	类型	必选	描述
        ts_code	str	N	股票代码(三选一)
        suspend_date	str	N	停牌日期(三选一)
        resume_date	str	N	复牌日期(三选一)
        输出参数

        名称	类型	描述
        ts_code	str	股票代码
        suspend_date	str	停牌日期
        resume_date	str	复牌日期
        ann_date	str	公告日期
        suspend_reason	str	停牌原因
        reason_type	str	停牌原因类别
        '''
        data = self.api.suspend(ts_code=self.ts_code)
        return data


class StockFinance(Stock, Trade):
    '''# 股票财务信息类'''
    def __init__(self, para):
        self.api = para.pro
        Stock.__init__(self, ts_code=para.ts_code)
        Trade.__init__(self,
                        start_date=para.start_date, 
                        end_date=para.end_date, 
                        trade_date=para.trade_date)

    def getIncome(self):
        '''# 获取上市公司财务利润
        
                
        输入参数

        名称	类型	必选	描述

        ts_code	str	Y	股票代码
        ann_date	str	N	公告日期
        start_date	str	N	公告开始日期
        end_date	str	N	公告结束日期
        period	str	N	报告期(每个季度最后一天的日期，比如20171231表示年报)
        report_type	str	N	报告类型： 参考下表说明
        comp_type	str	N	公司类型：1一般工商业 2银行 3保险 4证券
       
        输出参数

        名称	类型	默认显示	描述

        ts_code	str	Y	TS代码
        ann_date	str	Y	公告日期
        f_ann_date	str	Y	实际公告日期
        end_date	str	Y	报告期
        report_type	str	Y	报告类型 1合并报表 2单季合并 3调整单季合并表 4调整合并报表 5调整前合并报表 6母公司报表 7母公司单季表 8 母公司调整单季表 9母公司调整表 10母公司调整前报表 11调整前合并报表 12母公司调整前报表
        comp_type	str	Y	公司类型(1一般工商业2银行3保险4证券)
        basic_eps	float	Y	基本每股收益
        diluted_eps	float	Y	稀释每股收益
        total_revenue	float	Y	营业总收入
        revenue	float	Y	营业收入
        int_income	float	Y	利息收入
        prem_earned	float	Y	已赚保费
        comm_income	float	Y	手续费及佣金收入
        n_commis_income	float	Y	手续费及佣金净收入
        n_oth_income	float	Y	其他经营净收益
        n_oth_b_income	float	Y	加:其他业务净收益
        prem_income	float	Y	保险业务收入
        out_prem	float	Y	减:分出保费
        une_prem_reser	float	Y	提取未到期责任准备金
        reins_income	float	Y	其中:分保费收入
        n_sec_tb_income	float	Y	代理买卖证券业务净收入
        n_sec_uw_income	float	Y	证券承销业务净收入
        n_asset_mg_income	float	Y	受托客户资产管理业务净收入
        oth_b_income	float	Y	其他业务收入
        fv_value_chg_gain	float	Y	加:公允价值变动净收益
        invest_income	float	Y	加:投资净收益
        ass_invest_income	float	Y	其中:对联营企业和合营企业的投资收益
        forex_gain	float	Y	加:汇兑净收益
        total_cogs	float	Y	营业总成本
        oper_cost	float	Y	减:营业成本
        int_exp	float	Y	减:利息支出
        comm_exp	float	Y	减:手续费及佣金支出
        biz_tax_surchg	float	Y	减:营业税金及附加
        sell_exp	float	Y	减:销售费用
        admin_exp	float	Y	减:管理费用
        fin_exp	float	Y	减:财务费用
        assets_impair_loss	float	Y	减:资产减值损失
        prem_refund	float	Y	退保金
        compens_payout	float	Y	赔付总支出
        reser_insur_liab	float	Y	提取保险责任准备金
        div_payt	float	Y	保户红利支出
        reins_exp	float	Y	分保费用
        oper_exp	float	Y	营业支出
        compens_payout_refu	float	Y	减:摊回赔付支出
        insur_reser_refu	float	Y	减:摊回保险责任准备金
        reins_cost_refund	float	Y	减:摊回分保费用
        other_bus_cost	float	Y	其他业务成本
        operate_profit	float	Y	营业利润
        non_oper_income	float	Y	加:营业外收入
        non_oper_exp	float	Y	减:营业外支出
        nca_disploss	float	Y	其中:减:非流动资产处置净损失
        total_profit	float	Y	利润总额
        income_tax	float	Y	所得税费用
        n_income	float	Y	净利润(含少数股东损益)
        n_income_attr_p	float	Y	净利润(不含少数股东损益)
        minority_gain	float	Y	少数股东损益
        oth_compr_income	float	Y	其他综合收益
        t_compr_income	float	Y	综合收益总额
        compr_inc_attr_p	float	Y	归属于母公司(或股东)的综合收益总额
        compr_inc_attr_m_s	float	Y	归属于少数股东的综合收益总额
        ebit	float	Y	息税前利润
        ebitda	float	Y	息税折旧摊销前利润
        insurance_exp	float	Y	保险业务支出
        undist_profit	float	Y	年初未分配利润
        distable_profit	float	Y	可分配利润
        update_flag	str	N	更新标识

        主要报表类型说明

        代码	类型	说明

        1	合并报表	上市公司最新报表（默认）
        2	单季合并	单一季度的合并报表
        3	调整单季合并表	调整后的单季合并报表（如果有）
        4	调整合并报表	本年度公布上年同期的财务报表数据，报告期为上年度
        5	调整前合并报表	数据发生变更，将原数据进行保留，即调整前的原数据
        6	母公司报表	该公司母公司的财务报表数据
        7	母公司单季表	母公司的单季度表
        8	母公司调整单季表	母公司调整后的单季表
        9	母公司调整表	该公司母公司的本年度公布上年同期的财务报表数据
        10	母公司调整前报表	母公司调整之前的原始财务报表数据
        11	调整前合并报表	调整之前合并报表原数据
        12	母公司调整前报表	母公司报表发生变更前保留的原数据
        '''
        data = self.api.income(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date)
        return data
    
    def getBalanceSheet(self):
        '''# 获取上市公司资产负债表
        
        输入参数

        名称	类型	必选	描述

        ts_code	str	Y	股票代码
        ann_date	str	N	公告日期
        start_date	str	N	公告开始日期
        end_date	str	N	公告结束日期
        period	str	N	报告期(每个季度最后一天的日期，比如20171231表示年报)
        report_type	str	N	报告类型：见下方详细说明
        comp_type	str	N	公司类型：1一般工商业 2银行 3保险 4证券
        
        输出参数

        名称	类型	默认显示	描述

        ts_code	str	Y	TS股票代码
        ann_date	str	Y	公告日期
        f_ann_date	str	Y	实际公告日期
        end_date	str	Y	报告期
        report_type	str	Y	报表类型
        comp_type	str	Y	公司类型
        total_share	float	Y	期末总股本
        cap_rese	float	Y	资本公积金
        undistr_porfit	float	Y	未分配利润
        surplus_rese	float	Y	盈余公积金
        special_rese	float	Y	专项储备
        money_cap	float	Y	货币资金
        trad_asset	float	Y	交易性金融资产
        notes_receiv	float	Y	应收票据
        accounts_receiv	float	Y	应收账款
        oth_receiv	float	Y	其他应收款
        prepayment	float	Y	预付款项
        div_receiv	float	Y	应收股利
        int_receiv	float	Y	应收利息
        inventories	float	Y	存货
        amor_exp	float	Y	长期待摊费用
        nca_within_1y	float	Y	一年内到期的非流动资产
        sett_rsrv	float	Y	结算备付金
        loanto_oth_bank_fi	float	Y	拆出资金
        premium_receiv	float	Y	应收保费
        reinsur_receiv	float	Y	应收分保账款
        reinsur_res_receiv	float	Y	应收分保合同准备金
        pur_resale_fa	float	Y	买入返售金融资产
        oth_cur_assets	float	Y	其他流动资产
        total_cur_assets	float	Y	流动资产合计
        fa_avail_for_sale	float	Y	可供出售金融资产
        htm_invest	float	Y	持有至到期投资
        lt_eqt_invest	float	Y	长期股权投资
        invest_real_estate	float	Y	投资性房地产
        time_deposits	float	Y	定期存款
        oth_assets	float	Y	其他资产
        lt_rec	float	Y	长期应收款
        fix_assets	float	Y	固定资产
        cip	float	Y	在建工程
        const_materials	float	Y	工程物资
        fixed_assets_disp	float	Y	固定资产清理
        produc_bio_assets	float	Y	生产性生物资产
        oil_and_gas_assets	float	Y	油气资产
        intan_assets	float	Y	无形资产
        r_and_d	float	Y	研发支出
        goodwill	float	Y	商誉
        lt_amor_exp	float	Y	长期待摊费用
        defer_tax_assets	float	Y	递延所得税资产
        decr_in_disbur	float	Y	发放贷款及垫款
        oth_nca	float	Y	其他非流动资产
        total_nca	float	Y	非流动资产合计
        cash_reser_cb	float	Y	现金及存放中央银行款项
        depos_in_oth_bfi	float	Y	存放同业和其它金融机构款项
        prec_metals	float	Y	贵金属
        deriv_assets	float	Y	衍生金融资产
        rr_reins_une_prem	float	Y	应收分保未到期责任准备金
        rr_reins_outstd_cla	float	Y	应收分保未决赔款准备金
        rr_reins_lins_liab	float	Y	应收分保寿险责任准备金
        rr_reins_lthins_liab	float	Y	应收分保长期健康险责任准备金
        refund_depos	float	Y	存出保证金
        ph_pledge_loans	float	Y	保户质押贷款
        refund_cap_depos	float	Y	存出资本保证金
        indep_acct_assets	float	Y	独立账户资产
        client_depos	float	Y	其中：客户资金存款
        client_prov	float	Y	其中：客户备付金
        transac_seat_fee	float	Y	其中:交易席位费
        invest_as_receiv	float	Y	应收款项类投资
        total_assets	float	Y	资产总计
        lt_borr	float	Y	长期借款
        st_borr	float	Y	短期借款
        cb_borr	float	Y	向中央银行借款
        depos_ib_deposits	float	Y	吸收存款及同业存放
        loan_oth_bank	float	Y	拆入资金
        trading_fl	float	Y	交易性金融负债
        notes_payable	float	Y	应付票据
        acct_payable	float	Y	应付账款
        adv_receipts	float	Y	预收款项
        sold_for_repur_fa	float	Y	卖出回购金融资产款
        comm_payable	float	Y	应付手续费及佣金
        payroll_payable	float	Y	应付职工薪酬
        taxes_payable	float	Y	应交税费
        int_payable	float	Y	应付利息
        div_payable	float	Y	应付股利
        oth_payable	float	Y	其他应付款
        acc_exp	float	Y	预提费用
        deferred_inc	float	Y	递延收益
        st_bonds_payable	float	Y	应付短期债券
        payable_to_reinsurer	float	Y	应付分保账款
        rsrv_insur_cont	float	Y	保险合同准备金
        acting_trading_sec	float	Y	代理买卖证券款
        acting_uw_sec	float	Y	代理承销证券款
        non_cur_liab_due_1y	float	Y	一年内到期的非流动负债
        oth_cur_liab	float	Y	其他流动负债
        total_cur_liab	float	Y	流动负债合计
        bond_payable	float	Y	应付债券
        lt_payable	float	Y	长期应付款
        specific_payables	float	Y	专项应付款
        estimated_liab	float	Y	预计负债
        defer_tax_liab	float	Y	递延所得税负债
        defer_inc_non_cur_liab	float	Y	递延收益-非流动负债
        oth_ncl	float	Y	其他非流动负债
        total_ncl	float	Y	非流动负债合计
        depos_oth_bfi	float	Y	同业和其它金融机构存放款项
        deriv_liab	float	Y	衍生金融负债
        depos	float	Y	吸收存款
        agency_bus_liab	float	Y	代理业务负债
        oth_liab	float	Y	其他负债
        prem_receiv_adva	float	Y	预收保费
        depos_received	float	Y	存入保证金
        ph_invest	float	Y	保户储金及投资款
        reser_une_prem	float	Y	未到期责任准备金
        reser_outstd_claims	float	Y	未决赔款准备金
        reser_lins_liab	float	Y	寿险责任准备金
        reser_lthins_liab	float	Y	长期健康险责任准备金
        indept_acc_liab	float	Y	独立账户负债
        pledge_borr	float	Y	其中:质押借款
        indem_payable	float	Y	应付赔付款
        policy_div_payable	float	Y	应付保单红利
        total_liab	float	Y	负债合计
        treasury_share	float	Y	减:库存股
        ordin_risk_reser	float	Y	一般风险准备
        forex_differ	float	Y	外币报表折算差额
        invest_loss_unconf	float	Y	未确认的投资损失
        minority_int	float	Y	少数股东权益
        total_hldr_eqy_exc_min_int	float	Y	股东权益合计(不含少数股东权益)
        total_hldr_eqy_inc_min_int	float	Y	股东权益合计(含少数股东权益)
        total_liab_hldr_eqy	float	Y	负债及股东权益总计
        lt_payroll_payable	float	Y	长期应付职工薪酬
        oth_comp_income	float	Y	其他综合收益
        oth_eqt_tools	float	Y	其他权益工具
        oth_eqt_tools_p_shr	float	Y	其他权益工具(优先股)
        lending_funds	float	Y	融出资金
        acc_receivable	float	Y	应收款项
        st_fin_payable	float	Y	应付短期融资款
        payables	float	Y	应付款项
        hfs_assets	float	Y	持有待售的资产
        hfs_sales	float	Y	持有待售的负债
        update_flag	str	N	更新标识
        
        主要报表类型说明

        代码	类型	说明

        1	合并报表	上市公司最新报表（默认）
        2	单季合并	单一季度的合并报表
        3	调整单季合并表	调整后的单季合并报表（如果有）
        4	调整合并报表	本年度公布上年同期的财务报表数据，报告期为上年度
        5	调整前合并报表	数据发生变更，将原数据进行保留，即调整前的原数据
        6	母公司报表	该公司母公司的财务报表数据
        7	母公司单季表	母公司的单季度表
        8	母公司调整单季表	母公司调整后的单季表
        9	母公司调整表	该公司母公司的本年度公布上年同期的财务报表数据
        10	母公司调整前报表	母公司调整之前的原始财务报表数据
        11	调整前合并报表	调整之前合并报表原数据
        12	母公司调整前报表	母公司报表发生变更前保留的原数据'''
        
        data = self.api.balancesheet(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date)
        return data
    
    def getCashflow(self):
        '''# 获取上市公司现金流量表
        
        输入参数

        名称	类型	必选	描述

        ts_code	str	Y	股票代码
        ann_date	str	N	公告日期
        start_date	str	N	公告开始日期
        end_date	str	N	公告结束日期
        period	str	N	报告期(每个季度最后一天的日期，比如20171231表示年报)
        report_type	str	N	报告类型：见下方详细说明
        comp_type	str	N	公司类型：1一般工商业 2银行 3保险 4证券
        输出参数

        名称	类型	默认显示	描述

        ts_code	str	Y	TS股票代码
        ann_date	str	Y	公告日期
        f_ann_date	str	Y	实际公告日期
        end_date	str	Y	报告期
        comp_type	str	Y	报表类型
        report_type	str	Y	公司类型
        net_profit	float	Y	净利润
        finan_exp	float	Y	财务费用
        c_fr_sale_sg	float	Y	销售商品、提供劳务收到的现金
        recp_tax_rends	float	Y	收到的税费返还
        n_depos_incr_fi	float	Y	客户存款和同业存放款项净增加额
        n_incr_loans_cb	float	Y	向中央银行借款净增加额
        n_inc_borr_oth_fi	float	Y	向其他金融机构拆入资金净增加额
        prem_fr_orig_contr	float	Y	收到原保险合同保费取得的现金
        n_incr_insured_dep	float	Y	保户储金净增加额
        n_reinsur_prem	float	Y	收到再保业务现金净额
        n_incr_disp_tfa	float	Y	处置交易性金融资产净增加额
        ifc_cash_incr	float	Y	收取利息和手续费净增加额
        n_incr_disp_faas	float	Y	处置可供出售金融资产净增加额
        n_incr_loans_oth_bank	float	Y	拆入资金净增加额
        n_cap_incr_repur	float	Y	回购业务资金净增加额
        c_fr_oth_operate_a	float	Y	收到其他与经营活动有关的现金
        c_inf_fr_operate_a	float	Y	经营活动现金流入小计
        c_paid_goods_s	float	Y	购买商品、接受劳务支付的现金
        c_paid_to_for_empl	float	Y	支付给职工以及为职工支付的现金
        c_paid_for_taxes	float	Y	支付的各项税费
        n_incr_clt_loan_adv	float	Y	客户贷款及垫款净增加额
        n_incr_dep_cbob	float	Y	存放央行和同业款项净增加额
        c_pay_claims_orig_inco	float	Y	支付原保险合同赔付款项的现金
        pay_handling_chrg	float	Y	支付手续费的现金
        pay_comm_insur_plcy	float	Y	支付保单红利的现金
        oth_cash_pay_oper_act	float	Y	支付其他与经营活动有关的现金
        st_cash_out_act	float	Y	经营活动现金流出小计
        n_cashflow_act	float	Y	经营活动产生的现金流量净额
        oth_recp_ral_inv_act	float	Y	收到其他与投资活动有关的现金
        c_disp_withdrwl_invest	float	Y	收回投资收到的现金
        c_recp_return_invest	float	Y	取得投资收益收到的现金
        n_recp_disp_fiolta	float	Y	处置固定资产、无形资产和其他长期资产收回的现金净额
        n_recp_disp_sobu	float	Y	处置子公司及其他营业单位收到的现金净额
        stot_inflows_inv_act	float	Y	投资活动现金流入小计
        c_pay_acq_const_fiolta	float	Y	购建固定资产、无形资产和其他长期资产支付的现金
        c_paid_invest	float	Y	投资支付的现金
        n_disp_subs_oth_biz	float	Y	取得子公司及其他营业单位支付的现金净额
        oth_pay_ral_inv_act	float	Y	支付其他与投资活动有关的现金
        n_incr_pledge_loan	float	Y	质押贷款净增加额
        stot_out_inv_act	float	Y	投资活动现金流出小计
        n_cashflow_inv_act	float	Y	投资活动产生的现金流量净额
        c_recp_borrow	float	Y	取得借款收到的现金
        proc_issue_bonds	float	Y	发行债券收到的现金
        oth_cash_recp_ral_fnc_act	float	Y	收到其他与筹资活动有关的现金
        stot_cash_in_fnc_act	float	Y	筹资活动现金流入小计
        free_cashflow	float	Y	企业自由现金流量
        c_prepay_amt_borr	float	Y	偿还债务支付的现金
        c_pay_dist_dpcp_int_exp	float	Y	分配股利、利润或偿付利息支付的现金
        incl_dvd_profit_paid_sc_ms	float	Y	其中:子公司支付给少数股东的股利、利润
        oth_cashpay_ral_fnc_act	float	Y	支付其他与筹资活动有关的现金
        stot_cashout_fnc_act	float	Y	筹资活动现金流出小计
        n_cash_flows_fnc_act	float	Y	筹资活动产生的现金流量净额
        eff_fx_flu_cash	float	Y	汇率变动对现金的影响
        n_incr_cash_cash_equ	float	Y	现金及现金等价物净增加额
        c_cash_equ_beg_period	float	Y	期初现金及现金等价物余额
        c_cash_equ_end_period	float	Y	期末现金及现金等价物余额
        c_recp_cap_contrib	float	Y	吸收投资收到的现金
        incl_cash_rec_saims	float	Y	其中:子公司吸收少数股东投资收到的现金
        uncon_invest_loss	float	Y	未确认投资损失
        prov_depr_assets	float	Y	加:资产减值准备
        depr_fa_coga_dpba	float	Y	固定资产折旧、油气资产折耗、生产性生物资产折旧
        amort_intang_assets	float	Y	无形资产摊销
        lt_amort_deferred_exp	float	Y	长期待摊费用摊销
        decr_deferred_exp	float	Y	待摊费用减少
        incr_acc_exp	float	Y	预提费用增加
        loss_disp_fiolta	float	Y	处置固定、无形资产和其他长期资产的损失
        loss_scr_fa	float	Y	固定资产报废损失
        loss_fv_chg	float	Y	公允价值变动损失
        invest_loss	float	Y	投资损失
        decr_def_inc_tax_assets	float	Y	递延所得税资产减少
        incr_def_inc_tax_liab	float	Y	递延所得税负债增加
        decr_inventories	float	Y	存货的减少
        decr_oper_payable	float	Y	经营性应收项目的减少
        incr_oper_payable	float	Y	经营性应付项目的增加
        others	float	Y	其他
        im_net_cashflow_oper_act	float	Y	经营活动产生的现金流量净额(间接法)
        conv_debt_into_cap	float	Y	债务转为资本
        conv_copbonds_due_within_1y	float	Y	一年内到期的可转换公司债券
        fa_fnc_leases	float	Y	融资租入固定资产
        end_bal_cash	float	Y	现金的期末余额
        beg_bal_cash	float	Y	减:现金的期初余额
        end_bal_cash_equ	float	Y	加:现金等价物的期末余额
        beg_bal_cash_equ	float	Y	减:现金等价物的期初余额
        im_n_incr_cash_equ	float	Y	现金及现金等价物净增加额(间接法)
        update_flag	str	N	更新标识

        主要报表类型说明

        代码	类型	说明

        1	合并报表	上市公司最新报表（默认）
        2	单季合并	单一季度的合并报表
        3	调整单季合并表	调整后的单季合并报表（如果有）
        4	调整合并报表	本年度公布上年同期的财务报表数据，报告期为上年度
        5	调整前合并报表	数据发生变更，将原数据进行保留，即调整前的原数据
        6	母公司报表	该公司母公司的财务报表数据
        7	母公司单季表	母公司的单季度表
        8	母公司调整单季表	母公司调整后的单季表
        9	母公司调整表	该公司母公司的本年度公布上年同期的财务报表数据
        10	母公司调整前报表	母公司调整之前的原始财务报表数据
        11	调整前合并报表	调整之前合并报表原数据
        12	母公司调整前报表	母公司报表发生变更前保留的原数据'''
        data = self.api.cashflow(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date)
        return data
    
    def getForecast(self):
        '''# 获取业绩预告数据
        
        输入参数

        名称	类型	必选	描述

        ts_code	str	N	股票代码(二选一)
        ann_date	str	N	公告日期 (二选一)
        start_date	str	N	公告开始日期
        end_date	str	N	公告结束日期
        period	str	N	报告期(每个季度最后一天的日期，比如20171231表示年报)
        type	str	N	预告类型(预增/预减/扭亏/首亏/续亏/续盈/略增/略减)
        
        输出参数

        名称	类型	描述

        ts_code	str	TS股票代码
        ann_date	str	公告日期
        end_date	str	报告期
        type	str	业绩预告类型(预增/预减/扭亏/首亏/续亏/续盈/略增/略减)
        p_change_min	float	预告净利润变动幅度下限（%）
        p_change_max	float	预告净利润变动幅度上限（%）
        net_profit_min	float	预告净利润下限（万元）
        net_profit_max	float	预告净利润上限（万元）
        last_parent_net	float	上年同期归属母公司净利润
        first_ann_date	str	首次公告日
        summary	str	业绩预告摘要
        change_reason	str	业绩变动原因
        '''
        data = self.api.forecast(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date)
        return data
    
    def getExpress(self):
        '''# 获取上市公司业绩快报
        
        输入参数

        名称	类型	必选	描述

        ts_code	str	Y	股票代码
        ann_date	str	N	公告日期
        start_date	str	N	公告开始日期
        end_date	str	N	公告结束日期
        period	str	N	报告期(每个季度最后一天的日期,比如20171231表示年报)
       
        输出参数

        名称	类型	描述

        ts_code	str	TS股票代码
        ann_date	str	公告日期
        end_date	str	报告期
        revenue	float	营业收入(元)
        operate_profit	float	营业利润(元)
        total_profit	float	利润总额(元)
        n_income	float	净利润(元)
        total_assets	float	总资产(元)
        total_hldr_eqy_exc_min_int	float	股东权益合计(不含少数股东权益)(元)
        diluted_eps	float	每股收益(摊薄)(元)
        diluted_roe	float	净资产收益率(摊薄)(%)
        yoy_net_profit	float	去年同期修正后净利润
        bps	float	每股净资产
        yoy_sales	float	同比增长率:营业收入
        yoy_op	float	同比增长率:营业利润
        yoy_tp	float	同比增长率:利润总额
        yoy_dedu_np	float	同比增长率:归属母公司股东的净利润
        yoy_eps	float	同比增长率:基本每股收益
        yoy_roe	float	同比增减:加权平均净资产收益率
        growth_assets	float	比年初增长率:总资产
        yoy_equity	float	比年初增长率:归属母公司的股东权益
        growth_bps	float	比年初增长率:归属于母公司股东的每股净资产
        or_last_year	float	去年同期营业收入
        op_last_year	float	去年同期营业利润
        tp_last_year	float	去年同期利润总额
        np_last_year	float	去年同期净利润
        eps_last_year	float	去年同期每股收益
        open_net_assets	float	期初净资产
        open_bps	float	期初每股净资产
        perf_summary	str	业绩简要说明
        is_audit	int	是否审计： 1是 0否
        remark	str	备注
        '''
        data = self.api.express(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date)
        return data
    
    def getDividend(self):
        '''# 获取分红送股
        
        输入参数

        名称	类型	必选	描述

        ts_code	str	N	TS代码
        ann_date	str	N	公告日
        record_date	str	N	股权登记日期
        ex_date	str	N	除权除息日
        imp_ann_date	str	N	实施公告日

        以上参数至少有一个不能为空

        输出参数

        名称	类型	默认显示	描述

        ts_code	str	Y	TS代码
        end_date	str	Y	分红年度
        ann_date	str	Y	预案公告日
        div_proc	str	Y	实施进度
        stk_div	float	Y	每股送转
        stk_bo_rate	float	Y	每股送股比例
        stk_co_rate	float	Y	每股转增比例
        cash_div	float	Y	每股分红（税后）
        cash_div_tax	float	Y	每股分红（税前）
        record_date	str	Y	股权登记日
        ex_date	str	Y	除权除息日
        pay_date	str	Y	派息日
        div_listdate	str	Y	红股上市日
        imp_ann_date	str	Y	实施公告日
        base_date	str	N	基准日
        base_share	float	N	基准股本（万）'''
        data = self.api.dividend(ts_code=self.ts_code)
        return data
    
    def getFinacialIndicator(self):
        '''# 获取上市公司财务指标数据 
        
        输入参数

        名称	类型	必选	描述
        ts_code	str	Y	TS股票代码,e.g. 600001.SH/000001.SZ
        ann_date	str	N	公告日期
        start_date	str	N	报告期开始日期
        end_date	str	N	报告期结束日期
        period	str	N	报告期(每个季度最后一天的日期,比如20171231表示年报)
        输出参数

        名称	类型	默认显示	描述
        ts_code	str	Y	TS代码
        ann_date	str	Y	公告日期
        end_date	str	Y	报告期
        eps	float	Y	基本每股收益
        dt_eps	float	Y	稀释每股收益
        total_revenue_ps	float	Y	每股营业总收入
        revenue_ps	float	Y	每股营业收入
        capital_rese_ps	float	Y	每股资本公积
        surplus_rese_ps	float	Y	每股盈余公积
        undist_profit_ps	float	Y	每股未分配利润
        extra_item	float	Y	非经常性损益
        profit_dedt	float	Y	扣除非经常性损益后的净利润
        gross_margin	float	Y	毛利
        current_ratio	float	Y	流动比率
        quick_ratio	float	Y	速动比率
        cash_ratio	float	Y	保守速动比率
        invturn_days	float	N	存货周转天数
        arturn_days	float	N	应收账款周转天数
        inv_turn	float	N	存货周转率
        ar_turn	float	Y	应收账款周转率
        ca_turn	float	Y	流动资产周转率
        fa_turn	float	Y	固定资产周转率
        assets_turn	float	Y	总资产周转率
        op_income	float	Y	经营活动净收益
        valuechange_income	float	N	价值变动净收益
        interst_income	float	N	利息费用
        daa	float	N	折旧与摊销
        ebit	float	Y	息税前利润
        ebitda	float	Y	息税折旧摊销前利润
        fcff	float	Y	企业自由现金流量
        fcfe	float	Y	股权自由现金流量
        current_exint	float	Y	无息流动负债
        noncurrent_exint	float	Y	无息非流动负债
        interestdebt	float	Y	带息债务
        netdebt	float	Y	净债务
        tangible_asset	float	Y	有形资产
        working_capital	float	Y	营运资金
        networking_capital	float	Y	营运流动资本
        invest_capital	float	Y	全部投入资本
        retained_earnings	float	Y	留存收益
        diluted2_eps	float	Y	期末摊薄每股收益
        bps	float	Y	每股净资产
        ocfps	float	Y	每股经营活动产生的现金流量净额
        retainedps	float	Y	每股留存收益
        cfps	float	Y	每股现金流量净额
        ebit_ps	float	Y	每股息税前利润
        fcff_ps	float	Y	每股企业自由现金流量
        fcfe_ps	float	Y	每股股东自由现金流量
        netprofit_margin	float	Y	销售净利率
        grossprofit_margin	float	Y	销售毛利率
        cogs_of_sales	float	Y	销售成本率
        expense_of_sales	float	Y	销售期间费用率
        profit_to_gr	float	Y	净利润/营业总收入
        saleexp_to_gr	float	Y	销售费用/营业总收入
        adminexp_of_gr	float	Y	管理费用/营业总收入
        finaexp_of_gr	float	Y	财务费用/营业总收入
        impai_ttm	float	Y	资产减值损失/营业总收入
        gc_of_gr	float	Y	营业总成本/营业总收入
        op_of_gr	float	Y	营业利润/营业总收入
        ebit_of_gr	float	Y	息税前利润/营业总收入
        roe	float	Y	净资产收益率
        roe_waa	float	Y	加权平均净资产收益率
        roe_dt	float	Y	净资产收益率(扣除非经常损益)
        roa	float	Y	总资产报酬率
        npta	float	Y	总资产净利润
        roic	float	Y	投入资本回报率
        roe_yearly	float	Y	年化净资产收益率
        roa2_yearly	float	Y	年化总资产报酬率
        roe_avg	float	N	平均净资产收益率(增发条件)
        opincome_of_ebt	float	N	经营活动净收益/利润总额
        investincome_of_ebt	float	N	价值变动净收益/利润总额
        n_op_profit_of_ebt	float	N	营业外收支净额/利润总额
        tax_to_ebt	float	N	所得税/利润总额
        dtprofit_to_profit	float	N	扣除非经常损益后的净利润/净利润
        salescash_to_or	float	N	销售商品提供劳务收到的现金/营业收入
        ocf_to_or	float	N	经营活动产生的现金流量净额/营业收入
        ocf_to_opincome	float	N	经营活动产生的现金流量净额/经营活动净收益
        capitalized_to_da	float	N	资本支出/折旧和摊销
        debt_to_assets	float	Y	资产负债率
        assets_to_eqt	float	Y	权益乘数
        dp_assets_to_eqt	float	Y	权益乘数(杜邦分析)
        ca_to_assets	float	Y	流动资产/总资产
        nca_to_assets	float	Y	非流动资产/总资产
        tbassets_to_totalassets	float	Y	有形资产/总资产
        int_to_talcap	float	Y	带息债务/全部投入资本
        eqt_to_talcapital	float	Y	归属于母公司的股东权益/全部投入资本
        currentdebt_to_debt	float	Y	流动负债/负债合计
        longdeb_to_debt	float	Y	非流动负债/负债合计
        ocf_to_shortdebt	float	Y	经营活动产生的现金流量净额/流动负债
        debt_to_eqt	float	Y	产权比率
        eqt_to_debt	float	Y	归属于母公司的股东权益/负债合计
        eqt_to_interestdebt	float	Y	归属于母公司的股东权益/带息债务
        tangibleasset_to_debt	float	Y	有形资产/负债合计
        tangasset_to_intdebt	float	Y	有形资产/带息债务
        tangibleasset_to_netdebt	float	Y	有形资产/净债务
        ocf_to_debt	float	Y	经营活动产生的现金流量净额/负债合计
        ocf_to_interestdebt	float	N	经营活动产生的现金流量净额/带息债务
        ocf_to_netdebt	float	N	经营活动产生的现金流量净额/净债务
        ebit_to_interest	float	N	已获利息倍数(EBIT/利息费用)
        longdebt_to_workingcapital	float	N	长期债务与营运资金比率
        ebitda_to_debt	float	N	息税折旧摊销前利润/负债合计
        turn_days	float	Y	营业周期
        roa_yearly	float	Y	年化总资产净利率
        roa_dp	float	Y	总资产净利率(杜邦分析)
        fixed_assets	float	Y	固定资产合计
        profit_prefin_exp	float	N	扣除财务费用前营业利润
        non_op_profit	float	N	非营业利润
        op_to_ebt	float	N	营业利润／利润总额
        nop_to_ebt	float	N	非营业利润／利润总额
        ocf_to_profit	float	N	经营活动产生的现金流量净额／营业利润
        cash_to_liqdebt	float	N	货币资金／流动负债
        cash_to_liqdebt_withinterest	float	N	货币资金／带息流动负债
        op_to_liqdebt	float	N	营业利润／流动负债
        op_to_debt	float	N	营业利润／负债合计
        roic_yearly	float	N	年化投入资本回报率
        total_fa_trun	float	N	固定资产合计周转率
        profit_to_op	float	Y	利润总额／营业收入
        q_opincome	float	N	经营活动单季度净收益
        q_investincome	float	N	价值变动单季度净收益
        q_dtprofit	float	N	扣除非经常损益后的单季度净利润
        q_eps	float	N	每股收益(单季度)
        q_netprofit_margin	float	N	销售净利率(单季度)
        q_gsprofit_margin	float	N	销售毛利率(单季度)
        q_exp_to_sales	float	N	销售期间费用率(单季度)
        q_profit_to_gr	float	N	净利润／营业总收入(单季度)
        q_saleexp_to_gr	float	Y	销售费用／营业总收入 (单季度)
        q_adminexp_to_gr	float	N	管理费用／营业总收入 (单季度)
        q_finaexp_to_gr	float	N	财务费用／营业总收入 (单季度)
        q_impair_to_gr_ttm	float	N	资产减值损失／营业总收入(单季度)
        q_gc_to_gr	float	Y	营业总成本／营业总收入 (单季度)
        q_op_to_gr	float	N	营业利润／营业总收入(单季度)
        q_roe	float	Y	净资产收益率(单季度)
        q_dt_roe	float	Y	净资产单季度收益率(扣除非经常损益)
        q_npta	float	Y	总资产净利润(单季度)
        q_opincome_to_ebt	float	N	经营活动净收益／利润总额(单季度)
        q_investincome_to_ebt	float	N	价值变动净收益／利润总额(单季度)
        q_dtprofit_to_profit	float	N	扣除非经常损益后的净利润／净利润(单季度)
        q_salescash_to_or	float	N	销售商品提供劳务收到的现金／营业收入(单季度)
        q_ocf_to_sales	float	Y	经营活动产生的现金流量净额／营业收入(单季度)
        q_ocf_to_or	float	N	经营活动产生的现金流量净额／经营活动净收益(单季度)
        basic_eps_yoy	float	Y	基本每股收益同比增长率(%)
        dt_eps_yoy	float	Y	稀释每股收益同比增长率(%)
        cfps_yoy	float	Y	每股经营活动产生的现金流量净额同比增长率(%)
        op_yoy	float	Y	营业利润同比增长率(%)
        ebt_yoy	float	Y	利润总额同比增长率(%)
        netprofit_yoy	float	Y	归属母公司股东的净利润同比增长率(%)
        dt_netprofit_yoy	float	Y	归属母公司股东的净利润-扣除非经常损益同比增长率(%)
        ocf_yoy	float	Y	经营活动产生的现金流量净额同比增长率(%)
        roe_yoy	float	Y	净资产收益率(摊薄)同比增长率(%)
        bps_yoy	float	Y	每股净资产相对年初增长率(%)
        assets_yoy	float	Y	资产总计相对年初增长率(%)
        eqt_yoy	float	Y	归属母公司的股东权益相对年初增长率(%)
        tr_yoy	float	Y	营业总收入同比增长率(%)
        or_yoy	float	Y	营业收入同比增长率(%)
        q_gr_yoy	float	N	营业总收入同比增长率(%)(单季度)
        q_gr_qoq	float	N	营业总收入环比增长率(%)(单季度)
        q_sales_yoy	float	Y	营业收入同比增长率(%)(单季度)
        q_sales_qoq	float	N	营业收入环比增长率(%)(单季度)
        q_op_yoy	float	N	营业利润同比增长率(%)(单季度)
        q_op_qoq	float	Y	营业利润环比增长率(%)(单季度)
        q_profit_yoy	float	N	净利润同比增长率(%)(单季度)
        q_profit_qoq	float	N	净利润环比增长率(%)(单季度)
        q_netprofit_yoy	float	N	归属母公司股东的净利润同比增长率(%)(单季度)
        q_netprofit_qoq	float	N	归属母公司股东的净利润环比增长率(%)(单季度)
        equity_yoy	float	Y	净资产同比增长率
        rd_exp	float	N	研发费用
        update_flag	str	N	更新标识
        '''
        
        fields = '''ts_code,
        ann_date,
        end_date,
        eps,
        dt_eps,
        total_revenue_ps,
        revenue_ps,
        capital_rese_ps,
        surplus_rese_ps,
        undist_profit_ps,
        extra_item,
        profit_dedt,
        gross_margin,
        current_ratio,
        quick_ratio,
        cash_ratio,
        invturn_days,
        arturn_days,
        inv_turn,
        ar_turn,
        ca_turn,
        fa_turn,
        assets_turn,
        op_income,
        valuechange_income,
        interst_income,
        daa,
        ebit,
        ebitda,
        fcff,
        fcfe,
        current_exint,
        noncurrent_exint,
        interestdebt,
        netdebt,
        tangible_asset,
        working_capital,
        networking_capital,
        invest_capital,
        retained_earnings,
        diluted2_eps,
        bps,
        ocfps,
        retainedps,
        cfps,
        ebit_ps,
        fcff_ps,
        fcfe_ps,
        netprofit_margin,
        grossprofit_margin,
        cogs_of_sales,
        expense_of_sales,
        profit_to_gr,
        saleexp_to_gr,
        adminexp_of_gr,
        finaexp_of_gr,
        impai_ttm,
        gc_of_gr,
        op_of_gr,
        ebit_of_gr,
        roe,
        roe_waa,
        roe_dt,
        roa,
        npta,
        roic,
        roe_yearly,
        roa2_yearly,
        roe_avg,
        opincome_of_ebt,
        investincome_of_ebt,
        n_op_profit_of_ebt,
        tax_to_ebt,
        dtprofit_to_profit,
        salescash_to_or,
        ocf_to_or,
        ocf_to_opincome,
        capitalized_to_da,
        debt_to_assets,
        assets_to_eqt,
        dp_assets_to_eqt,
        ca_to_assets,
        nca_to_assets,
        tbassets_to_totalassets,
        int_to_talcap,
        eqt_to_talcapital,
        currentdebt_to_debt,
        longdeb_to_debt,
        ocf_to_shortdebt,
        debt_to_eqt,
        eqt_to_debt,
        eqt_to_interestdebt,
        tangibleasset_to_debt,
        tangasset_to_intdebt,
        tangibleasset_to_netdebt,
        ocf_to_debt,
        ocf_to_interestdebt,
        ocf_to_netdebt,
        ebit_to_interest,
        longdebt_to_workingcapital,
        ebitda_to_debt,
        turn_days,
        roa_yearly,
        roa_dp,
        fixed_assets,
        profit_prefin_exp,
        non_op_profit,
        op_to_ebt,
        nop_to_ebt,
        ocf_to_profit,
        cash_to_liqdebt,
        cash_to_liqdebt_withinterest,
        op_to_liqdebt,
        op_to_debt,
        roic_yearly,
        total_fa_trun,
        profit_to_op,
        q_opincome,
        q_investincome,
        q_dtprofit,
        q_eps,
        q_netprofit_margin,
        q_gsprofit_margin,
        q_exp_to_sales,
        q_profit_to_gr,
        q_saleexp_to_gr,
        q_adminexp_to_gr,
        q_finaexp_to_gr,
        q_impair_to_gr_ttm,
        q_gc_to_gr,
        q_op_to_gr,
        q_roe,
        q_dt_roe,
        q_npta,
        q_opincome_to_ebt,
        q_investincome_to_ebt,
        q_dtprofit_to_profit,
        q_salescash_to_or,
        q_ocf_to_sales,
        q_ocf_to_or,
        basic_eps_yoy,
        dt_eps_yoy,
        cfps_yoy,
        op_yoy,
        ebt_yoy,
        netprofit_yoy,
        dt_netprofit_yoy,
        ocf_yoy,
        roe_yoy,
        bps_yoy,
        assets_yoy,
        eqt_yoy,
        tr_yoy,
        or_yoy,
        q_gr_yoy,
        q_gr_qoq,
        q_sales_yoy,
        q_sales_qoq,
        q_op_yoy,
        q_op_qoq,
        q_profit_yoy,
        q_profit_qoq,
        q_netprofit_yoy,
        q_netprofit_qoq,
        equity_yoy,
        rd_exp'''


        data = self.api.fina_indicator(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date,
        field=fields)
        return data
    
    def getFinacialAudit(self):
        '''# 获取上市公司定期财务审计意见数据
        
        输入参数

        名称	类型	必选	描述

        ts_code	str	Y	股票代码
        ann_date	str	N	公告日期
        start_date	str	N	公告开始日期
        end_date	str	N	公告结束日期
        period	str	N	报告期(每个季度最后一天的日期,比如20171231表示年报)
        
        输出参数

        名称	类型	描述

        ts_code	str	TS股票代码
        ann_date	str	公告日期
        end_date	str	报告期
        audit_result	str	审计结果
        audit_fees	float	审计总费用（元）
        audit_agency	str	会计事务所
        audit_sign	str	签字会计师'''
        data = self.api.fina_audit(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date)
        return data
    
    def getFinacialMain(self):
        '''# 获得上市公司主营业务构成
        
        输入参数

        名称	类型	必选	描述

        ts_code	str	Y	股票代码
        period	str	N	报告期(每个季度最后一天的日期,比如20171231表示年报)
        type	str	N	类型：P按产品 D按地区（请输入大写字母P或者D）
        start_date	str	N	报告期开始日期
        end_date	str	N	报告期结束日期
        
        输出参数

        名称	类型	描述

        ts_code	str	TS代码
        end_date	str	报告期
        bz_item	str	主营业务来源
        bz_sales	float	主营业务收入(元)
        bz_profit	float	主营业务利润(元)
        bz_cost	float	主营业务成本(元)
        curr_type	str	货币代码
        update_flag	str	是否更新
        '''
        data = self.api.fina_mainbz(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date)
        return data


class Market(Stock, Trade):
    '''# 市场参考数据'''
    def __init__(self, para):
        self.api = para.pro
        Stock.__init__(self, ts_code=para.ts_code)
        Trade.__init__(self,
                        start_date=para.start_date, 
                        end_date=para.end_date, 
                        trade_date=para.trade_date)    

    def getMoneyflow_HSGT(self):
        ''' # 获取沪股通、深股通、港股通每日资金流向数据
        
        输入参数

        名称	类型	必选	描述

        trade_date	str	N	交易日期 (二选一)
        start_date	str	N	开始日期 (二选一)
        end_date	str	N	结束日期

        输出参数

        名称	类型	描述

        trade_date	str	交易日期
        ggt_ss	str	港股通（上海）
        ggt_sz	str	港股通（深圳）
        hgt	str	沪股通（百万元）
        sgt	str	深股通（百万元）
        north_money	str	北向资金（百万元）
        south_money	str	南向资金（百万元）'''
        data = self.api.moneyflow_hsgt(start_date=self.start_date,
        end_date=self.end_date)
        return data

    def getMargin(self):
        ''' # 获取融资融券每日交易汇总数据
        
        输入参数

        名称	类型	必选	描述

        trade_date	str	N	交易日期
        exchange_id	str	N	交易所代码
        start_date	str	N	开始日期
        end_date	str	N	结束日期

        输出参数

        名称	类型	描述

        trade_date	str	交易日期
        exchange_id	str	交易所代码（SSE上交所SZSE深交所）
        rzye	float	融资余额(元)
        rzmre	float	融资买入额(元)
        rzche	float	融资偿还额(元)
        rqye	float	融券余额(元)
        rqmcl	float	融券卖出量(股,份,手)
        rzrqye	float	融资融券余额(元)'''
        data = self.api.margin(start_date=self.start_date,
        end_date=self.end_date)
        return data

    def getMarginDetail(self):
        '''# 获取融资融券每日交易详细数据
        
        输入参数

        名称	类型	必选	描述

        trade_date	str	N	交易日期
        ts_code	str	N	TS代码
        start_date	str	N	开始日期
        end_date	str	N	结束日期
        
        输出参数

        名称	类型	描述

        trade_date	str	交易日期
        ts_code	str	TS股票代码
        rzye	float	融资余额(元)
        rqye	float	融券余额(元)
        rzmre	float	融资买入额(元)
        rqyl	float	融券余量（手）
        rzche	float	融资偿还额(元)
        rqchl	float	融券偿还量(手)
        rqmcl	float	融券卖出量(股,份,手)
        rzrqye	float	融资融券余额(元)'''
        data = self.api.margin_detail(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date)
        return data

    def getPledgeState(self):
        '''# 获取股权质押统计数据
        
        输入参数

        名称	类型	必选	描述

        ts_code	str	Y	股票代码

        输出参数

        名称	类型	默认显示	描述

        ts_code	str	Y	TS代码
        end_date	str	Y	截至日期
        pledge_count	int	Y	质押次数
        unrest_pledge	float	Y	无限售股质押数量（万）
        rest_pledge	float	Y	限售股份质押数量（万）
        total_share	float	Y	总股本
        pledge_ratio	float	Y	质押比例'''
        data = self.api.pledge_stat(ts_code=self.ts_code)
        return data

    def getPledgeDetail(self):
        '''# 获取股权质押明细数据
        
        输入参数

        名称	类型	必选	描述

        ts_code	str	Y	股票代码

        输出参数

        名称	类型	默认显示	描述

        ts_code	str	Y	TS股票代码
        ann_date	str	Y	公告日期
        holder_name	str	Y	股东名称
        pledge_amount	float	Y	质押数量
        start_date	str	Y	质押开始日期
        end_date	str	Y	质押结束日期
        is_release	str	Y	是否已解押
        release_date	str	Y	解押日期
        pledgor	str	Y	质押方
        holding_amount	float	Y	持股总数
        pledged_amount	float	Y	质押总数
        p_total_ratio	float	Y	本次质押占总股本比例
        h_total_ratio	float	Y	持股总数占总股本比例
        is_buyback	str	Y	是否回购'''
        data = self.api.pledge_detail(ts_code=self.ts_code)
        return data

    def getRepurchase(self):
        '''# 获取上市公司回购股票数据
        
        输入参数

        名称	类型	必选	描述

        ann_date	str	N	公告日期（任意填参数，如果都不填，单次默认返回2000条）
        start_date	str	N	公告开始日期
        end_date	str	N	公告结束日期

        以上日期格式为：YYYYMMDD，比如20181010

        输出参数

        名称	类型	默认显示	描述

        ts_code	str	Y	TS代码
        ann_date	str	Y	公告日期
        end_date	str	Y	截止日期
        proc	str	Y	进度
        exp_date	str	Y	过期日期
        vol	float	Y	回购数量
        amount	float	Y	回购金额
        high_limit	float	Y	回购最高价
        low_limit	float	Y	回购最低价'''
        data = self.api.repurchase(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date)
        return data
        
    def getDesterilization(self):
        '''# 获取限售股解禁
        
        输入参数

        名称	类型	必选	描述

        ts_code	str	N	TS股票代码（至少输入一个参数）
        ann_date	str	N	公告日期（日期格式：YYYYMMDD，下同）
        float_date	str	N	解禁日期
        start_date	str	N	解禁开始日期
        end_date	str	N	解禁结束日期


        输出参数

        名称	类型	默认显示	描述

        ts_code	str	Y	TS代码
        ann_date	str	Y	公告日期
        float_date	str	Y	解禁日期
        float_share	float	Y	流通股份
        float_ratio	float	Y	流通股份占总股本比率
        holder_name	str	Y	股东名称
        share_type	str	Y	股份类型'''
        data = self.api.share_float(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date)
        return data

    def getBlockTrade(self):
        '''# 获取大宗交易
        
        输入参数

        名称	类型	必选	描述

        ts_code	str	N	TS代码（股票代码和日期至少输入一个参数）
        trade_date	str	N	交易日期（格式：YYYYMMDD，下同）
        start_date	str	N	开始日期
        end_date	str	N	结束日期


        输出参数

        名称	类型	默认显示	描述

        ts_code	str	Y	TS代码
        trade_date	str	Y	交易日历
        price	float	Y	成交价
        vol	float	Y	成交量（万股）
        amount	float	Y	成交金额
        buyer	str	Y	买方营业部
        seller	str	Y	卖方营业部'''
        data = self.api.block_trade(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date)
        return data

    def getStockHolder(self):
        '''# 获取上市公司增减持数据，了解重要股东近期及历史上的股份增减变化
        
        输入参数

        名称	类型	必选	描述
        ts_code	str	N	TS股票代码
        ann_date	str	N	公告日期
        start_date	str	N	公告开始日期
        end_date	str	N	公告结束日期
        trade_type	str	N	交易类型IN增持DE减持
        holder_type	str	N	股东类型C公司P个人G高管


        输出参数

        名称	类型	默认显示	描述
        ts_code	str	Y	TS代码
        ann_date	str	Y	公告日期
        holder_name	str	Y	股东名称
        holder_type	str	Y	股东类型G高管P个人C公司
        in_de	str	Y	类型IN增持DE减持
        change_vol	float	Y	变动数量
        change_ratio	float	Y	占流通比例（%）
        after_share	float	Y	变动后持股
        after_ratio	float	Y	变动后占流通比例（%）
        avg_price	float	Y	平均价格
        total_share	float	Y	持股总数
        begin_date	str	N	增减持开始日期
        close_date	str	N	增减持结束日期'''
        data = self.api.stk_holdertrade(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date)
        return data
    

class IndexData(Stock, Trade):
    
    '''
        指数数据
        市场说明(market)
    
    市场代码	说明
    MSCI	MSCI指数
    CSI	中证指数
    SSE	上交所指数
    SZSE	深交所指数
    CICC	中金所指数
    SW	申万指数
    OTH	其他指数
    
        指数列表

    主题指数、规模指数、策略指数、风格指数、综合指数、成长指数、
    价值指数、有色指数、化工指数、能源指数、其他指数、外汇指数、
    基金指数、商品指数、债券指数、行业指数、贵金属指数、
    农副产品指数、软商品指数、油脂油料指数、非金属建材指数、
    煤焦钢矿指数、谷物指数
    '''
    def __init__(self, pro, para):
        self.api = pro
        Stock.__init__(self, ts_code=para.ts_code)
        '''# 传入的是指数代码 不是股票代码'''
        Trade.__init__(self,
                        start_date=para.start_date, 
                        end_date=para.end_date, 
                        trade_date=para.trade_date,
                        market=para.market)

    def getIndexBasic(self, publisher=None, category=None):# 获取指数基本数据
        data = self.api.index_basic(market=self.market,
                                    publisher=publisher,
                                    category=category)
        return data

    def getIndexDaily(self):
        '''# 获取指数每日行情，还可以通过bar接口获取'''
        data = self.api.index_daily(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date)
        return data

    def getStockMarketIndex(self):
        '''# 上证综指指标数据'''
        data = self.api.index_daily(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date)
        return data


class FuturesData(Stock, Trade):
    '''# 期货数据'''
    def __init__(self, pro, para):
        self.api = pro
        Stock.__init__(self, ts_code=para.ts_code)
        '''# 传入的是期货代码 不是股票代码'''
        Trade.__init__(self,
                        start_date=para.start_date, 
                        end_date=para.end_date, 
                        trade_date=para.trade_date,
                        exchange=para.exchange)

    def getFuturesBasic(self, fut_type=None):
        '''# 获取指数基本数据'''
        data = self.api.fut_basic(exchange=self.exchange,
                                    fut_type=fut_type)
        return data

    def getFuturesDaily(self):
        '''# 获取期货日线'''
        data = self.api.fut_daily(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date)
        return data

    def getFuturesHolding(self):
        '''# 获取每日成交量 这里没有指定哪一个合约 因为在API中，合约代码用的是symbol，而不是ts_code'''
        data = self.api.fut_holding(
        start_date=self.start_date,
        end_date=self.end_date)
        return data

    def getFuturesWSR(self):
        '''# 获取仓单日报 这里没有指定哪一个合约 因为在API中，合约代码用的是symbol，而不是ts_code'''
        data = self.api.fut_wsr(
        start_date=self.start_date,
        end_date=self.end_date)
        return data


class Interest(Trade):
    '''# 利率以及借贷信息'''
    def __init__(self, para):
        self.api = para.pro
        Trade.__init__(self,
                        start_date=para.start_date, 
                        end_date=para.end_date, 
                        trade_date=para.trade_date
                        )

    def getShibor(self):
        '''# 上海银行间同业拆放利率'''
        data = self.api.shibor(
        start_date=self.start_date,
        end_date=self.end_date)
        return data

    def getShiborQuote(self):
        ''' # Shibor报价数据'''
        data = self.api.shibor_quote(
        start_date=self.start_date,
        end_date=self.end_date)
        return data

    def getShibor_LPR(self):
        '''# 贷款基础利率'''
        data = self.api.shibor_lpr(
        start_date=self.start_date,
        end_date=self.end_date)
        return data

    def getLibor(self):
        '''# Libor（London Interbank Offered Rate ），即伦敦同业拆借利率'''
        data = self.api.libor(
        start_date=self.start_date,
        end_date=self.end_date)
        return data

    def getHibor(self):
        '''# HIBOR (Hongkong InterBank Offered Rate)，是香港银行同行业拆借利率'''
        data = self.api.hibor(
        start_date=self.start_date,
        end_date=self.end_date)
        return data

    def getWenZhouIndex(self):
        '''# 温州指数 '''
        data = self.api.wz_index(
        start_date=self.start_date,
        end_date=self.end_date)
        return data


class News(Stock, Trade):
    '''# 新闻数据'''
    def __init__(self, para):
        self.api = para.pro
        Stock.__init__(self, ts_code=para.ts_code)
        Trade.__init__(self,
                        start_date=para.start_date, 
                        end_date=para.end_date, 
                        date=para.date
                        )
    
    def getNews(self, src=None):
        '''
        # 获取新闻资讯
        数据源

        来源名称	src标识	描述
        新浪财经	sina	获取新浪财经实时资讯
        华尔街见闻	wallstreetcn	华尔街见闻快讯
        同花顺	10jqka	同花顺财经新闻
        东方财富	eastmoney	东方财富财经新闻
        云财经	yuncaijing	云财经新闻
        '''
        data = self.api.news(src=src,
        start_date=self.start_date,
        end_date=self.end_date,fields='datetime, content, title, channels')
        return data

    def getCCTV(self):
        '''# 获取央视新闻'''
        data = self.api.cctv_news(date=self.date)
        return data

    def getCompanyPublic(self, year=2019):
        '''获取上市公司公报

        year年度，目前接口不能跨年获取，设定一个年份即可，比如2019'''
        data = self.api.news(ts_code=self.ts_code,
        start_date=self.start_date,
        end_date=self.end_date,
        year=year)
        return data


class General_API(Parameters):
    '''
    通用接口类：
        目前整合了股票（未复权、前复权、后复权）、指数、数字货币、ETF基金、期货、期权的行情数据，
        未来还将整合包括外汇在内的所有交易行情数据，同时提供分钟数据。
        由于这个通用接口直接在接口这一层返回数据，不需要在api传递到下一层
        所以这个类将直接返回数据
    输入参数：

        名称	类型	必选	描述
        ts_code	str	Y	证券代码
        api	str	N	pro版api对象，如果初始化了set_token，此参数可以不需要
        start_date	str	N	开始日期 (格式：YYYYMMDD)
        end_date	str	N	结束日期 (格式：YYYYMMDD)
        asset	str	Y	资产类别：E股票 I沪深指数 C数字货币 FT期货 FD基金 O期权，默认E
        adj	str	N	复权类型(只针对股票)：None未复权 qfq前复权 hfq后复权 , 默认None
        freq	str	Y	数据频度 ：支持分钟(min)/日(D)/周(W)/月(M)K线，其中1min表示1分钟（类推1/5/15/30/60分钟） ，默认D。
        ma	list	N	均线，支持任意合理int数值
        factors	list	N	股票因子（asset='E'有效）支持 tor换手率 vr量比
        adjfactor	str	N	复权因子，在复权数据是，如果此参数为True，返回的数据中则带复权因子，默认为False。

    输出指标

        具体输出的数据指标可参考各行情具体指标。

    '''
    def __init__(self, 
                asset='E', 
                adj=None,
                adjfactor=False,
                factors=['tor', 'vr'], 
                freq='1min', 
                ma=[7, 21],
                ts_code=None, 
                start_date=None, 
                end_date=None, 
                    ):
        '''
        初始化：
            资产类型为股票，数据频率为1分钟，股票因子返回换手率和量比，
        '''


        with open('bin\\base\\token.tkn','r') as token:
            mytoken = token.readline().rstrip('\n')
            # print('Your token is ' + mytoken)
        ts.set_token(mytoken)
        self.asset = asset
        self.adj = adj
        self.adjfactor = adjfactor
        self.factors = factors
        self.freq = freq
        self.ma = ma
        self.ts_code = ts_code
        self.start_date = start_date
        self.end_date = end_date
        
    def getMinuteStock(self):
        pro_data = ts.pro_bar(ts_code=self.ts_code, 
                                start_date=self.start_date, 
                                end_date=self.end_date,
                                asset=self.asset,
                                factors=self.factors,
                                adj=self.adj,
                                adjfactor=self.adjfactor,
                                freq=self.freq,
                                ma=self.ma,
                                )
        return pro_data
