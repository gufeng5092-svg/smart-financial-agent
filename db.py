"""数据库工具：连接、执行查询、获取schema"""
import pandas as pd
from sqlalchemy import create_engine, text
from config import DB_CONFIG

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        cfg = DB_CONFIG
        url = (f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
               f"@{cfg['host']}:{cfg.get('port', 3306)}/{cfg['database']}"
               f"?charset={cfg.get('charset', 'utf8mb4')}")
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def execute_query(sql: str) -> pd.DataFrame:
    """执行SQL，返回DataFrame"""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    return df


def execute_sql_safe(sql: str):
    """执行SQL，返回(df, error)"""
    try:
        df = execute_query(sql)
        return df, None
    except Exception as e:
        return None, str(e)


# 数据库schema描述，供LLM参考
DB_SCHEMA = """
包含以下6张表（report_period格式：'一季报'/'半年报'/'三季报'/'年报'）

1. company_info（公司信息表）
   - stock_code VARCHAR(10) PK: 股票代码
   - company_name VARCHAR(100): 公司全称
   - short_name VARCHAR(50): 公司简称
   - industry VARCHAR(50): 行业

2. income_sheet（利润表）
   联合主键: stock_code, report_year INT, report_period VARCHAR(20)
   - operating_revenue DECIMAL: 营业收入（万元）
   - operating_cost DECIMAL: 营业成本
   - gross_profit DECIMAL: 毛利润
    - selling_expense DECIMAL: 销售费用（有数据，禁用selling_expenses）
    - admin_expense DECIMAL: 管理费用（有数据，禁用management_expenses）
    - rd_expense DECIMAL: 研发费用（有数据，禁用rd_expenses）
    - financial_expense DECIMAL: 财务费用（有数据，禁用financial_expenses）
   - total_profit DECIMAL: 利润总额
   - net_profit DECIMAL: 净利润
   - net_profit_attributable DECIMAL: 归母净利润
   - tax_surcharges/operating_tax DECIMAL: 营业税金及附加
   - is_anomaly TINYINT: 异常标记(0正常/1异常)

3. balance_sheet（资产负债表）
   联合主键: stock_code, report_year INT, report_period VARCHAR(20)
   - monetary_funds DECIMAL: 货币资金
   - accounts_receivable DECIMAL: 应收账款
   - inventory DECIMAL: 存货
   - fixed_assets DECIMAL: 固定资产
   - total_current_assets DECIMAL: 流动资产合计
   - total_non_current_assets DECIMAL: 非流动资产合计
   - total_assets DECIMAL: 资产总计
   - short_term_borrowing/short_term_borrowings DECIMAL: 短期借款
   - total_current_liabilities DECIMAL: 流动负债合计
   - total_non_current_liabilities DECIMAL: 非流动负债合计
   - total_liabilities DECIMAL: 负债合计
   - undistributed_profit DECIMAL: 未分配利润
   - total_equity DECIMAL: 所有者权益合计
   - is_anomaly TINYINT: 异常标记

4. cash_flow_sheet（现金流量表）
   联合主键: stock_code, report_year INT, report_period VARCHAR(20)
   - net_operating_cash_flow DECIMAL: 经营活动现金流量净额
   - net_investing_cash_flow DECIMAL: 投资活动现金流量净额
   - net_financing_cash_flow DECIMAL: 筹资活动现金流量净额
   - ending_cash DECIMAL: 期末现金及现金等价物余额
   - is_anomaly TINYINT: 异常标记

5. core_performance_indicators_sheet（核心业绩指标表）
   联合主键: stock_code, report_year INT, report_period VARCHAR(20)
   - total_revenue_indicator DECIMAL: 营业总收入（万元）
   - net_profit_indicator DECIMAL: 净利润
   - net_profit_deducted/net_profit_deducted_indicator DECIMAL: 扣非净利润
   - basic_eps/basic_eps_indicator DECIMAL: 基本每股收益
   - diluted_eps/diluted_eps_indicator DECIMAL: 稀释每股收益
   - weighted_roe/weighted_roe_indicator DECIMAL: 加权平均净资产收益率
   - net_cash_per_share DECIMAL: 每股经营现金流
   - net_assets_per_share DECIMAL: 每股净资产
   - total_assets/total_assets_indicator DECIMAL: 总资产
   - net_assets_indicator DECIMAL: 净资产
   - is_anomaly TINYINT: 异常标记

6. subject_mapping（科目映射表）
   - standard_name VARCHAR(50): 标准科目名
   - synonyms JSON: 同义词列表
   - target_table VARCHAR(50): 对应表名
   - target_column VARCHAR(50): 对应字段名

重要说明：
- 营业总收入优先用 core_performance_indicators_sheet.total_revenue_indicator
- 扣非净利润用 core_performance_indicators_sheet.net_profit_deducted 或 net_profit_deducted_indicator
- report_period值：'一季报'、'半年报'、'三季报'、'年报'
- 金额单位均为万元
- 查询公司时可用 company_info JOIN 其他表，通过 stock_code 关联
- 查询公司名称时支持模糊匹配：WHERE c.short_name LIKE '%云南白药%' 或 c.company_name LIKE '%云南白药%'
"""


def get_schema() -> str:
    return DB_SCHEMA
