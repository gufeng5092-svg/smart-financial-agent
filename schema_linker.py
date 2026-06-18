"""
Schema Linking 模块
将自然语言问题中的实体精确映射到数据库表和字段
流程：
  1. 实体识别：从问题中提取公司名、时间段、财务指标
  2. 公司名解析：模糊匹配数据库中的 stock_code
  3. 指标映射：将中文指标名映射到具体表.字段
  4. 输出精简的 linked schema（只含相关表和字段）
"""
import re
import json
from db import execute_sql_safe

# 缓存所有公司简称，避免重复查询
_company_cache = None

def _get_all_companies():
    global _company_cache
    if _company_cache is None:
        df, _ = execute_sql_safe(
            "SELECT stock_code, short_name, company_name FROM company_info"
        )
        if df is not None and not df.empty:
            _company_cache = df.to_dict('records')
        else:
            _company_cache = []
    return _company_cache

# ─────────────────────────────────────────────
# 1. 静态指标映射表（中文名 → 表.字段）
# ─────────────────────────────────────────────
METRIC_MAP = {
    # 利润表
    "营业收入":         ("income_sheet", "operating_revenue"),
    "营业总收入":       ("core_performance_indicators_sheet", "total_revenue_indicator"),
    "营业成本":         ("income_sheet", "operating_cost"),
    "毛利润":           ("income_sheet", "gross_profit"),
    "销售费用":         ("income_sheet", "selling_expense"),
    "管理费用":         ("income_sheet", "admin_expense"),
    "研发费用":         ("income_sheet", "rd_expense"),
    "财务费用":         ("income_sheet", "financial_expense"),
    "利润总额":         ("income_sheet", "total_profit"),
    "净利润":           ("income_sheet", "net_profit"),
    "归母净利润":       ("income_sheet", "net_profit_attributable"),
    "扣非净利润":       ("core_performance_indicators_sheet", "net_profit_deducted"),
    "所得税":           ("income_sheet", "income_tax"),
    # 资产负债表
    "货币资金":         ("balance_sheet", "monetary_funds"),
    "应收账款":         ("balance_sheet", "accounts_receivable"),
    "存货":             ("balance_sheet", "inventory"),
    "固定资产":         ("balance_sheet", "fixed_assets"),
    "流动资产":         ("balance_sheet", "total_current_assets"),
    "非流动资产":       ("balance_sheet", "total_non_current_assets"),
    "总资产":           ("balance_sheet", "total_assets"),
    "资产总额":         ("balance_sheet", "total_assets"),
    "资产总计":         ("balance_sheet", "total_assets"),
    "短期借款":         ("balance_sheet", "short_term_borrowing"),
    "流动负债":         ("balance_sheet", "total_current_liabilities"),
    "非流动负债":       ("balance_sheet", "total_non_current_liabilities"),
    "负债总额":         ("balance_sheet", "total_liabilities"),
    "负债合计":         ("balance_sheet", "total_liabilities"),
    "未分配利润":       ("balance_sheet", "undistributed_profit"),
    "所有者权益":       ("balance_sheet", "total_equity"),
    "股东权益":         ("balance_sheet", "total_equity"),
    "应付账款":         ("balance_sheet", "accounts_payable"),
    # 现金流量表
    "经营现金流":       ("cash_flow_sheet", "net_operating_cash_flow"),
    "经营性现金流量净额": ("cash_flow_sheet", "net_operating_cash_flow"),
    "投资现金流":       ("cash_flow_sheet", "net_investing_cash_flow"),
    "投资性现金流量净额": ("cash_flow_sheet", "net_investing_cash_flow"),
    "筹资现金流":       ("cash_flow_sheet", "net_financing_cash_flow"),
    "期末现金":         ("cash_flow_sheet", "ending_cash"),
    # 核心业绩指标表
    "基本每股收益":     ("core_performance_indicators_sheet", "basic_eps"),
    "稀释每股收益":     ("core_performance_indicators_sheet", "diluted_eps"),
    "加权平均净资产收益率": ("core_performance_indicators_sheet", "weighted_roe_indicator"),
    "加权平均净资产收益率（扣非）": ("core_performance_indicators_sheet", "weighted_roe_indicator"),
    "ROE":              ("core_performance_indicators_sheet", "weighted_roe_indicator"),
    "扣非ROE":          ("core_performance_indicators_sheet", "weighted_roe_indicator"),
    "每股净资产":       ("core_performance_indicators_sheet", "net_assets_per_share"),
    "每股经营现金流":   ("core_performance_indicators_sheet", "net_cash_per_share"),
    "净资产":           ("core_performance_indicators_sheet", "net_assets_indicator"),
    # 公司信息
    "股票代码":         ("company_info", "stock_code"),
    "公司简称":         ("company_info", "short_name"),
    "公司名称":         ("company_info", "company_name"),
    "行业":             ("company_info", "industry"),
}

# 同义词扩展
SYNONYMS = {
    "主营业务收入": "营业收入",
    "营收": "营业总收入",
    "收入": "营业总收入",
    "利润": "净利润",
    "盈利": "净利润",
    "亏损": "净利润",
    "研发支出": "研发费用",
    "销售成本": "营业成本",
    "资产": "总资产",
    "负债": "负债总额",
    "权益": "所有者权益",
    "现金流": "经营现金流",
    "经营现金流净额": "经营现金流",
    "投资现金流净额": "投资现金流",
    "扣非": "扣非净利润",
    "加权ROE": "加权平均净资产收益率",
    "净资产收益率": "加权平均净资产收益率",
}

# 时间段关键词 → (report_year, report_period)
# 数据库实际格式：report_year INT, report_period VARCHAR ('一季报'/'半年报'/'三季报'/'年报')
PERIOD_MAP = {
    "2025年第三季度": (2025, "三季报"),
    "2025年三季报":   (2025, "三季报"),
    "2025三季报":     (2025, "三季报"),
    "2025年半年报":   (2025, "半年报"),
    "2025年中报":     (2025, "半年报"),
    "2025年第二季度": (2025, "半年报"),
    "2025年一季报":   (2025, "一季报"),
    "2025年第一季度": (2025, "一季报"),
    "2025年全年":     (2025, "年报"),
    "2025年年报":     (2025, "年报"),
    "2024年第三季度": (2024, "三季报"),
    "2024年三季报":   (2024, "三季报"),
    "2024三季报":     (2024, "三季报"),
    "2024年半年报":   (2024, "半年报"),
    "2024年中报":     (2024, "半年报"),
    "2024年第二季度": (2024, "半年报"),
    "2024年一季报":   (2024, "一季报"),
    "2024年第一季度": (2024, "一季报"),
    "2024年全年":     (2024, "年报"),
    "2024年年报":     (2024, "年报"),
    "2023年第三季度": (2023, "三季报"),
    "2023年三季报":   (2023, "三季报"),
    "2023年全年":     (2023, "年报"),
    "2023年年报":     (2023, "年报"),
    "2022年全年":     (2022, "年报"),
    "2022年年报":     (2022, "年报"),
    "去年":           (2024, "年报"),
    "今年":           (2025, "三季报"),
    "前三季度":       (2025, "三季报"),
    "近几年":         None,
    "近3年":          None,
    "近四年":         None,
}

# 完整的表结构（精简版，用于SQL生成）
TABLE_SCHEMAS = {
    "company_info": {
        "desc": "公司信息表",
        "pk": "stock_code",
        "cols": {
            "stock_code": "股票代码",
            "company_name": "公司全称",
            "short_name": "公司简称",
            "industry": "行业",
        }
    },
    "income_sheet": {
        "desc": "利润表",
        "pk": "stock_code, report_year, report_period",
        "cols": {
            "stock_code": "股票代码",
            "report_year": "报告年度",
            "report_period": "报告期('一季报'/'半年报'/'三季报'/'年报')",
            "operating_revenue": "营业收入(万元)",
            "operating_cost": "营业成本(万元)",
            "gross_profit": "毛利润(万元)",
            "selling_expenses": "销售费用(万元，注意：实际有数据的字段是selling_expense)",
            "management_expenses": "管理费用(万元，注意：实际有数据的字段是admin_expense)",
            "rd_expenses": "研发费用(万元，注意：此字段为NULL，必须用rd_expense)",
            "rd_expense": "研发费用(万元，有数据，优先使用此字段)",
            "financial_expenses": "财务费用(万元，注意：实际有数据的字段是financial_expense)",
            "financial_expense": "财务费用(万元，有数据，优先使用此字段)",
            "total_profit": "利润总额(万元)",
            "net_profit": "净利润(万元)",
            "net_profit_attributable": "归母净利润(万元)",
            "income_tax": "所得税(万元)",
        }
    },
    "balance_sheet": {
        "desc": "资产负债表",
        "pk": "stock_code, report_year, report_period",
        "cols": {
            "stock_code": "股票代码",
            "report_year": "报告年度",
            "report_period": "报告期",
            "monetary_funds": "货币资金(万元)",
            "accounts_receivable": "应收账款(万元)",
            "inventory": "存货(万元)",
            "fixed_assets": "固定资产(万元)",
            "total_current_assets": "流动资产合计(万元)",
            "total_non_current_assets": "非流动资产合计(万元)",
            "total_assets": "资产总计(万元)",
            "short_term_borrowing": "短期借款(万元)",
            "total_current_liabilities": "流动负债合计(万元)",
            "total_non_current_liabilities": "非流动负债合计(万元)",
            "total_liabilities": "负债合计(万元)",
            "undistributed_profit": "未分配利润(万元)",
            "total_equity": "所有者权益合计(万元)",
            "accounts_payable": "应付账款(万元)",
        }
    },
    "cash_flow_sheet": {
        "desc": "现金流量表",
        "pk": "stock_code, report_year, report_period",
        "cols": {
            "stock_code": "股票代码",
            "report_year": "报告年度",
            "report_period": "报告期",
            "net_operating_cash_flow": "经营活动现金流量净额(万元)",
            "net_investing_cash_flow": "投资活动现金流量净额(万元)",
            "net_financing_cash_flow": "筹资活动现金流量净额(万元)",
            "ending_cash": "期末现金(万元)",
        }
    },
    "core_performance_indicators_sheet": {
        "desc": "核心业绩指标表",
        "pk": "stock_code, report_year, report_period",
        "cols": {
            "stock_code": "股票代码",
            "report_year": "报告年度",
            "report_period": "报告期",
            "total_revenue_indicator": "营业总收入(万元)",
            "net_profit_indicator": "净利润(万元)",
            "net_profit_deducted": "扣非净利润(万元)",
            "basic_eps": "基本每股收益(元)",
            "diluted_eps": "稀释每股收益(元)",
            "weighted_roe_indicator": "加权平均净资产收益率(%，含扣非，唯一ROE字段，禁止使用weighted_roe或roe_deducted)",
            "net_cash_per_share": "每股经营现金流(元)",
            "net_assets_per_share": "每股净资产(元)",
            "net_assets_indicator": "净资产(万元)",
        }
    },
}


def resolve_synonyms(text: str) -> str:
    """将同义词替换为标准名"""
    for syn, std in SYNONYMS.items():
        text = text.replace(syn, std)
    return text


def extract_periods(text: str):
    """
    从文本中提取时间段
    返回: [(year, period_str), ...] 或 None（多年范围）
    例如: [(2025, '三季报')] 或 None
    """
    # 先匹配长的关键词
    for kw, val in sorted(PERIOD_MAP.items(), key=lambda x: -len(x[0])):
        if kw in text:
            if val is None:
                return None  # 多期，不限定
            return [val]  # val = (year, period_str)

    # 提取年份范围，如"2022年至2025年"或"2022-2025年"
    # 同时匹配 "2025年" 和 "2025" 两种写法
    year_range = re.findall(r'(\d{4})年?', text)
    # 去重，避免同一年份被匹配两次
    year_range = list(dict.fromkeys(year_range))
    if len(year_range) >= 2:
        # 判断是否同时指定了季报类型（如"2022-2025年三季报"）
        if '第三季度' in text or '三季报' in text:
            return [(int(y), '三季报') for y in year_range]
        elif '第二季度' in text or '半年报' in text or '中报' in text:
            return [(int(y), '半年报') for y in year_range]
        elif '第一季度' in text or '一季报' in text:
            return [(int(y), '一季报') for y in year_range]
        elif '全年' in text or '年报' in text:
            return [(int(y), '年报') for y in year_range]
        return None  # 多年范围，未指定季报类型

    # 单独年份 + 无明确季度 → 不限定period
    if len(year_range) == 1:
        year = int(year_range[0])
        if '第三季度' in text or '三季报' in text:
            return [(year, '三季报')]
        elif '第二季度' in text or '半年报' in text or '中报' in text:
            return [(year, '半年报')]
        elif '第一季度' in text or '一季报' in text:
            return [(year, '一季报')]
        elif '全年' in text or '年报' in text:
            return [(year, '年报')]
        else:
            # 只有年份，不限定季度，返回 None 让 LLM 自己判断
            # 或者你想默认给年报：
            return [(year, '年报')]  # 按需改成 None

    return None


def extract_metrics(text: str) -> list:
    """从文本中提取财务指标，返回 [(中文名, 表名, 字段名)]"""
    text = resolve_synonyms(text)
    found = []
    seen_tables = set()
    for metric, (table, col) in sorted(METRIC_MAP.items(), key=lambda x: -len(x[0])):
        if metric in text:
            key = (table, col)
            if key not in seen_tables:
                found.append((metric, table, col))
                seen_tables.add(key)
    return found


def lookup_companies(names: list) -> list:
    """
    根据公司名列表模糊查询数据库，返回 [{stock_code, short_name, company_name}]
    """
    if not names:
        return []
    conditions = " OR ".join(
        [f"short_name LIKE '%{n}%' OR company_name LIKE '%{n}%'" for n in names]
    )
    sql = f"SELECT stock_code, short_name, company_name FROM company_info WHERE {conditions}"
    df, err = execute_sql_safe(sql)
    if err or df is None or df.empty:
        return []
    return df.to_dict('records')


def extract_company_names(text: str) -> list:
    """
    从文本中提取公司名/简称，优先用数据库缓存做精确匹配
    策略：
      1. 括号内6位股票代码 → 直接查库
      2. "企业名称：XXX" 模式
      3. 用数据库所有short_name/company_name在文本中做子串匹配（最准确）
    """
    # 1. 括号内股票代码
    codes = re.findall(r'[（(](\d{6})[）)]', text)
    if codes:
        code_list = "','".join(codes)
        sql = f"SELECT stock_code, short_name, company_name FROM company_info WHERE stock_code IN ('{code_list}')"
        df, _ = execute_sql_safe(sql)
        if df is not None and not df.empty:
            return df.to_dict('records')

    # 2. "企业名称：XXX" 模式
    explicit_names = []
    m = re.search(r'企业名称[：:]\s*([^\s，,。？?、]+)', text)
    if m:
        explicit_names.append(m.group(1).strip())

    # 3. 用数据库缓存做子串匹配（核心改进）
    all_companies = _get_all_companies()
    matched = {}  # stock_code → record，避免重复

    # 先用explicit_names精确匹配
    for name in explicit_names:
        for c in all_companies:
            sn = c.get('short_name') or ''
            cn = c.get('company_name') or ''
            if name in sn or name in cn or sn in name:
                matched[c['stock_code']] = c

    # 再用short_name在文本中做子串匹配（按长度降序，优先匹配长名称）
    sorted_companies = sorted(all_companies,
                               key=lambda x: len(x.get('short_name') or ''), reverse=True)
    for c in sorted_companies:
        sn = c.get('short_name') or ''
        cn = c.get('company_name') or ''
        # short_name子串匹配
        if len(sn) >= 2 and sn in text:
            matched[c['stock_code']] = c
        # company_name关键词匹配（去掉常见后缀后匹配）
        elif cn:
            # 提取company_name的核心词（去掉股份/有限/公司等）
            core = re.sub(r'(股份有限公司|有限责任公司|有限公司|股份|集团|医药|药业|制药)', '', cn)
            core = re.sub(r'^(广州市?|上海|北京|云南|山东|成都|浙江|湖南|四川|广东|福建)', '', core)
            if len(core) >= 2 and core in text:
                matched[c['stock_code']] = c

    # 处理特殊别名（如"999"→华润三九）
    ALIAS_MAP = {
        '999': '000999',
        '三九': '000999',
        '华润三九': '000999',
    }
    for alias, code in ALIAS_MAP.items():
        if alias in text:
            for c in all_companies:
                if c['stock_code'] == code:
                    matched[code] = c
                    break

    return list(matched.values())


def extract_industry(text: str) -> str | None:
    """
    从文本中提取行业关键词，返回行业名称字符串，未识别返回 None
    匹配 company_info.industry 列中存储的行业名称片段
    """
    # 常见行业关键词 → 数据库 industry 列中的匹配值（模糊匹配用）
    INDUSTRY_KEYWORDS = [
        "中药", "化学制药", "生物制品", "医疗器械", "医药商业",
        "银行", "保险", "证券", "房地产", "钢铁", "煤炭",
        "有色金属", "化工", "电力", "汽车", "电子", "半导体",
        "计算机", "通信", "传媒", "食品饮料", "农业", "纺织",
        "建筑", "交通运输", "零售", "旅游", "教育",
    ]
    for kw in INDUSTRY_KEYWORDS:
        if kw in text:
            return kw
    return None


def build_linked_schema(question: str, context: dict = None) -> dict:
    """
    Schema Linking 主函数
    输入：自然语言问题
    输出：{
        "companies": [{stock_code, short_name}],  # 涉及的公司
        "periods": [(2025, "三季报"), ...],       # 涉及的时间段
        "metrics": [(中文名, 表名, 字段名)],       # 涉及的指标
        "tables": {表名: {desc, pk, relevant_cols}},  # 精简的schema
        "linked_schema_text": str,                # 给LLM的精简schema文本
        "industry": str | None,                   # 识别到的行业关键词
    }
    """
    # 合并上下文
    full_text = question
    if context:
        full_text += " " + json.dumps(context, ensure_ascii=False)

    # 1. 提取公司
    companies = extract_company_names(full_text)

    # 2. 提取时间段
    periods = extract_periods(full_text)

    # 3. 提取指标
    metrics = extract_metrics(full_text)

    # 4. 提取行业关键词
    industry = extract_industry(question)  # 只从原始问题提取，不混入context

    # 5. 确定涉及的表
    involved_tables = set()
    involved_tables.add("company_info")  # 总是需要

    for _, table, _ in metrics:
        involved_tables.add(table)

    # 如果没有识别到指标，加入所有表（fallback）
    if len(metrics) == 0:
        involved_tables = set(TABLE_SCHEMAS.keys())

    # 行业查询时必须包含 income_sheet（计算毛利率/净利率需要）
    if industry:
        involved_tables.add("income_sheet")

    # 6. 构建精简schema文本
    schema_lines = ["【相关数据库表结构】"]
    schema_lines.append("重要：report_year是整数年份(2022/2023/2024/2025)，report_period是中文字符串('一季报'/'半年报'/'三季报'/'年报')，金额单位万元")
    schema_lines.append("")

    for tname in sorted(involved_tables):
        if tname not in TABLE_SCHEMAS:
            continue
        tinfo = TABLE_SCHEMAS[tname]
        schema_lines.append(f"表名: {tname}（{tinfo['desc']}）")
        schema_lines.append(f"  主键: {tinfo['pk']}")

        # 只列出相关字段（涉及的指标字段 + 主键字段）
        relevant_cols = {"stock_code", "report_year", "report_period"}
        if tname == "company_info":
            relevant_cols.update({"short_name", "company_name", "industry"})
        for _, t, col in metrics:
            if t == tname:
                relevant_cols.add(col)

        # 如果该表没有识别到具体字段，列出所有字段
        if len(relevant_cols) <= 3:
            relevant_cols = set(tinfo['cols'].keys())

        for col, desc in tinfo['cols'].items():
            if col in relevant_cols:
                schema_lines.append(f"  - {col}: {desc}")
        schema_lines.append("")

    # 6. 添加公司信息提示
    if companies:
        schema_lines.append("【已识别的公司（直接用stock_code过滤，无需模糊匹配）】")
        for c in companies:
            schema_lines.append(f"  stock_code='{c['stock_code']}' short_name='{c.get('short_name','')}' company_name='{c.get('company_name','')}'")
        schema_lines.append("")

    # 7. 添加时间段提示（用正确格式）
    if periods:
        # 判断是否为跨年同一报告期（如连续4年三季报）
        unique_periods = list(dict.fromkeys(pd_str for _, pd_str in periods))
        unique_years = [yr for yr, _ in periods]
        if len(unique_periods) == 1 and len(unique_years) > 1:
            # 跨年同一报告期：用 IN 语法提示
            years_str = ', '.join(str(y) for y in unique_years)
            schema_lines.append(f"【已识别的时间段】report_period='{unique_periods[0]}' AND report_year IN ({years_str})")
            schema_lines.append(f"  注意：这是跨年同一报告期查询，请用 report_year IN ({years_str}) 而非年报")
        else:
            period_hints = []
            for yr, pd_str in periods:
                period_hints.append(f"report_year={yr} AND report_period='{pd_str}'")
            schema_lines.append(f"【已识别的时间段】{' 或 '.join(period_hints)}")
        schema_lines.append("")
    else:
        # periods=None 表示多年范围且未指定季报类型，提示LLM只取年报做趋势对比
        schema_lines.append("【时间段说明】问题涉及多年趋势/近N年对比（未指定具体季报类型），请只查询 report_period='年报' 的数据，按 report_year 排序")
        schema_lines.append("")

    # 8. 添加指标提示
    if metrics:
        schema_lines.append("【已识别的财务指标映射】")
        for cn, table, col in metrics:
            schema_lines.append(f"  {cn} → {table}.{col}")
        schema_lines.append("")

    # 9. 添加行业过滤提示
    if industry:
        schema_lines.append(f"【行业过滤】问题涉及行业查询，行业字段在 company_info.industry，")
        schema_lines.append(f"  请用 company_info.industry LIKE '%{industry}%' 过滤行业内所有公司，")
        schema_lines.append(f"  不要假设存在 financial_indicators 等不在上方列出的表。")
        schema_lines.append(f"  行业均值计算方式：先对该行业所有公司计算各自指标值，再取 AVG()。")
        schema_lines.append("")

    return {
        "companies": companies,
        "periods": periods,
        "metrics": metrics,
        "tables": list(involved_tables),
        "linked_schema_text": "\n".join(schema_lines),
        "industry": industry,
    }
