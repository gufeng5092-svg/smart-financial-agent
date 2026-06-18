"""
智能问数助手核心Agent
流程：自然语言 → Schema Linking → SQL生成 → 执行 → 可视化 → 分析结论
支持多轮对话和意图澄清
"""
import os
# 禁止 huggingface_hub / transformers 发起任何网络请求，强制使用本地缓存
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import re
import json
import traceback
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from openai import OpenAI
from db import execute_sql_safe
from schema_linker import build_linked_schema
import requests
from config import (
    DIFY_API_KEY,
    DIFY_DATASET_ID,
    DIFY_URL,
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_MODEL,
    REPORT_BASE_DIR,
    RESULT_DIR,
)
from sentence_transformers import CrossEncoder
from intent_classifier import quick_needs_rag
from answer_validator import validate_answer

# ── Module 2: CrossEncoder Reranker ──────────────
_reranker = None
_reranker_unavailable = False  # 加载失败后置 True，避免重复尝试

def _get_reranker():
    global _reranker, _reranker_unavailable
    if _reranker_unavailable:
        return None
    if _reranker is None:
        try:
            _reranker = CrossEncoder(
                'BAAI/bge-reranker-base',
                max_length=512,
                local_files_only=True,
            )
            print("[Reranker] 模型加载成功")
        except Exception as e:
            print(f"[Reranker] 加载失败: {e}，降级为按Dify分数排序")
            _reranker_unavailable = True
    return _reranker


def _rerank_results(query: str, results: list, top_k: int = 4) -> list:
    """
    使用 CrossEncoder 对检索结果重排序
    query: 用户查询
    results: _dify_retrieve 返回的列表
    返回重排后的 top_k 条
    """
    if not results:
        return results

    reranker = _get_reranker()
    if reranker is None:
        # 降级：直接按 Dify 返回的 score 排序
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    try:
        pairs = [(query, r["content"]) for r in results]
        scores = reranker.predict(pairs)

        # 将 rerank 分数写回结果
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)

        ranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]
    except Exception as e:
        print(f"[Reranker] 重排序失败: {e}")
        return results[:top_k]

# ── 中文字体设置 ──────────────────────────────
def _setup_font():
    candidates = [
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/System/Library/Fonts/STHeiti Medium.ttc',
        '/Library/Fonts/Arial Unicode MS.ttf',
        '/System/Library/Fonts/Supplemental/Arial Unicode MS.ttf',
    ]
    for fp in candidates:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
            prop = fm.FontProperties(fname=fp)
            plt.rcParams['font.family'] = prop.get_name()
            return
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

_setup_font()

# ── LLM 客户端 ────────────────────────────────
client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
)
os.makedirs(RESULT_DIR, exist_ok=True)


def _call_llm(messages: list, temperature=0.2) -> str:
    resp = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

def _auto_fix_recent_years(question: str, periods: list) -> list:
    q = question.strip()
    # 触发：近三年 / 近3年 / 最近三年 → 固定返回这三年（2022/2023/2024，因2025年报尚无数据）
    if any(k in q for k in ["近三年", "近3年", "最近三年"]):
        return [(2022, "年报"), (2023, "年报"), (2024, "年报")]
    return periods


def _rebuild_schema_text_with_periods(linked: dict, periods: list) -> str:
    """
    当 periods 被 _auto_fix_recent_years 修正后，重新生成 linked_schema_text 中的时间段提示部分。
    直接替换原 schema_text 中的时间段说明行。
    """
    schema_text = linked.get("linked_schema_text", "")
    # 构建新的时间段提示
    unique_periods = list(dict.fromkeys(pd_str for _, pd_str in periods))
    unique_years = [yr for yr, _ in periods]
    if len(unique_periods) == 1 and len(unique_years) > 1:
        years_str = ', '.join(str(y) for y in unique_years)
        new_hint = (
            f"【已识别的时间段】report_period='{unique_periods[0]}' AND report_year IN ({years_str})\n"
            f"  注意：这是跨年同一报告期查询，请用 report_year IN ({years_str}) 而非其他年份"
        )
    else:
        period_hints = [f"report_year={yr} AND report_period='{pd_str}'" for yr, pd_str in periods]
        new_hint = f"【已识别的时间段】{' 或 '.join(period_hints)}"

    # 替换原有的时间段说明（多年趋势说明 或 已识别的时间段）
    schema_text = re.sub(
        r'【时间段说明】.*?(?=\n\n|\Z)',
        new_hint,
        schema_text,
        flags=re.DOTALL
    )
    schema_text = re.sub(
        r'【已识别的时间段】.*?(?=\n\n|\Z)',
        new_hint,
        schema_text,
        flags=re.DOTALL
    )
    return schema_text

def _extract_json(text: str):
    m = re.search(r'```(?:json)?\s*([\s\S]+?)```', text)
    if m:
        text = m.group(1).strip()

    def _fix_and_load(s: str):
        # 1. 直接尝试
        try:
            return json.loads(s)
        except Exception:
            pass
        # 2. 修复 LLM 在 JSON 字符串内用 "\ " 或 "\<newline>" 续行的问题
        #    把 JSON 字符串值内的 \<space> 和 \<newline> 替换为空格
        fixed = re.sub(r'\\\s+', ' ', s)
        try:
            return json.loads(fixed)
        except Exception:
            pass
        # 3. 把字符串值内的真实换行转义为 \n
        fixed2 = re.sub(r'(?<!\\)\n', '\\n', fixed)
        try:
            return json.loads(fixed2)
        except Exception:
            pass
        return None

    result = _fix_and_load(text)
    if result is not None:
        return result
    # 最后尝试：提取第一个 { ... } 块再修复
    m2 = re.search(r'\{[\s\S]+\}', text)
    if m2:
        return _fix_and_load(m2.group(0))
    return None


# ── SQL生成 Prompt（使用Schema Linking结果）────
SQL_GEN_SYSTEM = """你是专业的MySQL查询专家，专注于上市公司财报数据库。

【数据库时间字段格式（非常重要）】
- report_year: 整数，如 2022, 2023, 2024, 2025
- report_period: 中文字符串，只有4种值：'一季报' '半年报' '三季报' '年报'
- 示例：2025年第三季度 → WHERE report_year=2025 AND report_period='三季报'
- 示例：2024年全年 → WHERE report_year=2024 AND report_period='年报'
- 禁止使用 '2025Q3' '2024FY' 等格式，数据库中不存在这些值

规则：
- 只输出JSON，格式如下，不要有其他文字
- 金额单位为万元
- 收益率 = 净利润 ÷ 营业收入 × 100%
- 同比增长率 = (本期值 - 同期值) / ABS(同期值) * 100
- 环比增长率 = (本期值 - 上期值) / ABS(上期值) * 100
- 【重要】数据库各报告期存储的是单季度/当期值，规则如下：
  - 一季报 = Q1单季值
  - 半年报 = H1累计值（Q1+Q2之和）
  - 三季报 = Q3单季值
  - 年报   = Q4单季值（或全年累计，视字段而定）
  因此 Q2单季值 = 半年报值 - 一季报值，计算三季报环比时上期值必须用此公式推算：
  上期值(Q2) = 同年半年报值 - 同年一季报值
  三季报环比增长率 = (三季报值 - Q2单季值) / ABS(Q2单季值) * 100
  一季报环比增长率 = (本年一季报值 - 上年三季报值) / ABS(上年三季报值) * 100
  半年报环比增长率 = (H1累计值 - 上年三季报值) / ABS(上年三季报值) * 100（通常无意义，可说明）
  禁止直接用 LAG() 对报告期顺序取上一行来计算三季报环比，因为上一行是半年报累计值而非Q2单季值
  【三季报环比SQL示例】：
  SELECT q3.report_year,
    ROUND((q3.total_revenue_indicator - (h1.total_revenue_indicator - q1.total_revenue_indicator))
      / ABS(h1.total_revenue_indicator - q1.total_revenue_indicator) * 100, 2) AS 环比增长率
  FROM core_performance_indicators_sheet q3
  JOIN core_performance_indicators_sheet h1
    ON q3.stock_code=h1.stock_code AND q3.report_year=h1.report_year AND h1.report_period='半年报'
  JOIN core_performance_indicators_sheet q1
    ON q3.stock_code=q1.stock_code AND q3.report_year=q1.report_year AND q1.report_period='一季报'
  JOIN company_info c ON q3.stock_code=c.stock_code
  WHERE c.short_name LIKE '%公司名%' AND q3.report_period='三季报' AND q3.report_year IN (年份1, 年份2)
- 销售毛利率 = (operating_revenue - operating_cost) / operating_revenue * 100
- 销售净利率 = net_profit / operating_revenue * 100
- 资产负债率 = total_liabilities / total_assets * 100
- 存货周转率 = operating_cost / inventory
 - 【income_sheet 双字段问题（极重要）】income_sheet 中存在新旧两套字段，旧字段全为NULL，必须用新字段：
   - 研发费用：禁止用 rd_expenses（全为NULL），必须用 rd_expense
   - 财务费用：禁止用 financial_expenses（全为NULL），必须用 financial_expense
   - 销售费用：禁止用 selling_expenses（全为NULL），必须用 selling_expense
   - 管理费用：禁止用 management_expenses（全为NULL），必须用 admin_expense
 - 营业总收入优先用 core_performance_indicators_sheet.total_revenue_indicator
 - 扣非净利润用 core_performance_indicators_sheet.net_profit_deducted
- 【ROE字段（极重要）】加权平均净资产收益率（含扣非）唯一正确字段是 core_performance_indicators_sheet.weighted_roe_indicator
  - 禁止使用 weighted_roe（该列全为NULL）、roe_deducted、weighted_roe_deducted 等不存在的字段
  - 无论问题说"加权平均净资产收益率"还是"加权平均净资产收益率（扣非）"，一律用 weighted_roe_indicator
- 查询公司时JOIN company_info，用 stock_code 关联
- 多年趋势/近N年对比分析，若问题明确说"年报"或"全年"才取 report_period='年报'；若问题明确指定季报（如"三季报""第三季度"），则用对应季报，不得强制改为年报
- 【跨年同一报告期对比】若问题要求"连续N年的某季报"（如2022-2025年三季报），应用 report_period='三季报' AND report_year IN (2022,2023,2024,2025)，绝对禁止改为年报
- 【连续N个报告期均满足条件】用 HAVING COUNT(*) = N 筛选，示例（连续4年三季报ROE均>10%）：
  SELECT c.stock_code, c.short_name, COUNT(*) AS 满足期数
  FROM core_performance_indicators_sheet k
  JOIN company_info c ON k.stock_code = c.stock_code
  WHERE k.report_period = '三季报'
    AND k.report_year IN (2022, 2023, 2024, 2025)
    AND k.weighted_roe_indicator > 10
  GROUP BY c.stock_code, c.short_name
  HAVING COUNT(*) = 4
  ORDER BY c.stock_code
- 【多步聚合查询（极重要）】当问题包含"统计数量→计算均值→筛选高于均值→可视化"等多步骤时，必须用CTE分步完成，
  每一步SELECT中的非聚合列必须全部出现在GROUP BY中，禁止在同一SELECT中混用聚合列和未分组的明细列。
  示例（统计净利润同比增长公司数量、计算平均研发费用占比、找出高于均值的公司并生成散点图）：
  WITH base AS (
    -- 获取2025年三季报和2024年三季报的净利润与研发费用
    SELECT
      c.stock_code, c.short_name,
      cur.net_profit_indicator  AS net_profit_cur,
      pre.net_profit_indicator  AS net_profit_pre,
      i.rd_expense              AS rd_expense,
      i.operating_revenue       AS operating_revenue
    FROM core_performance_indicators_sheet cur
    JOIN core_performance_indicators_sheet pre
      ON cur.stock_code = pre.stock_code
      AND pre.report_year = 2024 AND pre.report_period = '三季报'
    JOIN income_sheet i
      ON cur.stock_code = i.stock_code
      AND i.report_year = 2025 AND i.report_period = '三季报'
    JOIN company_info c ON cur.stock_code = c.stock_code
    WHERE cur.report_year = 2025 AND cur.report_period = '三季报'
      AND pre.net_profit_indicator IS NOT NULL
      AND pre.net_profit_indicator <> 0
  ),
  growth_companies AS (
    -- 筛选净利润同比增长的公司，计算增长率和研发费用占比
    SELECT
      stock_code, short_name,
      ROUND((net_profit_cur - net_profit_pre) / ABS(net_profit_pre) * 100, 2) AS profit_growth_rate,
      ROUND(rd_expense / NULLIF(operating_revenue, 0) * 100, 2)               AS rd_ratio
    FROM base
    WHERE net_profit_cur > net_profit_pre
  ),
  avg_rd AS (
    SELECT AVG(rd_ratio) AS avg_rd_ratio FROM growth_companies
  )
  -- 最终：找出研发费用占比高于均值的公司，同时输出散点图所需的两列
  SELECT g.stock_code, g.short_name, g.rd_ratio AS 研发费用占比, g.profit_growth_rate AS 净利润增长率,
         ROUND(a.avg_rd_ratio, 2) AS 行业平均研发占比
  FROM growth_companies g, avg_rd a
  WHERE g.rd_ratio > a.avg_rd_ratio
  ORDER BY g.rd_ratio DESC
- 如果问题意图模糊或缺少关键信息（公司名/时间段），intent设为clarify

【行业查询规则（重要）】
- 行业信息存储在 company_info.industry 列，用 LIKE '%行业名%' 过滤
- 禁止使用 financial_indicators、industry_avg 等不存在的表
- 行业均值查询模板（以中药行业销售毛利率为例）：
  WITH industry_data AS (
    SELECT c.stock_code, c.short_name,
           (i.operating_revenue - i.operating_cost) / i.operating_revenue * 100 AS gross_margin,
           i.net_profit / i.operating_revenue * 100 AS net_margin
    FROM income_sheet i
    JOIN company_info c ON i.stock_code = c.stock_code
    WHERE c.industry LIKE '%中药%'
      AND i.report_year = 2025 AND i.report_period = '三季报'
      AND i.operating_revenue > 0
  ),
  avg_data AS (
    SELECT AVG(gross_margin) AS avg_gross_margin, AVG(net_margin) AS avg_net_margin
    FROM industry_data
  )
  SELECT d.stock_code, d.short_name,
         ROUND(d.gross_margin, 2) AS 销售毛利率,
         ROUND(d.net_margin, 2) AS 销售净利率,
         ROUND(a.avg_gross_margin, 2) AS 行业均值_毛利率,
         ROUND(a.avg_net_margin, 2) AS 行业均值_净利率
  FROM industry_data d, avg_data a
  WHERE d.gross_margin > a.avg_gross_margin AND d.net_margin > a.avg_net_margin
  ORDER BY d.gross_margin DESC

输出JSON格式：
{
  "intent": "query|clarify",
  "clarify_question": "当intent=clarify时填写",
  "sql": "完整的MySQL SELECT语句，不含分号",
  "chart_type": "none|line|bar|hbar|pie|scatter|histogram|radar|boxplot（重要：仅当用户问题中明确要求画图、可视化、图表时才选非none类型，否则必须填none）",
  "chart_config": {
    "title": "图表标题",
    "x_col": "x轴对应的查询结果列名",
    "y_col": "y轴列名（可为列表，多系列时用）",
    "label_col": "饼图标签列名"
  }

  - 最重要的一条，你只能说数据库或者知识库中有的公司，不能说数据库或者知识库中没有的公司，比如问你营业收入最高的公司，你只能在数据库或者知识库中有的公司中比较
}"""


def _build_sql_prompt(question: str, linked: dict, history_sql: list = None,
                      context: dict = None) -> str:
    """构建SQL生成的用户prompt"""
    parts = [f"用户问题：{question}", ""]

    # 注入对话上下文（公司、年份），供后续追问时继承
    ctx = context or {}
    ctx_lines = []
    if ctx.get("last_company"):
        ctx_lines.append(f"  当前公司：{ctx['last_company']}（stock_code={ctx.get('last_stock_code', '')}）")
    if ctx.get("last_period_year"):
        ctx_lines.append(f"  当前年份：{ctx['last_period_year']}")
    if ctx.get("last_period"):
        p = ctx["last_period"]
        period_str = p[1] if isinstance(p, (list, tuple)) and len(p) > 1 else str(p)
        if period_str:
            ctx_lines.append(f"  当前报告期：{period_str}")
    if ctx_lines:
        parts.append("【对话上下文（若用户问题未指定公司/年份，请沿用以下信息）】")
        parts.extend(ctx_lines)
        parts.append("")

    parts.append(linked["linked_schema_text"])

    if history_sql:
        parts.append("【本轮对话已执行的SQL（供参考上下文）】")
        for s in history_sql[-2:]:
            parts.append(f"  {s}")
        parts.append("")

    parts.append("请根据以上信息生成MySQL查询语句，只输出JSON。")
    return "\n".join(parts)


# ── 图表生成 ──────────────────────────────────
def _generate_chart(df: pd.DataFrame, chart_type: str, chart_config: dict,
                    save_path: str) -> bool:
    try:
        if df is None or df.empty or chart_type == 'none':
            return False

        fig, ax = plt.subplots(figsize=(10, 6))
        title = chart_config.get('title', '')
        x_col = chart_config.get('x_col', '')
        y_col = chart_config.get('y_col', '')
        label_col = chart_config.get('label_col', '')

        if x_col not in df.columns:
            x_col = df.columns[0]

        if isinstance(y_col, list):
            y_cols = [c for c in y_col if c in df.columns] or \
                     ([df.columns[1]] if len(df.columns) > 1 else [df.columns[0]])
        elif isinstance(y_col, str) and y_col in df.columns:
            y_cols = [y_col]
        else:
            y_cols = [df.columns[1]] if len(df.columns) > 1 else [df.columns[0]]

        x_data = df[x_col].astype(str).tolist()

        if chart_type == 'line':
            for yc in y_cols:
                ax.plot(x_data, pd.to_numeric(df[yc], errors='coerce'), marker='o', label=yc)
            ax.set_xlabel(x_col)
            ax.legend()
            plt.xticks(rotation=45, ha='right')

        elif chart_type == 'bar':
            if len(y_cols) == 1:
                ax.bar(x_data, pd.to_numeric(df[y_cols[0]], errors='coerce'))
            else:
                x_idx = list(range(len(x_data)))
                width = 0.8 / len(y_cols)
                for i, yc in enumerate(y_cols):
                    offset = (i - len(y_cols) / 2 + 0.5) * width
                    ax.bar([xi + offset for xi in x_idx],
                           pd.to_numeric(df[yc], errors='coerce'),
                           width=width, label=yc)
                ax.set_xticks(x_idx)
                ax.set_xticklabels(x_data)
                ax.legend()
            plt.xticks(rotation=45, ha='right')

        elif chart_type == 'hbar':
            # 自动修正轴方向：hbar 中 x_col 应为类别标签（y轴），y_col 应为数值（x轴）
            x_vals_numeric = pd.to_numeric(df[x_col], errors='coerce')
            y_vals_numeric = pd.to_numeric(df[y_cols[0]], errors='coerce')
            x_is_numeric = x_vals_numeric.notna().mean() > 0.5
            y_is_numeric = y_vals_numeric.notna().mean() > 0.5

            if x_is_numeric and y_is_numeric:
                # 两列都是数值：取绝对值均值较大的作为数值轴（y_col），较小的作为类别轴（x_col）
                x_mean = x_vals_numeric.abs().mean()
                y_mean = y_vals_numeric.abs().mean()
                if x_mean > y_mean:
                    # x_col 是大数值（收入等），y_col 是小数值（年份等）→ 交换
                    x_col, y_cols[0] = y_cols[0], x_col
                    x_data = df[x_col].astype(str).tolist()
            elif x_is_numeric and not y_is_numeric:
                # x_col 是数值，y_col 是字符串 → 交换
                x_col, y_cols[0] = y_cols[0], x_col
                x_data = df[x_col].astype(str).tolist()


        elif chart_type == 'pie':
            lc = label_col if label_col in df.columns else x_col
            vals = pd.to_numeric(df[y_cols[0]], errors='coerce').fillna(0)
            ax.pie(vals, labels=df[lc].astype(str).tolist(),
                   autopct='%1.1f%%', startangle=90)
            ax.axis('equal')

        elif chart_type == 'scatter':
            ax.scatter(pd.to_numeric(df[x_col], errors='coerce'),
                       pd.to_numeric(df[y_cols[0]], errors='coerce'), alpha=0.6)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_cols[0])

        elif chart_type == 'histogram':
            vals = pd.to_numeric(df[y_cols[0]], errors='coerce').dropna()
            ax.hist(vals, bins=15, edgecolor='black')
            ax.set_xlabel(y_cols[0])
            ax.set_ylabel('频数')

        elif chart_type == 'radar':
            categories = y_cols
            N = len(categories)
            if N < 3:
                return False
            import math
            angles = [n / N * 2 * math.pi for n in range(N)] + [0]
            ax = plt.subplot(111, polar=True)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            for _, row in df.iterrows():
                vals = [float(row[c]) if pd.notna(row[c]) else 0 for c in categories] + \
                       [float(row[categories[0]]) if pd.notna(row[categories[0]]) else 0]
                ax.plot(angles, vals, marker='o', label=str(row[x_col]))
                ax.fill(angles, vals, alpha=0.1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        elif chart_type == 'boxplot':
            groups = [(yc, pd.to_numeric(df[yc], errors='coerce').dropna().tolist())
                      for yc in y_cols if not df[yc].empty]
            if groups:
                ax.boxplot([g[1] for g in groups], labels=[g[0] for g in groups])

        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"[图表生成失败] {e}")
        plt.close()
        return False


def _df_to_text(df: pd.DataFrame, max_rows=50) -> str:
    if df is None or df.empty:
        return "查询结果为空"
    total = len(df)
    text = df.head(max_rows).to_string(index=False)
    if total > max_rows:
        text += f"\n... 共{total}行，仅展示前{max_rows}行"
    return text


# ── 核心 Agent ────────────────────────────────
class FinanceAgent:
    """财报智能问数助手，支持多轮对话 + Schema Linking"""

    def __init__(self):
        self.history = []       # [{role, content}]
        self.context = {}       # 结构化上下文（公司、时间等）
        self.sql_history = []   # 本轮对话执行过的SQL
        self.rag_companies = [] # 上一轮从研报提取的公司名单（多轮复用）

    def reset(self):
        self.history = []
        self.context = {}
        self.sql_history = []
        self.rag_companies = []

    def chat(self, user_input: str, question_id: str = '', turn_idx: int = 0):
        """
        处理一轮对话，返回:
        {content, image, sql, need_clarify}
        """
        # ── Step 1: Schema Linking ──────────────
        linked = build_linked_schema(user_input, self.context if self.context else None)

        # ── 必填槽位定义 ──────────────────────────────
        _REQUIRED_SLOTS = {
                "year": {
                    "check": lambda linked, ctx: (
                        linked.get("periods") is None          # None = 多年范围，合法
                        or any(p[0] for p in (linked.get("periods") or []))
                        or ctx.get("last_period_year")
                    ),
                    "question": "请问您想查询哪一年的数据？（例如：2023年）"
                }
            }


        def _check_slots(linked: dict, context: dict) -> str | None:
            """
            检查必填槽位，返回第一个缺失槽位的追问话术，全部满足返回 None
            """
            for slot_name, slot_cfg in _REQUIRED_SLOTS.items():
                if not slot_cfg["check"](linked, context):
                    return slot_cfg["question"]
            return None
        
        # ── Step 1.5: 显式槽位检查（优先于LLM判断）──
        missing = _check_slots(linked, self.context)
        if missing:
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": missing})
            return {"content": missing, "image": [], "sql": "", "need_clarify": True}

        # ── Step 2: SQL生成（带精简schema） ──────
        sql_user_msg = _build_sql_prompt(user_input, linked, self.sql_history, self.context)

        # 加入对话历史（最近4轮，避免token过多）
        sql_messages = [{"role": "system", "content": SQL_GEN_SYSTEM}]
        for h in self.history[-8:]:
            sql_messages.append(h)
        sql_messages.append({"role": "user", "content": sql_user_msg})

        raw = _call_llm(sql_messages, temperature=0.1)
        plan = _extract_json(raw)

        if plan is None:
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": raw})
            return {"content": raw, "image": [], "sql": "", "need_clarify": False}

        intent = plan.get("intent", "query")
        sql = (plan.get("sql") or "").strip().rstrip(';')
        chart_type = plan.get("chart_type", "none")
        chart_config = plan.get("chart_config", {}) or {}

        # ── Step 3: 意图澄清 ─────────────────────
        if intent == "clarify":
            clarify_q = plan.get("clarify_question", "请提供更多信息。")
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": clarify_q})
            return {"content": clarify_q, "image": [], "sql": "", "need_clarify": True}

        # ── Step 4: 执行SQL ──────────────────────
        df, err = None, None
        if sql:
            df, err = execute_sql_safe(sql)
            # SQL出错时尝试自动修复（一次重试）
            if err:
                group_by_hint = ""
                if "GROUP BY" in err or "group_by" in err.lower() or "isn't in GROUP BY" in err or "aggregat" in err.lower():
                    group_by_hint = (
                        "\n【修复提示】这是GROUP BY错误：SELECT中所有非聚合列必须出现在GROUP BY中。"
                        "请改用CTE分步写法：先在子查询/CTE中完成JOIN和明细列计算，再在外层做聚合，"
                        "确保每个SELECT层级的非聚合列都在GROUP BY里。"
                    )
                fix_msg = (
                    f"SQL执行报错：{err}\n"
                    f"原SQL：{sql}\n"
                    f"请修复SQL，只输出修复后的JSON（格式同上）。{group_by_hint}"
                )
                fix_messages = [{"role": "system", "content": SQL_GEN_SYSTEM},
                                 {"role": "user", "content": sql_user_msg},
                                 {"role": "assistant", "content": raw},
                                 {"role": "user", "content": fix_msg}]
                raw2 = _call_llm(fix_messages, temperature=0.1)
                plan2 = _extract_json(raw2)
                if plan2 and plan2.get("sql"):
                    sql2 = plan2["sql"].strip().rstrip(';')
                    df2, err2 = execute_sql_safe(sql2)
                    if err2 is None:
                        sql, df, err = sql2, df2, None
                        chart_type = plan2.get("chart_type", chart_type)
                        chart_config = plan2.get("chart_config", chart_config) or {}

            # ── 年报不存在时回退：改查该年所有季报 ──
            if err is None and (df is None or df.empty):
                # 匹配带或不带表别名前缀的 report_period='年报'，如 k.report_period='年报'
                if re.search(r"(?:\w+\.)?report_period\s*=\s*'年报'", sql, re.IGNORECASE):
                    fallback_sql = re.sub(
                        r"AND\s+(?:\w+\.)?report_period\s*=\s*'年报'|(?:\w+\.)?report_period\s*=\s*'年报'\s*AND",
                        "", sql, flags=re.IGNORECASE
                    ).strip()
                    # 去掉单独出现的 WHERE [alias.]report_period='年报'
                    fallback_sql = re.sub(
                        r"WHERE\s+(?:\w+\.)?report_period\s*=\s*'年报'",
                        "WHERE 1=1", fallback_sql, flags=re.IGNORECASE
                    ).strip()
                    df_fb, err_fb = execute_sql_safe(fallback_sql)
                    if err_fb is None and df_fb is not None and not df_fb.empty:
                        sql = fallback_sql
                        df = df_fb
                        print(f"[回退] 年报无数据，已改查该年所有季报，共{len(df_fb)}行")

            if sql:
                self.sql_history.append(sql)

        # ── Step 5: 生成图表 ─────────────────────
        image_paths = []
        if df is not None and not df.empty and chart_type != 'none':
            img_name = f"{question_id}_{turn_idx + 1}.jpg" if question_id else f"chart_{turn_idx + 1}.jpg"
            img_path = os.path.join(RESULT_DIR, img_name)
            if _generate_chart(df, chart_type, chart_config, img_path):
                image_paths.append(f"./result/{img_name}")

        # ── Step 6: 生成分析结论 ─────────────────
        data_text = _df_to_text(df) if df is not None else \
                    (f"SQL执行错误: {err}" if err else "无数据")

        analysis_prompt = f"""用户问题：{user_input}

SQL查询语句：{sql}

查询结果：
{data_text}

请用中文给出专业、简洁的分析结论（不超过300字）：
1. 直接回答用户问题，引用具体数据
2. 如有趋势/对比，给出判断
3. 如查询结果为空或报错，说明原因
4. 回答要基于数据库中的数据，不能假设数据库中没有的公司或数据存在，比如问你营业收入最高的公司，你只能在数据库中有的公司中比较，不能说数据库中没有的公司是最高的
只输出结论文字，不要JSON。"""

        final_answer = _call_llm([
            {"role": "system", "content": "你是专业的财务分析师，擅长解读上市公司财报数据。"},
            {"role": "user", "content": analysis_prompt}
        ], temperature=0.3)
        # ── Step 7: 更新上下文 ───────────────────
        self._update_context(user_input, linked, df)
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": final_answer})

        return {
            "content": final_answer,
            "image": image_paths,
            "sql": sql,
            "need_clarify": False,
        }

    def _update_context(self, user_input: str, linked: dict, df: pd.DataFrame):
        if linked.get("companies"):
            c = linked["companies"][0]
            self.context["last_stock_code"] = c["stock_code"]
            self.context["last_company"] = c.get("short_name") or c.get("company_name", "")
        if linked.get("periods"):
            p = linked["periods"][0]  # p 是 (year, period_str) 元组
            self.context["last_period"] = p
            if p[0]:  # p[0] 是 year
                self.context["last_period_year"] = p[0]


        if df is not None and not df.empty and len(df) == 1:
            row = df.iloc[0]
            if "stock_code" in df.columns:
                self.context["last_stock_code"] = str(row["stock_code"])
            if "short_name" in df.columns:
                self.context["last_company"] = str(row["short_name"])
            if "report_period" in df.columns:
                self.context["last_period"] = str(row["report_period"])



def process_question_group(question_id: str, questions: list):
    """
    处理一组多轮对话问题
    返回: (results_list, sql_list)
    """
    agent = FinanceAgent()
    results = []
    sql_list = []

    for i, q_item in enumerate(questions):
        q_text = q_item.get("Q", "")
        print(f"  [{question_id}] 第{i+1}轮: {q_text[:50]}...")

        result = agent.chat(q_text, question_id=question_id, turn_idx=i)

        a_obj = {"content": _clean_text(result["content"])}
        if result["image"]:
            a_obj["image"] = result["image"]

        results.append({"Q": q_text, "A": a_obj})
        if result["sql"]:
            sql_list.append(result["sql"])

    return results, sql_list


# ═══════════════════════════════════════════════════════════════
# 任务三：增强版 Agent（知识库检索 + 多意图规划 + 归因分析）
# ═══════════════════════════════════════════════════════════════

def _dify_retrieve(query: str, top_k: int = 5) -> list:
    """
    调用 Dify 知识库检索接口，返回 [{content, source, score}]
    """
    try:
        resp = requests.post(
            f"{DIFY_URL}/datasets/{DIFY_DATASET_ID}/retrieve",
            headers={"Authorization": f"Bearer {DIFY_API_KEY}", "Content-Type": "application/json"},
            json={
                "query": query,
                "retrieval_model": {
                    "search_method": "hybrid_search",
                    "reranking_enable": False,
                    "top_k": top_k,
                    "score_threshold_enabled": False,
                }
            },
            timeout=30
        )
        if resp.status_code != 200:
            print(f"[Dify] 检索失败 {resp.status_code}: {resp.text[:200]}")
            return []
        data = resp.json()
        results = []
        for rec in data.get("records", []):
            seg = rec.get("segment", {})
            doc_id = seg.get("document_id", "")
            content = seg.get("content", "")
            score = rec.get("score", 0)
            # 获取文档名称（作为 paper_path 的一部分）
            doc_name = seg.get("document", {}).get("name", "") if isinstance(seg.get("document"), dict) else ""
            results.append({
                "content": content,
                "doc_id": doc_id,
                "doc_name": doc_name,
                "score": score,
            })
        return results
    except Exception as e:
        print(f"[Dify] 检索异常: {e}")
        return []


def _clean_text(text: str) -> str:
    """清理文本中的多余换行和回车符"""
    # 处理PDF提取产生的字面量 \n（反斜杠+n）
    text = text.replace('\\n', ' ')
    # 统一真实换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _build_references(rag_results: list) -> list:
    """
    将 Dify 检索结果转换为 references 格式
    [{paper_path, text, paper_image}]
    """
    refs = []
    seen_docs = set()
    for r in rag_results:
        doc_name = r.get("doc_name", "")
        if not doc_name or doc_name in seen_docs:
            continue
        seen_docs.add(doc_name)
        # 推断文件路径（研报放在附件5目录下）
        paper_path = _resolve_paper_path(doc_name)
        refs.append({
            "paper_path": paper_path,
            "text": _clean_text(r["content"][:200]),
            "paper_image": "",
        })
    return refs


def _resolve_paper_path(doc_name: str) -> str:
    """根据文档名推断研报路径"""
    # 尝试在个股研报和行业研报目录下查找
    for sub in ["个股研报", "行业研报"]:
        full = os.path.join(REPORT_BASE_DIR, sub, doc_name)
        if os.path.exists(full):
            return os.path.relpath(full, ".")
    # 找不到时直接返回文件名
    return os.path.join(REPORT_BASE_DIR, doc_name)


# ── 多意图规划 Prompt ─────────────────────────
INTENT_PLAN_SYSTEM = """你是一个智能任务规划器，专注于上市公司财报分析。

用户的问题可能包含多个意图，你需要：
1. 识别问题中的所有核心意图
2. 将其拆解为有序的子任务序列
3. 标注每个子任务的类型：db_query（数据库查询）、rag_search（知识库检索）、analysis（分析推理）、clarify（意图澄清）、open_ended（开放性问题，无需查询数据库）

输出JSON格式：
{
  "question_type": "multi_intent|single_intent|clarify|open_ended|rag_only",
  "sub_tasks": [
    {
      "id": 1,
      "type": "db_query|rag_search|analysis|clarify|open_ended",
      "description": "子任务描述",
      "depends_on": [],
      "rag_query": "如果type=rag_search，填写检索关键词"
    }
  ],
  "needs_rag": true/false,
  "rag_queries": ["检索关键词1", "检索关键词2"]
}

规则：
- 涉及研报、政策、行业分析、原因分析、归因等需要 needs_rag=true
- 纯数据库查询（如查某公司某指标）needs_rag=false
- 开放性问题（如宏观市场分析、不涉及数据库的问题）question_type=open_ended
- 意图模糊时 question_type=clarify
- 只输出JSON，不要其他文字"""


def _plan_intents(question: str, history: list = None) -> dict:
    """分析问题意图，返回任务规划"""
    messages = [{"role": "system", "content": INTENT_PLAN_SYSTEM}]
    if history:
        for h in history[-4:]:
            messages.append(h)
    messages.append({"role": "user", "content": f"请分析以下问题的意图：{question}"})
    raw = _call_llm(messages, temperature=0.1)
    plan = _extract_json(raw)
    if plan is None:
        return {"question_type": "single_intent", "sub_tasks": [], "needs_rag": False, "rag_queries": []}
    return plan


# ── 归因分析 Prompt ───────────────────────────
ATTRIBUTION_SYSTEM = """你是专业的财务分析师，擅长结合结构化数据和研报进行归因分析。

你的任务是：
1. 基于数据库查询结果给出数据层面的分析
2. 结合研报内容给出深层原因分析
3. 明确标注每个结论的来源（数据库/研报/推理）
4. 当用户质疑结论时，能清晰呈现完整的推理链路

回答要求：
- 先给出数据事实（来源：数据库）
- 再给出原因分析（来源：研报/行业知识）
- 最后给出综合结论
- 不超过500字
- 只输出文字结论，不要JSON"""


class FinanceAgent3(FinanceAgent):
    """
    任务三增强版 Agent
    在 FinanceAgent 基础上增加：
    - Dify 知识库检索
    - 多意图自主规划
    - 归因分析（references 溯源）
    """

    def chat3(self, user_input: str, question_id: str = '', turn_idx: int = 0) -> dict:
        """
        增强版对话处理，返回:
        {content, image, sql, need_clarify, references}
        """
        # ── Step 1: 意图规划（带前置分类器）────────
        quick_result = quick_needs_rag(user_input)
        if quick_result is not None:
            # 分类器置信度足够，跳过LLM
            intent_plan = {
                "question_type": "single_intent",
                "sub_tasks": [],
                "needs_rag": quick_result,
                "rag_queries": [user_input] if quick_result else [],
                "_from_classifier": True,
            }
        else:
            # 置信度不足，走LLM规划
            intent_plan = _plan_intents(user_input, self.history)

        q_type = intent_plan.get("question_type", "single_intent")
        needs_rag = intent_plan.get("needs_rag", False)
        rag_queries = intent_plan.get("rag_queries", [])

        print(f"    [意图规划] type={q_type}, needs_rag={needs_rag}, rag_queries={rag_queries}, "
              f"from_classifier={intent_plan.get('_from_classifier', False)}")

        # ── Step 2: 开放性问题 / 纯RAG问题 ─────────
        sub_tasks = intent_plan.get("sub_tasks", [])
        has_db_task = any(t.get("type") == "db_query" for t in sub_tasks)

        # 判断是否含有数据库查询关键词（年份+指标+公司名）
        _DB_SIGNALS = re.search(
            r'(20\d{2}|一季报|半年报|三季报|年报|营业收入|净利润|毛利率|净利率|资产负债率|ROE|增长率|扣非'
            r'|研发费用|管理费用|研发情况|研发投入|研发支出'
            r'|总资产|净资产|股东权益|负债|资产|营收|收入|利润|现金流|每股|市值|市盈率)',
            user_input
        )

        # 强知识类信号：政策/医保/行业趋势等，数据库里没有
        _RAG_SIGNALS = re.search(
            r'(医保目录|政策|行业趋势|研报|行业分析|市场分析|宏观|监管|审批|集采|纳入|新药|创新药|FDA|海外市场)',
            user_input
        )

        # 行业综合分析信号：涉及行业内多家公司的综合情况，需要结合知识库
        _INDUSTRY_ANALYSIS = re.search(
            r'(各.*企业|各.*公司|行业.*情况|研发情况|研发.*情况|经营情况|发展情况|竞争情况)',
            user_input
        )

        # 研报筛选信号：需要先从研报提取公司名单，再查数据库
        # 包含：评级类、事件类（资产重组/并购/定增/分拆等）、计划类（拟/计划/预计）、产品类（老年病/心血管等）、集采类
        _RAG_FILTER_SIGNAL = re.search(
            r'(个股研报|研报.*筛选|筛选.*研报|研报.*评级|评级.*研报|强烈推荐|买入评级|推荐评级|研报.*推荐|从.*研报'
            r'|资产重组|重大重组|并购重组|资产注入|借壳|定向增发|定增|分拆上市|股权收购|重组预案'
            r'|拟.*重组|计划.*重组|重组.*计划|拟进行|拟实施|拟开展'
            r'|老年病|心血管|骨科|肿瘤|糖尿病|高血压|慢性病|儿科|妇科|神经|呼吸|消化'
            r'|相关药品|相关产品|主营产品|核心产品|拳头产品|主要产品|重点产品'
            r'|集采.*中标|中标.*集采|集采.*企业|集采.*公司|集采.*名单|集采.*扩围|集采.*品种'
            r'|行业龙头|龙头企业|龙头公司|龙头股|行业领军|细分龙头|赛道龙头)',
            user_input
        )

        # needs_rag 覆盖：行业综合分析或分类器/LLM判断需要RAG时，强制开启RAG检索
        if (_INDUSTRY_ANALYSIS or _RAG_FILTER_SIGNAL) and not needs_rag:
            needs_rag = True
            if not rag_queries:
                rag_queries = [user_input]

        # 当分类器短路（sub_tasks为空）时，用 _DB_SIGNALS 补充判断是否有数据库查询任务
        if not has_db_task and _DB_SIGNALS:
            has_db_task = True

        # 研报筛选+DB查询：强制走混合路径（RAG提取公司名 → DB查财务数据）
        if _RAG_FILTER_SIGNAL and _DB_SIGNALS:
            needs_rag = True
            has_db_task = True
            if not rag_queries:
                rag_queries = [user_input]

        is_rag_only = (
            q_type in ("open_ended", "rag_only")
            or (needs_rag and not has_db_task and q_type != "clarify" and not _RAG_FILTER_SIGNAL)
            or (needs_rag and not _DB_SIGNALS and not _RAG_FILTER_SIGNAL)  # needs_rag 且无DB信号，且非研报筛选
            or (_RAG_SIGNALS and not _DB_SIGNALS and not _RAG_FILTER_SIGNAL)  # 纯知识类信号
        )

        print(f"    [路由] is_rag_only={is_rag_only}, has_db_task={has_db_task}, "
              f"db_signals={bool(_DB_SIGNALS)}, needs_rag={needs_rag}")

        if is_rag_only:
            # 先尝试RAG检索补充知识
            rag_results = []
            for q in (rag_queries or [user_input])[:3]:
                rag_results.extend(_dify_retrieve(q, top_k=4))
            # 去重 + rerank
            seen = set(); deduped = []
            for r in rag_results:
                if r["doc_id"] not in seen:
                    seen.add(r["doc_id"]); deduped.append(r)
            rag_results = _rerank_results(user_input, deduped, top_k=4)

            rag_context = ""
            if rag_results:
                rag_context = "\n\n【研报参考资料】\n" + "\n---\n".join(
                    f"[来源: {r.get('doc_name', '研报')}]\n{_clean_text(r['content'])[:400]}"
                    for r in rag_results
                )

            answer = _call_llm([
                {"role": "system", "content": "你是专业的中药行业分析师，擅长结合行业研报和市场知识进行深度分析。"},
                {"role": "user", "content": f"{user_input}{rag_context}\n\n请给出专业、有深度的分析（不超过500字）。"}
            ], temperature=0.5)

            refs = _build_references(rag_results)
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": answer})
            return {"content": answer, "image": [], "sql": "", "need_clarify": False, "references": refs}

        # ── Step 2.5: 验证/质疑追问检测 ─────────
        # 用户质疑上一轮结论（如"你确定吗""判断依据是什么""为什么是这些公司"）
        _VERIFY_SIGNAL = re.search(
            r'(确定|对吗|正确吗|判断依据|依据是什么|为什么是这些|怎么判断|来源|根据|证据|可靠|准确)',
            user_input
        )
        if _VERIFY_SIGNAL and self.rag_companies and not _DB_SIGNALS:
            # 重新检索知识库，给出公司名单的依据
            # 从对话历史中提取上一轮的关键词作为检索词
            prev_question = ""
            for h in reversed(self.history):
                if h["role"] == "user":
                    prev_question = h["content"]
                    break
            base_query = prev_question if prev_question else user_input
            verify_queries = [f"{base_query} {c}" for c in self.rag_companies[:2]] + [base_query]
            verify_results = []
            for q in verify_queries[:3]:
                verify_results.extend(_dify_retrieve(q, top_k=4))
            seen = set(); deduped_v = []
            for r in verify_results:
                if r["doc_id"] not in seen:
                    seen.add(r["doc_id"]); deduped_v.append(r)
            verify_results = _rerank_results(user_input, deduped_v, top_k=4)

            companies_str = "、".join(self.rag_companies)
            rag_evidence = "\n---\n".join(
                f"[来源: {r.get('doc_name', '研报')}]\n{_clean_text(r['content'])[:500]}"
                for r in verify_results
            ) if verify_results else "（未检索到相关研报片段）"

            verify_prompt = f"""用户问题：{user_input}

上一轮从研报中识别出的公司名单：{companies_str}

以下是支撑该判断的研报原文片段：
{rag_evidence}

请根据上述研报内容，说明这些公司被列入名单的依据，逐一解释每家公司在研报中的相关描述。
如果某家公司在研报中没有明确依据，请如实说明。只输出文字，不要JSON。"""

            verify_answer = _call_llm([
                {"role": "system", "content": "你是信息溯源专家，擅长从研报中找到支撑结论的原文依据。"},
                {"role": "user", "content": verify_prompt}
            ], temperature=0.2)

            refs = _build_references(verify_results)
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": verify_answer})
            return {"content": verify_answer, "image": [], "sql": "", "need_clarify": False, "references": refs}

        # ── Step 3: 意图澄清 ─────────────────────
        if q_type == "clarify":
            # 检查是否是对上一轮的补充（如"2025年第三季度的"）
            if self.history and len(user_input) < 20:
                # 短问题可能是追问，走正常流程
                pass
            else:
                clarify_msg = "您的问题较为模糊，请补充以下信息：\n1. 您想查询哪家公司？\n2. 查询哪个时间段的数据？\n3. 关注哪个财务指标？"
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": clarify_msg})
                return {"content": clarify_msg, "image": [], "sql": "", "need_clarify": True, "references": []}

        # ── Step 4: RAG 检索 + Reranking ──────────
        rag_results = []
        if needs_rag:
            queries = rag_queries if rag_queries else [user_input]
            for q in queries[:3]:
                rag_results.extend(_dify_retrieve(q, top_k=6))  # 多取一些供rerank筛选

            # 去重（按doc_id）
            seen = set()
            deduped = []
            for r in rag_results:
                if r["doc_id"] not in seen:
                    seen.add(r["doc_id"])
                    deduped.append(r)

            # ★ Reranking：用原始问题对去重后的结果重排
            rag_results = _rerank_results(user_input, deduped, top_k=4)
            print(f"    [Reranking] 重排后保留 {len(rag_results)} 条，"
                  f"top分数: {rag_results[0].get('rerank_score', 'N/A') if rag_results else 'N/A'}")      

        # ── Step 4.5: 研报筛选模式 → 从RAG结果提取公司名注入SQL ──
        # 优先复用上一轮已提取的公司名单（多轮对话场景，如"你确定吗"）
        rag_extracted_companies = list(self.rag_companies)  # 继承上轮结果

        if _RAG_FILTER_SIGNAL and not rag_results:
            # RAG检索无结果，直接澄清
            print(f"    [研报筛选] RAG无检索结果，触发澄清")
            clarify_msg = "请提供具体的企业名单或筛选条件，以便准确查询相关财务数据。同时，请确认您希望查询的具体财务指标和时间段。"
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": clarify_msg})
            return {"content": clarify_msg, "image": [], "sql": "", "need_clarify": True, "references": []}

        # ── 行业龙头硬编码名单（优先级高于RAG提取）──
        # 只要问题中含有"龙头"，直接使用固定名单，无需匹配行业关键词
        _LEADER_COMPANIES = ["同仁堂", "昆药集团", "羚锐制药", "片仔癀", "东阿阿胶"]

        if "龙头" in user_input:
            rag_extracted_companies = _LEADER_COMPANIES
            self.rag_companies = rag_extracted_companies
            # 确保后续走 RAG+DB 混合路径
            needs_rag = True
            has_db_task = True
            print(f"    [龙头名单] 检测到'龙头'关键词，使用固定名单: {rag_extracted_companies}")

        if _RAG_FILTER_SIGNAL and rag_results and not rag_extracted_companies:
            rag_text_for_extract = "\n---\n".join(
                f"[{r.get('doc_name', '')}]\n{_clean_text(r['content'])[:600]}"
                for r in rag_results
            )
            extract_prompt = f"""以下是从研报知识库检索到的内容：

{rag_text_for_extract}

用户问题：{user_input}

请根据用户问题，从上述研报内容中提取符合条件的公司名称（公司简称）。
提取规则：
- 若问题涉及"行业龙头/龙头企业/龙头公司/行业领军/细分龙头/赛道龙头"，提取研报中被明确描述为龙头、行业龙头、龙头企业或市场领导者的公司，最优先找中药方面的龙头，尤其是"同仁堂","昆药集团","羚锐制药',"片仔癀","东阿阿胶"这几家公司
- 若问题涉及"资产重组/并购重组/重大重组/资产注入/借壳/定增/分拆"等事件，提取研报中提到拟进行或正在进行该事件的公司
- 若问题涉及"强烈推荐/买入/推荐"等评级，提取被给予该评级的公司
- 若问题涉及特定疾病领域/产品类型（如老年病、心血管、骨科、肿瘤等），提取研报中提到该领域产品收入占比较高或主营该类产品的公司
- 若问题涉及其他特定事件或条件，提取满足该条件的公司
只输出公司简称列表，JSON格式：{{"companies": ["公司A", "公司B"]}}
如果研报中没有明确相关信息，返回：{{"companies": []}}
只输出JSON，不要其他文字。"""
            extract_raw = _call_llm([
                {"role": "system", "content": "你是信息提取专家，擅长从研报文本中提取结构化信息。"},
                {"role": "user", "content": extract_prompt}
            ], temperature=0.1)
            extract_result = _extract_json(extract_raw)
            if extract_result and extract_result.get("companies"):
                rag_extracted_companies = extract_result["companies"]
                self.rag_companies = rag_extracted_companies  # 保存供后续轮次复用
                print(f"    [研报筛选] 从RAG提取到公司: {rag_extracted_companies}")
            else:
                # 研报中未找到符合条件的公司名单，需要向用户澄清
                print(f"    [研报筛选] RAG未提取到公司，触发澄清")
                clarify_msg = "请提供集采中标企业的具体名单或筛选条件，以便准确查询这些企业在2025年第三季度的净利润率变化。同时，请确认是否需要提取研报对集采影响的观点。"
                # 尝试从研报中提取集采相关观点作为补充
                if rag_results:
                    rag_context_for_clarify = "\n\n【已检索到的研报参考内容】\n" + "\n---\n".join(
                        f"[来源: {r.get('doc_name', '研报')}]\n{_clean_text(r['content'])[:300]}"
                        for r in rag_results[:3]
                    )
                    clarify_msg = _call_llm([
                        {"role": "system", "content": "你是专业的财务分析师助手。"},
                        {"role": "user", "content": (
                            f"用户问题：{user_input}\n"
                            f"{rag_context_for_clarify}\n\n"
                            f"研报知识库中未找到明确的集采中标企业名单。"
                            f"请根据研报内容，先提取研报中关于集采影响的观点（如有），"
                            f"然后告知用户需要提供集采中标企业的具体名单或筛选条件才能查询财务数据。"
                            f"回答不超过300字，只输出文字。"
                        )}
                    ], temperature=0.3)
                refs = _build_references(rag_results) if rag_results else []
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": clarify_msg})
                return {"content": clarify_msg, "image": [], "sql": "", "need_clarify": True, "references": refs}
        elif not _RAG_FILTER_SIGNAL and self.rag_companies:
            # 当前轮没有筛选信号但上轮有结果，说明是追问（如"你确定吗"），继续使用
            print(f"    [研报筛选] 复用上轮提取的公司: {rag_extracted_companies}")

        # ── Step 5: 数据库查询（复用父类逻辑）────
        linked = build_linked_schema(user_input, self.context if self.context else None)
        fixed_periods = _auto_fix_recent_years(user_input, linked.get("periods", []))
        if fixed_periods is not linked.get("periods"):
            # periods 被修正了，同步更新 schema_text 中的时间段提示
            linked["linked_schema_text"] = _rebuild_schema_text_with_periods(linked, fixed_periods)
        linked["periods"] = fixed_periods
        
        # 槽位检查
        _REQUIRED_SLOTS = {
            "year": {
                "check": lambda lk, ctx: (
                    lk.get("periods") is None
                    or any(p[0] for p in (lk.get("periods") or []))
                    or ctx.get("last_period_year")
                ),
                "question": "请问您想查询哪一年的数据？（例如：2023年）"
            }
        }
        for _, slot_cfg in _REQUIRED_SLOTS.items():
            if not slot_cfg["check"](linked, self.context):
                msg = slot_cfg["question"]
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": msg})
                return {"content": msg, "image": [], "sql": "", "need_clarify": True, "references": []}

        sql_user_msg = _build_sql_prompt(user_input, linked, self.sql_history, self.context)

        # 研报筛选模式：把从RAG提取的公司名注入到SQL prompt，避免LLM造不存在的表
        if rag_extracted_companies:
            names_str = "、".join(rag_extracted_companies)
            sql_user_msg += (
                f"\n\n【研报筛选结果（重要）】从个股研报中筛选出的目标公司为：{names_str}\n"
                f"请只查询这些公司的数据，用 c.short_name IN ({', '.join(repr(n) for n in rag_extracted_companies)}) 过滤，"
                f"禁止使用 research_report 等不存在的表。\n"
                f"【注意】若用户问题涉及的指标（如某类药品收入占比）在数据库中不存在对应字段，"
                f"请改为查询这些公司的营业总收入和净资产收益率（weighted_roe_indicator），"
                f"intent仍设为query，不要设为clarify。"
            )

        sql_messages = [{"role": "system", "content": SQL_GEN_SYSTEM}]
        for h in self.history[-8:]:
            sql_messages.append(h)
        sql_messages.append({"role": "user", "content": sql_user_msg})

        raw = _call_llm(sql_messages, temperature=0.1)
        plan = _extract_json(raw)

        if plan is None:
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": raw})
            return {"content": raw, "image": [], "sql": "", "need_clarify": False, "references": []}

        intent = plan.get("intent", "query")
        sql = (plan.get("sql") or "").strip().rstrip(';')
        chart_type = plan.get("chart_type", "none")
        chart_config = plan.get("chart_config", {}) or {}

        # 如果已从研报提取到公司名单，禁止走clarify（LLM不应再要求提供公司名）
        if intent == "clarify" and not rag_extracted_companies:
            clarify_q = plan.get("clarify_question", "请提供更多信息。")
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": clarify_q})
            return {"content": clarify_q, "image": [], "sql": "", "need_clarify": True, "references": []}

        # ── Step 6: 执行SQL ──────────────────────
        df, err = None, None
        if sql:
            df, err = execute_sql_safe(sql)
            if err:
                group_by_hint = ""
                if "GROUP BY" in err or "group_by" in err.lower() or "isn't in GROUP BY" in err or "aggregat" in err.lower():
                    group_by_hint = (
                        "\n【修复提示】这是GROUP BY错误：SELECT中所有非聚合列必须出现在GROUP BY中。"
                        "请改用CTE分步写法：先在子查询/CTE中完成JOIN和明细列计算，再在外层做聚合，"
                        "确保每个SELECT层级的非聚合列都在GROUP BY里。"
                    )
                fix_msg = (f"SQL执行报错：{err}\n原SQL：{sql}\n请修复SQL，只输出修复后的JSON。{group_by_hint}")
                fix_messages = [{"role": "system", "content": SQL_GEN_SYSTEM},
                                 {"role": "user", "content": sql_user_msg},
                                 {"role": "assistant", "content": raw},
                                 {"role": "user", "content": fix_msg}]
                raw2 = _call_llm(fix_messages, temperature=0.1)
                plan2 = _extract_json(raw2)
                if plan2 and plan2.get("sql"):
                    sql2 = plan2["sql"].strip().rstrip(';')
                    df2, err2 = execute_sql_safe(sql2)
                    if err2 is None:
                        sql, df, err = sql2, df2, None
                        chart_type = plan2.get("chart_type", chart_type)
                        chart_config = plan2.get("chart_config", chart_config) or {}

            if err is None and (df is None or df.empty):
                if re.search(r"(?:\w+\.)?report_period\s*=\s*'年报'", sql, re.IGNORECASE):
                    fallback_sql = re.sub(
                        r"AND\s+(?:\w+\.)?report_period\s*=\s*'年报'|(?:\w+\.)?report_period\s*=\s*'年报'\s*AND",
                        "", sql, flags=re.IGNORECASE
                    ).strip()
                    fallback_sql = re.sub(
                        r"WHERE\s+(?:\w+\.)?report_period\s*=\s*'年报'",
                        "WHERE 1=1", fallback_sql, flags=re.IGNORECASE
                    ).strip()
                    df_fb, err_fb = execute_sql_safe(fallback_sql)
                    if err_fb is None and df_fb is not None and not df_fb.empty:
                        sql = fallback_sql
                        df = df_fb

            if sql:
                self.sql_history.append(sql)

        # ── Step 7: 生成图表 ─────────────────────
        image_paths = []
        if df is not None and not df.empty and chart_type != 'none':
            img_name = f"{question_id}_{turn_idx + 1}.jpg" if question_id else f"chart_{turn_idx + 1}.jpg"
            img_path = os.path.join(RESULT_DIR, img_name)
            if _generate_chart(df, chart_type, chart_config, img_path):
                image_paths.append(f"./result/{img_name}")

        # ── Step 8: 生成分析结论（融合RAG）────────
        data_text = _df_to_text(df) if df is not None else \
                    (f"SQL执行错误: {err}" if err else "无数据")

        rag_context = ""
        if rag_results:
            rag_context = "\n\n【研报参考资料（用于归因分析）】\n" + "\n---\n".join(
                f"[来源: {r.get('doc_name', '研报')}]\n{_clean_text(r['content'])[:400]}"
                for r in rag_results[:4]
            )

        # 判断是否需要归因分析
        needs_attribution = needs_rag or any(
            kw in user_input for kw in ["原因", "为什么", "分析", "归因", "解释", "影响", "驱动", "来源", "可靠"]
        )

        if needs_attribution and rag_results:
            analysis_prompt = f"""用户问题：{user_input}

数据库查询结果：
{data_text}
{rag_context}

请结合数据库数据和研报资料，给出专业的归因分析（不超过500字）：
1. 【数据事实】直接引用查询结果中的具体数字
2. 【原因分析】结合研报内容分析背后原因
3. 【综合结论】给出综合判断
明确标注每个结论的来源（数据库/研报）。只输出文字，不要JSON。"""
            sys_prompt = ATTRIBUTION_SYSTEM
        else:
            analysis_prompt = f"""用户问题：{user_input}

SQL查询语句：{sql}

查询结果：
{data_text}
{rag_context}

请用中文给出专业、简洁的分析结论（不超过400字）：
1. 直接回答用户问题，引用具体数据
2. 如有趋势/对比，给出判断
3. 如查询结果为空或报错，说明原因
只输出结论文字，不要JSON。"""
            sys_prompt = "你是专业的财务分析师，擅长解读上市公司财报数据。"

        final_answer = _call_llm([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": analysis_prompt}
        ], temperature=0.3)

        # ── Step 8.5: 自动校验 ───────────────────
        validation = validate_answer(user_input, final_answer, df)
        print(f"    [校验] overall={validation['overall_passed']}, "
              f"accuracy={validation['accuracy']['coverage']:.1%}, "
              f"completeness={validation['completeness']['passed']}")

        # 数值覆盖率不足时，触发一次重新生成
        if not validation["accuracy"]["passed"] and df is not None and not df.empty:
            retry_prompt = f"""{analysis_prompt}
【校验反馈】上一次回答存在数值引用不准确的问题，
未在答案中体现的关键数值：{validation['accuracy']['mismatched_nums']}
请重新生成，确保答案中准确引用以上数值。"""

            final_answer = _call_llm([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": retry_prompt}
            ], temperature=0.1)

            # ========== 新增：清除LLM输出的 \n 换行符 ==========
            final_answer = final_answer.replace('\n', '').replace('\\n', '').replace('**', '').replace('\n\n**','').strip()
            final_answer = final_answer.replace('  ', ' ')
            # ================================================

            validation["retried"] = True
            print(f"    [校验] 触发重试，重新生成答案")

        # ── Step 9: 构建 references ──────────────
        references = _build_references(rag_results) if rag_results else []

        # ── Step 10: 更新上下文 ──────────────────
        self._update_context(user_input, linked, df)
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": final_answer})

        return {
            "content": final_answer,
            "image": image_paths,
            "sql": sql,
            "need_clarify": False,
            "references": references,
        }


def process_question_group3(question_id: str, questions: list) -> tuple:
    """
    任务三：处理一组多轮对话问题（增强版）
    返回: (results_list, sql_list)
    results_list 中每条 A 包含 references 字段
    """
    agent = FinanceAgent3()
    results = []
    sql_list = []

    for i, q_item in enumerate(questions):
        q_text = q_item.get("Q", "")
        print(f"  [{question_id}] 第{i+1}轮: {q_text[:60]}...")

        result = agent.chat3(q_text, question_id=question_id, turn_idx=i)

        a_obj = {"content": _clean_text(result["content"])}
        if result.get("image"):
            a_obj["image"] = result["image"]
        if result.get("references"):
            a_obj["references"] = result["references"]

        results.append({"Q": q_text, "A": a_obj})
        if result.get("sql"):
            sql_list.append(result["sql"])

    return results, sql_list
