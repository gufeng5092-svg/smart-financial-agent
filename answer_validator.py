"""
输出结果自动校验模块
校验LLM生成的文字结论与数据库查询结果之间的数值一致性
"""
import re
import numpy as np
import pandas as pd
from typing import Optional


def _extract_numbers_from_text(text: str) -> list[float]:
    """
    从文字中提取所有数值（含百分比、带逗号的大数）
    """
    # 先处理百分比：去掉%符号
    text_clean = re.sub(r'(\d+\.?\d*)%', r'\1', text)
    # 提取数字（含小数、千分位逗号）
    raw = re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?', text_clean)
    nums = []
    for r in raw:
        try:
            nums.append(float(r.replace(',', '')))
        except ValueError:
            pass
    return nums


def _extract_numbers_from_df(df: pd.DataFrame) -> list[float]:
    """
    从查询结果 DataFrame 中提取所有数值
    """
    nums = []
    for col in df.columns:
        for val in df[col]:
            try:
                f = float(val)
                if not np.isnan(f) and not np.isinf(f):
                    nums.append(f)
            except (TypeError, ValueError):
                pass
    return nums


def _numbers_match(a: float, b: float, rel_tol: float = 0.05, abs_tol: float = 0.01) -> bool:
    """
    判断两个数值是否匹配
    rel_tol: 相对误差容忍度（默认5%）
    abs_tol: 绝对误差容忍度（处理接近0的情况）
    """
    if abs(b) < abs_tol:
        return abs(a - b) < abs_tol
    return abs(a - b) / abs(b) <= rel_tol


class AnswerValidator:
    """
    答案校验器，检查三个维度：
    1. 数值准确性：文字中的数字与数据库结果一致
    2. 合理性：数值在合理范围内（如利润率不超过100%）
    3. 完整性：问题中的关键实体在答案中均有提及
    """

    # 财务指标合理范围（用于合理性校验）
    _REASONABLE_RANGES = {
        "毛利率": (-50, 100),
        "净利率": (-200, 100),
        "资产负债率": (0, 200),
        "增长率": (-500, 2000),
        "ROE": (-100, 100),
    }

    def validate_accuracy(
        self,
        answer_text: str,
        df: Optional[pd.DataFrame],
        threshold: float = 0.5,
    ) -> dict:
        """
        数值准确性校验
        threshold: 数据库数值被文字覆盖的最低比例
        返回 {passed, coverage, mismatched_nums, detail}
        """
        if df is None or df.empty:
            return {"passed": True, "coverage": 1.0, "mismatched_nums": [], "detail": "无数据库结果，跳过校验"}

        db_nums = _extract_numbers_from_df(df)
        text_nums = _extract_numbers_from_text(answer_text)

        if not db_nums:
            return {"passed": True, "coverage": 1.0, "mismatched_nums": [], "detail": "数据库结果无数值字段"}

        # 只校验数据库中的"关键数值"（过滤掉年份、编号等）
        key_nums = [n for n in db_nums if not (1990 <= n <= 2030) and n != 0]

        if not key_nums:
            return {"passed": True, "coverage": 1.0, "mismatched_nums": [], "detail": "无需校验的关键数值"}

        matched = 0
        mismatched = []
        for db_val in key_nums[:10]:  # 最多校验10个关键数值
            if any(_numbers_match(db_val, t) for t in text_nums):
                matched += 1
            else:
                mismatched.append(db_val)

        coverage = matched / len(key_nums[:10])
        passed = coverage >= threshold

        return {
            "passed": passed,
            "coverage": round(coverage, 3),
            "mismatched_nums": mismatched[:5],
            "detail": f"关键数值覆盖率 {coverage:.1%}，{'通过' if passed else '未通过'}校验"
        }

    def validate_reasonableness(self, answer_text: str) -> dict:
        """
        合理性校验：检测文字中的财务指标数值是否在合理范围内
        """
        issues = []
        for metric, (low, high) in self._REASONABLE_RANGES.items():
            # 匹配"XX%"或"XX（指标名）"模式
            pattern = rf'(\d+\.?\d*)\s*%?\s*(?:的)?{metric}|{metric}[^\d]*?(\d+\.?\d*)\s*%?'
            matches = re.findall(pattern, answer_text)
            for m in matches:
                val_str = m[0] or m[1]
                if val_str:
                    val = float(val_str)
                    if not (low <= val <= high):
                        issues.append(f"{metric}={val} 超出合理范围[{low}, {high}]")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "detail": "；".join(issues) if issues else "合理性校验通过"
        }

    def validate_completeness(self, question: str, answer_text: str) -> dict:
        """
        完整性校验：问题中提到的公司/年份是否在答案中有所体现
        """
        # 提取问题中的年份
        years = re.findall(r'20\d{2}', question)
        # 提取问题中的公司名（简单启发式：2-4个汉字+股份/医药/科技等）
        companies = re.findall(r'[\u4e00-\u9fa5]{2,6}(?:股份|医药|科技|生物|制药|集团)', question)

        missing = []
        for y in years:
            if y not in answer_text:
                missing.append(f"年份{y}")
        for c in companies:
            # 允许部分匹配（公司简称）
            short = c[:2]
            if short not in answer_text:
                missing.append(f"公司{c}")

        passed = len(missing) == 0
        return {
            "passed": passed,
            "missing_entities": missing,
            "detail": f"缺失实体：{missing}" if missing else "完整性校验通过"
        }

    def validate_all(
        self,
        question: str,
        answer_text: str,
        df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        综合校验入口，返回三个维度的校验结果
        """
        acc = self.validate_accuracy(answer_text, df)
        rea = self.validate_reasonableness(answer_text)
        com = self.validate_completeness(question, answer_text)

        overall_passed = acc["passed"] and rea["passed"] and com["passed"]

        return {
            "overall_passed": overall_passed,
            "accuracy": acc,
            "reasonableness": rea,
            "completeness": com,
        }


# 全局单例
_validator = AnswerValidator()


def validate_answer(question: str, answer_text: str, df: pd.DataFrame = None) -> dict:
    return _validator.validate_all(question, answer_text, df)
