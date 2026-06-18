"""
轻量意图分类器
基于关键词特征 + 逻辑回归，快速判断问题是否需要RAG检索
作为LLM意图规划的前置过滤，减少不必要的LLM调用
"""
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import os

MODEL_PATH = "./intent_classifier.pkl"

# ── 训练数据（少样本，覆盖主要场景）────────────────
_TRAIN_DATA = [
    # needs_rag=1：需要研报支撑的问题
    ("分析该公司营收下滑的原因", 1),
    ("为什么净利润大幅下降", 1),
    ("该公司业绩承压的主要驱动因素是什么", 1),
    ("结合行业背景分析毛利率变化", 1),
    ("研发投入增加对公司的影响", 1),
    ("行业竞争格局如何影响公司盈利", 1),
    ("公司战略转型的背景和原因", 1),
    ("分析中药行业的发展趋势", 1),
    ("该公司未来增长潜力如何", 1),
    ("归因分析净利润波动", 1),
    ("解释资产负债率上升的原因", 1),
    ("行业政策对公司的影响", 1),
    ("公司核心竞争力分析", 1),
    ("为何研发费用持续增长", 1),
    ("分析公司经营风险", 1),
    # needs_rag=0：纯数据库查询
    ("查询2024年三季报营业收入", 0),
    ("某公司净利润是多少", 0),
    ("2023年年报毛利率排名前五的公司", 0),
    ("近三年营收增长率对比", 0),
    ("资产负债率最高的公司", 0),
    ("2025年一季报扣非净利润", 0),
    ("研发费用占营收比例", 0),
    ("加权平均净资产收益率", 0),
    ("营业收入同比增长率", 0),
    ("存货周转率最高的公司", 0),
    ("2024年半年报销售净利率", 0),
    ("财务费用变化趋势", 0),
    ("管理费用占比", 0),
    ("总资产规模排名", 0),
    ("净利润环比增长率", 0),
]

# ── 特征工程：关键词权重 ──────────────────────────
_RAG_KEYWORDS = [
    "原因", "为什么", "为何", "分析", "归因", "解释", "影响",
    "驱动", "背景", "趋势", "战略", "竞争", "风险", "潜力",
    "政策", "行业", "核心", "未来", "预测", "展望"
]
_DB_KEYWORDS = [
    "查询", "多少", "排名", "对比", "增长率", "同比", "环比",
    "最高", "最低", "前五", "近三年", "季报", "年报", "半年报"
]


def _extract_features(text: str) -> dict:
    """提取关键词特征，用于增强TF-IDF"""
    rag_count = sum(1 for kw in _RAG_KEYWORDS if kw in text)
    db_count = sum(1 for kw in _DB_KEYWORDS if kw in text)
    has_why = int(bool(re.search(r'为什么|为何|原因|归因', text)))
    has_analysis = int(bool(re.search(r'分析|解释|影响|驱动', text)))
    return {
        "rag_keyword_count": rag_count,
        "db_keyword_count": db_count,
        "has_why": has_why,
        "has_analysis": has_analysis,
    }


class IntentClassifier:
    """
    两阶段意图分类器：
    1. 规则前置：高置信度关键词直接判断
    2. TF-IDF + 逻辑回归：处理模糊情况
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='char_wb',   # 字符级n-gram，适合中文
                ngram_range=(1, 3),
                max_features=500,
                sublinear_tf=True,
            )),
            ('clf', LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
            ))
        ])
        self._trained = False

    def train(self):
        texts = [d[0] for d in _TRAIN_DATA]
        labels = [d[1] for d in _TRAIN_DATA]
        self.pipeline.fit(texts, labels)
        self._trained = True
        joblib.dump(self.pipeline, MODEL_PATH)
        print(f"[IntentClassifier] 训练完成，样本数={len(texts)}")

    def load(self):
        if os.path.exists(MODEL_PATH):
            self.pipeline = joblib.load(MODEL_PATH)
            self._trained = True
            return True
        return False

    def _rule_based(self, text: str) -> int | None:
        """
        规则前置判断，返回 1/0 或 None（无法判断）
        置信度高的情况直接返回，跳过模型推断
        """
        # 强 RAG 信号
        if re.search(r'为什么|为何|原因|归因|解释.*原因', text):
            return 1
        # 强 DB 信号：纯数值查询
        if re.search(r'^\s*(查询|请查|帮我查)', text) and not re.search(r'分析|原因|影响', text):
            return 0
        return None

    def predict(self, text: str) -> tuple[int, float]:
        """
        返回 (label, confidence)
        label: 1=需要RAG, 0=纯数据库查询
        confidence: 置信度 0~1
        """
        # 规则前置
        rule_result = self._rule_based(text)
        if rule_result is not None:
            return rule_result, 0.95

        # 模型推断
        if not self._trained:
            if not self.load():
                self.train()

        proba = self.pipeline.predict_proba([text])[0]
        label = int(np.argmax(proba))
        confidence = float(proba[label])
        return label, confidence


# 全局单例
_classifier = IntentClassifier()


def quick_needs_rag(question: str, threshold: float = 0.6) -> bool:
    """
    快速判断问题是否需要RAG检索
    confidence < threshold 时降级到LLM判断
    """
    label, conf = _classifier.predict(question)
    print(f"    [IntentClassifier] label={label}, confidence={conf:.3f}")
    if conf >= threshold:
        return bool(label)
    # 置信度不足，返回 None 表示需要LLM进一步判断
    return None
