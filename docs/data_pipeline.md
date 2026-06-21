# 数据处理流程

本文档说明从 PDF 财报/研报到智能问答的推荐流程。

## 1. PDF 解析

使用 MinerU 对财报 PDF 和研报 PDF 做结构化解析：

```text
PDF -> Markdown / JSON / 表格片段 / 图片
```

建议保留：

- 文档标题、页码、段落文本。
- 表格结构和表头。
- 财务科目原始名称。
- PDF 文件名，用于后续溯源。

## 2. 财报结构化入库

将财报中的结构化指标清洗后写入 MySQL。当前 Agent 默认使用以下核心表：

- `company_info`
- `income_sheet`
- `balance_sheet`
- `cash_flow_sheet`
- `core_performance_indicators_sheet`
- `subject_mapping`

建表参考 [../sql/schema.sql](../sql/schema.sql)。

## 3. 研报知识库

将 MinerU 解析出的研报文本切分后导入 Dify Dataset。

推荐切分策略：

- 每个 chunk 保留公司名、研报标题、页码或文件名。
- chunk 长度控制在 500 到 1000 中文字。
- 对标题、核心观点、风险提示等段落优先保留完整语义。

Dify 检索接口由以下环境变量配置：

```text
DIFY_URL
DIFY_API_KEY
DIFY_DATASET_ID
```

## 4. 问答运行链路

```text
用户问题
  -> 意图分类
  -> Schema Linking
  -> SQL 生成
  -> MySQL 执行
  -> SQL 出错时自修复
  -> 需要原因/政策/行业信息时调用 Dify RAG
  -> 生成答案、引用、图表
  -> 答案数值校验
```

## 5. 本地数据目录建议

完整数据建议放在 `data/` 下：

```text
data/
  reports/
    个股研报/
    行业研报/
  mysql_exports/
  mineru_outputs/
```
