# 数据库结构说明

当前代码默认连接 MySQL，并围绕上市公司财报数据设计了 6 张表。

## company_info

公司基础信息。

- `stock_code`：股票代码。
- `company_name`：公司全称。
- `short_name`：公司简称。
- `industry`：行业。

## income_sheet

利润表。

- `stock_code`
- `report_year`
- `report_period`
- `operating_revenue`
- `operating_cost`
- `gross_profit`
- `selling_expense`
- `admin_expense`
- `rd_expense`
- `financial_expense`
- `total_profit`
- `net_profit`
- `net_profit_attributable`

## balance_sheet

资产负债表。

- `stock_code`
- `report_year`
- `report_period`
- `monetary_funds`
- `accounts_receivable`
- `inventory`
- `fixed_assets`
- `total_current_assets`
- `total_non_current_assets`
- `total_assets`
- `short_term_borrowing`
- `total_current_liabilities`
- `total_non_current_liabilities`
- `total_liabilities`
- `undistributed_profit`
- `total_equity`

## cash_flow_sheet

现金流量表。

- `stock_code`
- `report_year`
- `report_period`
- `net_operating_cash_flow`
- `net_investing_cash_flow`
- `net_financing_cash_flow`
- `ending_cash`

## core_performance_indicators_sheet

核心业绩指标表。

- `stock_code`
- `report_year`
- `report_period`
- `total_revenue_indicator`
- `net_profit_indicator`
- `net_profit_deducted`
- `basic_eps`
- `diluted_eps`
- `weighted_roe_indicator`
- `net_cash_per_share`
- `net_assets_per_share`
- `total_assets_indicator`
- `net_assets_indicator`

## subject_mapping

财务科目同义词映射。

- `standard_name`
- `synonyms`
- `target_table`
- `target_column`

注意：如果你的实际数据库字段名和上述字段不同，需要同步修改 [../schema_linker.py](../schema_linker.py) 和 [../db.py](../db.py) 中的 schema 描述。
