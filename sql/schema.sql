CREATE TABLE IF NOT EXISTS company_info (
  stock_code VARCHAR(10) PRIMARY KEY,
  company_name VARCHAR(100),
  short_name VARCHAR(50),
  industry VARCHAR(50)
) DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS income_sheet (
  stock_code VARCHAR(10),
  report_year INT,
  report_period VARCHAR(20),
  operating_revenue DECIMAL(20,4),
  operating_cost DECIMAL(20,4),
  gross_profit DECIMAL(20,4),
  selling_expense DECIMAL(20,4),
  admin_expense DECIMAL(20,4),
  rd_expense DECIMAL(20,4),
  financial_expense DECIMAL(20,4),
  total_profit DECIMAL(20,4),
  net_profit DECIMAL(20,4),
  net_profit_attributable DECIMAL(20,4),
  income_tax DECIMAL(20,4),
  is_anomaly TINYINT DEFAULT 0,
  PRIMARY KEY (stock_code, report_year, report_period)
) DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS balance_sheet (
  stock_code VARCHAR(10),
  report_year INT,
  report_period VARCHAR(20),
  monetary_funds DECIMAL(20,4),
  accounts_receivable DECIMAL(20,4),
  inventory DECIMAL(20,4),
  fixed_assets DECIMAL(20,4),
  total_current_assets DECIMAL(20,4),
  total_non_current_assets DECIMAL(20,4),
  total_assets DECIMAL(20,4),
  short_term_borrowing DECIMAL(20,4),
  total_current_liabilities DECIMAL(20,4),
  total_non_current_liabilities DECIMAL(20,4),
  total_liabilities DECIMAL(20,4),
  undistributed_profit DECIMAL(20,4),
  total_equity DECIMAL(20,4),
  accounts_payable DECIMAL(20,4),
  is_anomaly TINYINT DEFAULT 0,
  PRIMARY KEY (stock_code, report_year, report_period)
) DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS cash_flow_sheet (
  stock_code VARCHAR(10),
  report_year INT,
  report_period VARCHAR(20),
  net_operating_cash_flow DECIMAL(20,4),
  net_investing_cash_flow DECIMAL(20,4),
  net_financing_cash_flow DECIMAL(20,4),
  ending_cash DECIMAL(20,4),
  is_anomaly TINYINT DEFAULT 0,
  PRIMARY KEY (stock_code, report_year, report_period)
) DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS core_performance_indicators_sheet (
  stock_code VARCHAR(10),
  report_year INT,
  report_period VARCHAR(20),
  total_revenue_indicator DECIMAL(20,4),
  net_profit_indicator DECIMAL(20,4),
  net_profit_deducted DECIMAL(20,4),
  basic_eps DECIMAL(20,4),
  diluted_eps DECIMAL(20,4),
  weighted_roe_indicator DECIMAL(20,4),
  net_cash_per_share DECIMAL(20,4),
  net_assets_per_share DECIMAL(20,4),
  total_assets_indicator DECIMAL(20,4),
  net_assets_indicator DECIMAL(20,4),
  is_anomaly TINYINT DEFAULT 0,
  PRIMARY KEY (stock_code, report_year, report_period)
) DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS subject_mapping (
  standard_name VARCHAR(50) PRIMARY KEY,
  synonyms JSON,
  target_table VARCHAR(50),
  target_column VARCHAR(50)
) DEFAULT CHARSET=utf8mb4;
