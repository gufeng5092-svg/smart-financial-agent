"""项目配置。

所有敏感信息和本地路径都从环境变量读取。开发时可复制 `.env.example`
为 `.env`，并填写自己的 MySQL、通义千问和 Dify 配置。
"""
import os

from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "finance_qa"),
    "charset": os.getenv("DB_CHARSET", "utf8mb4"),
}

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-max")
QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

RESULT_DIR = os.getenv("RESULT_DIR", "./result")
REPORT_BASE_DIR = os.getenv("REPORT_BASE_DIR", "./data/reports")

# Dify 知识库配置（任务三）
DIFY_API_KEY = os.getenv("DIFY_API_KEY", "")
DIFY_DATASET_ID = os.getenv("DIFY_DATASET_ID", "")
DIFY_URL = os.getenv("DIFY_URL", "http://localhost/v1")
