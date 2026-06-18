"""
从company_name自动提取short_name并补全到数据库
规则：去掉"股份有限公司"、"有限公司"、"集团"等后缀，保留核心品牌名
"""
import re
import pymysql
from config import DB_CONFIG

# 常见后缀，按长度降序排列（先匹配长的）
SUFFIXES = [
    '股份有限公司', '有限责任公司', '有限公司',
    '集团股份有限', '集团有限责任', '集团股份', '集团有限',
    '医药科技', '医药控股', '科技集团', '药业集团',
    '股份', '集团', '医药', '药业', '制药',
    '（股份）', '(股份)',
]

# 常见地名前缀（2-3字），去掉后保留品牌核心
PROVINCE_PREFIXES = [
    '云南', '山东', '成都', '广州', '北京', '上海', '浙江', '江苏',
    '湖南', '湖北', '四川', '广东', '福建', '河南', '河北', '陕西',
    '贵州', '安徽', '江西', '辽宁', '吉林', '黑龙江', '内蒙古',
    '新疆', '西藏', '宁夏', '甘肃', '青海', '海南', '重庆', '天津',
    '深圳', '杭州', '南京', '武汉', '西安', '长沙', '沈阳', '哈尔滨',
    '通化', '亳州', '漳州', '桂林',
]

def extract_short_name(company_name: str) -> str:
    """从全称提取简称"""
    name = company_name.strip()
    # 去掉括号内容（如"（集团）"）
    name = re.sub(r'[（(][^）)]*[）)]', '', name)
    # 去掉后缀（多轮，直到不能再去）
    changed = True
    while changed:
        changed = False
        for suffix in SUFFIXES:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                changed = True
                break
    name = name.strip()
    # 如果结果超过4字且以地名开头，尝试去掉地名前缀
    if len(name) > 4:
        for prefix in PROVINCE_PREFIXES:
            if name.startswith(prefix) and len(name) - len(prefix) >= 2:
                name = name[len(prefix):]
                break
    return name.strip() if name.strip() else company_name


def fix_short_names():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("SELECT stock_code, company_name, short_name FROM company_info")
    rows = cursor.fetchall()

    updated = 0
    for stock_code, company_name, short_name in rows:
        if short_name:  # 已有简称，跳过
            continue
        new_short = extract_short_name(company_name)
        cursor.execute(
            "UPDATE company_info SET short_name = %s WHERE stock_code = %s",
            (new_short, stock_code)
        )
        updated += 1

    conn.commit()
    cursor.close()
    conn.close()
    print(f"已补全 {updated} 家公司的简称")

    # 验证
    conn2 = pymysql.connect(**DB_CONFIG)
    cursor2 = conn2.cursor()
    cursor2.execute("SELECT stock_code, company_name, short_name FROM company_info LIMIT 10")
    for row in cursor2.fetchall():
        print(f"  {row[0]}  {row[1]}  →  {row[2]}")
    cursor2.close()
    conn2.close()


if __name__ == '__main__':
    fix_short_names()
