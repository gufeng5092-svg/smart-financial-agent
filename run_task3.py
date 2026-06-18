"""
任务三批量处理脚本
读取样例问题 → 调用增强版Agent → 生成样例输出
"""
import os
import json
import time
import pandas as pd
from agent import process_question_group3
from config import RESULT_DIR

INPUT_FILE = os.getenv("INPUT_FILE", "examples/questions_sample.xlsx")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "examples/output_sample.xlsx")

os.makedirs(RESULT_DIR, exist_ok=True)


def main():
    df_q = pd.read_excel(INPUT_FILE)
    print(f"共 {len(df_q)} 道题")

    rows = []
    for _, row in df_q.iterrows():
        qid = str(row["编号"]).strip()
        q_type = str(row.get("问题类型", "")).strip()
        raw_q = str(row["问题"]).strip()

        # 解析问题列表
        try:
            questions = json.loads(raw_q)
        except Exception:
            questions = [{"Q": raw_q}]

        print(f"\n[{qid}] {q_type} — {len(questions)}轮")

        try:
            results, sql_list = process_question_group3(qid, questions)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            results = [{"Q": q.get("Q", ""), "A": {"content": f"处理出错: {e}"}} for q in questions]
            sql_list = []

        answer_json = json.dumps(results, ensure_ascii=False)
        sql_str = "\n\n".join(sql_list) if sql_list else ""

        # 提取图表格式标记
        has_image = any(
            r.get("A", {}).get("image") for r in results
        )
        img_format = "折线图/柱状图/饼图" if has_image else "无"

        rows.append({
            "编号": qid,
            "问题": raw_q,
            "SQL查询语句": sql_str,
            "图形格式": img_format,
            "回答": answer_json,
        })

        # 避免API限速
        time.sleep(1)

    df_out = pd.DataFrame(rows, columns=["编号", "问题", "SQL查询语句", "图形格式", "回答"])
    df_out.to_excel(OUTPUT_FILE, index=False)
    print(f"\n✅ 已保存到 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
