"""
单题更新脚本
用法: python run_single.py <编号>
例如: python run_single.py B2001

只处理指定编号的题目，并更新输出文件中对应的行。
如果该编号在结果文件中不存在，则追加一行。
"""
import os
import sys
import json
import pandas as pd
from agent import process_question_group3
from config import RESULT_DIR

INPUT_FILE = os.getenv("INPUT_FILE", "examples/questions_sample.xlsx")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "examples/output_sample.xlsx")

os.makedirs(RESULT_DIR, exist_ok=True)


def run_single(target_qid: str):
    df_q = pd.read_excel(INPUT_FILE)

    # 找到目标题目
    mask = df_q["编号"].astype(str).str.strip() == target_qid
    matched = df_q[mask]
    if matched.empty:
        print(f"❌ 在 {INPUT_FILE} 中找不到编号 [{target_qid}]")
        sys.exit(1)

    row = matched.iloc[0]
    qid = str(row["编号"]).strip()
    q_type = str(row.get("问题类型", "")).strip()
    raw_q = str(row["问题"]).strip()

    try:
        questions = json.loads(raw_q)
    except Exception:
        questions = [{"Q": raw_q}]

    print(f"[{qid}] {q_type} — {len(questions)}轮，开始处理...")

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
    has_image = any(r.get("A", {}).get("image") for r in results)
    img_format = "折线图/柱状图/饼图" if has_image else "无"

    new_row = {
        "编号": qid,
        "问题": raw_q,
        "SQL查询语句": sql_str,
        "图形格式": img_format,
        "回答": answer_json,
    }

    # 读取已有结果文件（若存在）
    if os.path.exists(OUTPUT_FILE):
        df_out = pd.read_excel(OUTPUT_FILE)
    else:
        df_out = pd.DataFrame(columns=["编号", "问题", "SQL查询语句", "图形格式", "回答"])

    out_mask = df_out["编号"].astype(str).str.strip() == qid
    if out_mask.any():
        # 更新已有行
        for col, val in new_row.items():
            df_out.loc[out_mask, col] = val
        print(f"✅ 已更新 [{qid}] 的结果")
    else:
        # 追加新行
        df_out = pd.concat([df_out, pd.DataFrame([new_row])], ignore_index=True)
        print(f"✅ 已追加 [{qid}] 的结果")

    df_out.to_excel(OUTPUT_FILE, index=False)
    print(f"💾 已保存到 {OUTPUT_FILE}")

    # 额外保存单题结果，方便单独查看
    single_file = f"result_single_{qid}.xlsx"
    pd.DataFrame([new_row]).to_excel(single_file, index=False)
    print(f"📄 单题结果已保存到 {single_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python run_single.py <编号>")
        print("例如: python run_single.py B2001")
        sys.exit(1)

    run_single(sys.argv[1].strip())
