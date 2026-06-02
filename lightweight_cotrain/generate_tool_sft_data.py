"""Generate SFT data for the mini tool-use environment."""

import json
import re
from pathlib import Path

from tool_environment import ToolEnvironment


MAIN_SYSTEM = (
    "你是工具规划 agent。根据 products 表和问题，输出严格格式：\n"
    "<thinking>简要说明需要查询或计算什么</thinking>\n"
    "[tool_call]query(\"SELECT ...\") 或 calculate(\"表达式\")[/tool_call]\n"
    "<result>商品名:数字</result>\n"
    "只输出一段结果，写完 </result> 后停止。"
)

SUB_SYSTEM = (
    "你是工具执行 agent。执行给定 query/calculate 工具调用，输出严格格式：\n"
    "<thinking>执行工具</thinking>\n"
    "<result>工具返回值</result>"
)


def _sql_for_task(task):
    if task.task_type == "lowest_price":
        m = re.search(r"stock >= (\d+) 且 rating >= ([0-9.]+)", task.question)
        stock, rating = m.group(1), m.group(2)
        return (
            "SELECT name, price FROM products "
            f"WHERE stock >= {stock} AND rating >= {rating} "
            "ORDER BY price ASC, id ASC LIMIT 1"
        )
    if task.task_type == "discount":
        m = re.search(r"stock >= (\d+).*打 ([0-9.]+) 折", task.question)
        stock, discount = m.group(1), m.group(2)
        return (
            "SELECT name, price FROM products "
            f"WHERE stock >= {stock} "
            "ORDER BY price DESC, id ASC LIMIT 1"
        ), float(discount)
    if task.task_type == "revenue":
        m = re.search(r"category = '([^']+)'", task.question)
        category = m.group(1)
        return (
            "SELECT name, price, sales, price * sales AS revenue FROM products "
            f"WHERE category = '{category}' "
            "ORDER BY revenue DESC, id ASC LIMIT 1"
        )
    raise ValueError(f"Unknown task type: {task.task_type}")


def build_main_sample(task):
    table = ToolEnvironment.render_table(task.db_rows)
    user = f"products 表：\n{table}\n\n问题：{task.question}"

    if task.task_type == "discount":
        sql, discount = _sql_for_task(task)
        tool_call = f'query("{sql}")'
        thinking = "先查询满足库存条件的最高价商品，再计算折扣价"
    else:
        sql = _sql_for_task(task)
        tool_call = f'query("{sql}")'
        thinking = "用 SQL 查询满足条件的目标商品"

    assistant = (
        f"<thinking>{thinking}</thinking>"
        f"[tool_call]{tool_call}[/tool_call]"
        f"<result>{task.expected_answer}</result>"
    )
    return {
        "messages": [
            {"role": "system", "content": MAIN_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "category": "main",
        "task_type": task.task_type,
    }


def build_sub_sample(task):
    if task.task_type == "discount":
        sql, discount = _sql_for_task(task)
        ok, query_out = ToolEnvironment.safe_query(task.db_rows, sql)
        assert ok
        calc = f"{task.expected_value / discount} * {discount}"
        ok, calc_out = ToolEnvironment.safe_calculate(calc)
        assert ok
        tool_call = f'calculate("{calc}")'
        result = calc_out
    else:
        sql = _sql_for_task(task)
        ok, result = ToolEnvironment.safe_query(task.db_rows, sql)
        assert ok
        tool_call = f'query("{sql}")'

    return {
        "messages": [
            {"role": "system", "content": SUB_SYSTEM},
            {"role": "user", "content": f"[tool_call]{tool_call}[/tool_call]"},
            {"role": "assistant", "content": f"<thinking>执行工具调用</thinking><result>{result}</result>"},
        ],
        "category": "sub",
        "task_type": task.task_type,
    }


def main():
    env = ToolEnvironment(seed=42, num_tasks=200)
    samples = []
    for task in env.tasks:
        samples.append(build_main_sample(task))
        samples.append(build_sub_sample(task))

    out = Path(__file__).parent / "tool_sft_data.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[tool-sft] wrote {len(samples)} samples to {out}")
    print(f"[tool-sft] main={sum(1 for s in samples if s['category']=='main')}")
    print(f"[tool-sft] sub={sum(1 for s in samples if s['category']=='sub')}")


if __name__ == "__main__":
    main()
