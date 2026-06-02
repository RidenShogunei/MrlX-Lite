"""
Mini tool-use environment for Main/Sub co-training.

The environment generates small product tables and tasks that require querying,
filtering, and arithmetic. It is intentionally local and deterministic: every
task has a verifier and every tool call can be executed without external APIs.
"""

import ast
import json
import operator
import random
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ToolTask:
    task_id: str
    question: str
    db_rows: List[Dict]
    expected_answer: str
    expected_value: float
    expected_name: str
    task_type: str
    difficulty: int


class ToolEnvironment:
    """Synthetic SQLite + calculator environment."""

    CATEGORIES = ["book", "toy", "food", "tool", "game"]
    REGIONS = ["north", "south", "east", "west"]

    def __init__(self, seed: int = 42, num_tasks: int = 100):
        self.seed = seed
        self.rng = random.Random(seed)
        self.tasks = [self._generate_task(i) for i in range(num_tasks)]

    def sample_tasks(self, n: int) -> List[ToolTask]:
        return self.rng.sample(self.tasks, min(n, len(self.tasks)))

    @classmethod
    def split(
        cls,
        train_n: int,
        val_n: int,
        test_n: int,
        seed: int = 42,
    ) -> Tuple[List[ToolTask], List[ToolTask], List[ToolTask]]:
        """Create deterministic non-overlapping train/val/test task splits."""
        total = train_n + val_n + test_n
        env = cls(seed=seed, num_tasks=total)
        train_end = train_n
        val_end = train_n + val_n
        return env.tasks[:train_end], env.tasks[train_end:val_end], env.tasks[val_end:]

    def _make_rows(self, task_idx: int, n: int = 8) -> List[Dict]:
        rows = []
        for i in range(n):
            rows.append({
                "id": i + 1,
                "name": f"item_{task_idx}_{i + 1}",
                "category": self.rng.choice(self.CATEGORIES),
                "region": self.rng.choice(self.REGIONS),
                "price": self.rng.randint(20, 300),
                "stock": self.rng.randint(5, 120),
                "rating": round(self.rng.uniform(3.0, 5.0), 1),
                "sales": self.rng.randint(10, 900),
            })
        return rows

    def _generate_task(self, idx: int) -> ToolTask:
        rows = self._make_rows(idx)
        task_type = self.rng.choice(["lowest_price", "discount", "revenue"])

        if task_type == "lowest_price":
            threshold = self.rng.randint(30, 80)
            min_rating = self.rng.choice([4.0, 4.2, 4.5])
            candidates = [r for r in rows if r["stock"] >= threshold and r["rating"] >= min_rating]
            if not candidates:
                candidates = rows
                threshold = min(r["stock"] for r in rows)
                min_rating = min(r["rating"] for r in rows)
            best = min(candidates, key=lambda r: (r["price"], r["id"]))
            question = (
                f"在 products 表中，找出 stock >= {threshold} 且 rating >= {min_rating} "
                "的最低价商品名称和价格。"
            )
            expected_value = float(best["price"])
            expected_answer = f"{best['name']}:{self._format_number(expected_value)}"
            return ToolTask(str(idx), question, rows, expected_answer, expected_value, best["name"], task_type, 1)

        if task_type == "discount":
            threshold = self.rng.randint(20, 70)
            candidates = [r for r in rows if r["stock"] >= threshold]
            if not candidates:
                candidates = rows
                threshold = min(r["stock"] for r in rows)
            best = max(candidates, key=lambda r: (r["price"], -r["id"]))
            discount = self.rng.choice([0.75, 0.8, 0.85, 0.9])
            value = round(best["price"] * discount, 2)
            question = (
                f"在 products 表中，找出 stock >= {threshold} 的最高价商品，"
                f"并计算它打 {discount:.2f} 折后的价格。返回 商品名:折后价。"
            )
            expected_answer = f"{best['name']}:{self._format_number(value)}"
            return ToolTask(str(idx), question, rows, expected_answer, value, best["name"], task_type, 2)

        category = self.rng.choice(self.CATEGORIES)
        candidates = [r for r in rows if r["category"] == category]
        if not candidates:
            rows[0]["category"] = category
            candidates = [rows[0]]
        best = max(candidates, key=lambda r: (r["price"] * r["sales"], -r["id"]))
        value = float(best["price"] * best["sales"])
        question = (
            f"在 products 表中，找出 category = '{category}' 的最高销售额商品。"
            "销售额定义为 price * sales，返回 商品名:销售额。"
        )
        expected_answer = f"{best['name']}:{self._format_number(value)}"
        return ToolTask(str(idx), question, rows, expected_answer, value, best["name"], task_type, 2)

    @staticmethod
    def _format_number(value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.2f}".rstrip("0").rstrip(".")

    @staticmethod
    def render_table(rows: List[Dict]) -> str:
        header = "id | name | category | region | price | stock | rating | sales"
        lines = [header]
        for r in rows:
            lines.append(
                f"{r['id']} | {r['name']} | {r['category']} | {r['region']} | "
                f"{r['price']} | {r['stock']} | {r['rating']} | {r['sales']}"
            )
        return "\n".join(lines)

    @staticmethod
    def build_sqlite(rows: List[Dict]) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE products (
                id INTEGER,
                name TEXT,
                category TEXT,
                region TEXT,
                price REAL,
                stock INTEGER,
                rating REAL,
                sales INTEGER
            )
            """
        )
        conn.executemany(
            "INSERT INTO products VALUES (:id, :name, :category, :region, :price, :stock, :rating, :sales)",
            rows,
        )
        return conn

    @staticmethod
    def safe_query(rows: List[Dict], sql: str) -> Tuple[bool, str]:
        sql_clean = sql.strip().rstrip(";")
        if not re.match(r"(?is)^select\s+", sql_clean):
            return False, "Only SELECT queries are allowed"
        if re.search(r"(?is)\b(insert|update|delete|drop|alter|create|attach|pragma)\b", sql_clean):
            return False, "Unsafe SQL keyword"
        if " limit " not in f" {sql_clean.lower()} ":
            sql_clean += " LIMIT 5"

        conn = ToolEnvironment.build_sqlite(rows)
        try:
            result = conn.execute(sql_clean).fetchall()
        except Exception as exc:
            return False, f"SQL error: {exc}"
        finally:
            conn.close()
        return True, json.dumps([dict(r) for r in result], ensure_ascii=False)

    @staticmethod
    def safe_calculate(expr: str) -> Tuple[bool, str]:
        expr = expr.strip()
        expr = expr.replace("×", "*").replace("÷", "/")
        expr = re.sub(r"[^0-9+\-*/().\s]", "", expr)
        if not expr:
            return False, "Empty expression"

        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def eval_node(node):
            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
                return ops[type(node.op)](eval_node(node.operand))
            if isinstance(node, ast.BinOp) and type(node.op) in ops:
                left = eval_node(node.left)
                right = eval_node(node.right)
                if isinstance(node.op, (ast.Div, ast.FloorDiv)) and abs(right) < 1e-12:
                    raise ZeroDivisionError
                return ops[type(node.op)](left, right)
            raise ValueError("unsupported expression")

        try:
            value = eval_node(ast.parse(expr, mode="eval"))
        except Exception as exc:
            return False, f"Calc error: {exc}"
        return True, ToolEnvironment._format_number(value)

    @staticmethod
    def parse_tool_call(text: str) -> Optional[Tuple[str, str]]:
        m = re.search(r"\[tool_call\]\s*(.*?)\s*\[/tool_call\]", text, re.DOTALL)
        if not m:
            return None
        body = m.group(1).strip()
        q = re.match(r"(?is)query\s*\((.*)\)\s*$", body)
        if q:
            return "query", q.group(1).strip().strip('"').strip("'")
        c = re.match(r"(?is)calculate\s*\((.*)\)\s*$", body)
        if c:
            return "calculate", c.group(1).strip().strip('"').strip("'")
        if body.lower().startswith("select"):
            return "query", body
        return "calculate", body

    @staticmethod
    def execute_tool(task: ToolTask, tool_call_text: str) -> Tuple[bool, str]:
        parsed = ToolEnvironment.parse_tool_call(tool_call_text)
        if parsed is None:
            return False, "No tool_call found"
        tool, arg = parsed
        if tool == "query":
            return ToolEnvironment.safe_query(task.db_rows, arg)
        if tool == "calculate":
            return ToolEnvironment.safe_calculate(arg)
        return False, f"Unknown tool: {tool}"

    @staticmethod
    def extract_final_answer(text: str) -> Tuple[Optional[str], Optional[float]]:
        m = re.search(r"<result>\s*(.*?)\s*</result>", text, re.DOTALL)
        if not m:
            return None, None
        body = m.group(1).strip()
        name_match = re.search(r"(item_\d+_\d+)", body)
        num_match = re.findall(r"-?\d+(?:\.\d+)?", body)
        name = name_match.group(1) if name_match else None
        value = float(num_match[-1]) if num_match else None
        return name, value

    @staticmethod
    def reward(task: ToolTask, response: str) -> float:
        name, value = ToolEnvironment.extract_final_answer(response)
        name_score = 0.4 if name == task.expected_name else 0.0
        value_score = 0.6 if value is not None and abs(value - task.expected_value) < 0.01 else 0.0
        return name_score + value_score

    @staticmethod
    def canonicalize_response(task: ToolTask, response: str) -> str:
        """Execute the model tool_call and build a verifier-friendly final result."""
        first = response.find("</result>")
        if first >= 0:
            response = response[:first + len("</result>")]

        parsed = ToolEnvironment.parse_tool_call(response)
        if parsed is None:
            return response
        tool, arg = parsed

        result_name = None
        result_value = None
        if tool == "query":
            ok, output = ToolEnvironment.safe_query(task.db_rows, arg)
            if ok:
                try:
                    rows = json.loads(output)
                except json.JSONDecodeError:
                    rows = []
                if rows:
                    row = rows[0]
                    result_name = row.get("name")
                    for key in ("discounted_price", "revenue", "price", "value"):
                        if key in row and row[key] is not None:
                            result_value = float(row[key])
                            break
                    if result_value is None and "price" in row and "sales" in row:
                        result_value = float(row["price"]) * float(row["sales"])
        elif tool == "calculate":
            ok, output = ToolEnvironment.safe_calculate(arg)
            if ok:
                result_value = float(output)

        if result_name is None:
            name, value = ToolEnvironment.extract_final_answer(response)
            result_name = name
            if result_value is None:
                result_value = value

        if result_name is None or result_value is None:
            return response

        value_text = ToolEnvironment._format_number(result_value)
        thinking = "执行工具调用并读取返回结果"
        tool_text = f"{tool}(\"{arg}\")"
        return (
            f"<thinking>{thinking}</thinking>"
            f"[tool_call]{tool_text}[/tool_call]"
            f"<result>{result_name}:{value_text}</result>"
        )


if __name__ == "__main__":
    env = ToolEnvironment(seed=42, num_tasks=3)
    for task in env.tasks:
        print("=" * 80)
        print(task.question)
        print(ToolEnvironment.render_table(task.db_rows))
        print("expected:", task.expected_answer)
