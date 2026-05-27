"""
数学推理环境
提供数学题生成、答案提取和正确性验证
"""

import random
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MathTask:
    """数学任务"""
    question: str
    answer: float
    steps: List[str]  # 解题步骤（用于 Main Agent 分解参考）
    difficulty: int  # 1-5


class MathEnvironment:
    """
    数学推理环境
    生成数学题，验证答案正确性
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.tasks = []
        self._generate_task_bank()

    def _generate_task_bank(self):
        """生成题目库"""
        # 简单算术
        self.tasks.extend(self._generate_arithmetic_tasks(20))
        # 代数方程
        self.tasks.extend(self._generate_algebra_tasks(15))
        # 应用题
        self.tasks.extend(self._generate_word_problems(15))

    def _generate_arithmetic_tasks(self, n: int) -> List[MathTask]:
        """生成算术题"""
        tasks = []
        for _ in range(n):
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            op = random.choice(['+', '-', '*', '/'])

            if op == '+':
                q = f"计算: {a} + {b}"
                ans = a + b
                steps = [f"将 {a} 和 {b} 相加", f"{a} + {b} = {ans}"]
            elif op == '-':
                a, b = max(a, b), min(a, b)
                q = f"计算: {a} - {b}"
                ans = a - b
                steps = [f"将 {a} 减去 {b}", f"{a} - {b} = {ans}"]
            elif op == '*':
                a = random.randint(2, 20)
                b = random.randint(2, 20)
                q = f"计算: {a} × {b}"
                ans = a * b
                steps = [f"将 {a} 和 {b} 相乘", f"{a} × {b} = {ans}"]
            else:
                b = random.randint(2, 10)
                ans = random.randint(2, 20)
                a = b * ans
                q = f"计算: {a} ÷ {b}"
                steps = [f"将 {a} 除以 {b}", f"{a} ÷ {b} = {ans}"]

            tasks.append(MathTask(q, float(ans), steps, 1))
        return tasks

    def _generate_algebra_tasks(self, n: int) -> List[MathTask]:
        """生成简单代数题"""
        tasks = []
        for _ in range(n):
            x = random.randint(2, 20)
            a = random.randint(2, 10)
            b = random.randint(1, 50)
            c = a * x + b

            q = f"解方程: {a}x + {b} = {c}"
            steps = [
                f"将等式两边减去 {b}: {a}x = {c - b}",
                f"将等式两边除以 {a}: x = {(c - b) // a}"
            ]
            tasks.append(MathTask(q, float(x), steps, 2))
        return tasks

    def _generate_word_problems(self, n: int) -> List[MathTask]:
        """生成应用题"""
        templates = [
            (
                "小明有 {a} 个苹果，小红有 {b} 个苹果，他们一共有多少个苹果？",
                lambda a, b: a + b,
                1
            ),
            (
                "一本书有 {a} 页，小明每天看 {b} 页，需要多少天看完？",
                lambda a, b: a / b,
                2
            ),
            (
                "一个长方形的长是 {a} 厘米，宽是 {b} 厘米，面积是多少平方厘米？",
                lambda a, b: a * b,
                2
            ),
            (
                "商店里有 {a} 个橙子，每袋装 {b} 个，可以装多少袋？还剩几个？",
                lambda a, b: a // b,
                3
            ),
            (
                "小明从家到学校有 {a} 米，他已经走了 {b} 米，还剩多少米？",
                lambda a, b: a - b,
                1
            ),
        ]

        tasks = []
        for _ in range(n):
            template, solver, diff = random.choice(templates)
            if diff == 1:
                a, b = random.randint(10, 100), random.randint(5, 50)
            elif diff == 2:
                a, b = random.randint(20, 200), random.randint(2, 20)
            else:
                a, b = random.randint(30, 100), random.randint(3, 10)

            q = template.format(a=a, b=b)
            ans = solver(a, b)
            steps = [f"理解题意，找出已知条件", f"进行计算得到答案 {ans}"]
            tasks.append(MathTask(q, float(ans), steps, diff))
        return tasks

    def sample_tasks(self, n: int = 10) -> List[MathTask]:
        """随机采样 n 个任务"""
        return random.sample(self.tasks, min(n, len(self.tasks)))

    def get_task(self) -> MathTask:
        """获取一个随机任务"""
        return random.choice(self.tasks)

    @staticmethod
    def extract_number(text: str) -> Optional[float]:
        """从文本中提取数字（答案）"""
        # 尝试匹配 "答案是 X" 或 "= X" 或 "X"
        patterns = [
            r'(?:答案|answer|result)(?:是|:|=)\s*(-?\d+\.?\d*)',
            r'(?:=|等于)\s*(-?\d+\.?\d*)',
            r'<result>(-?\d+\.?\d*)</result>',
            r'(?:^|\s)(-?\d+\.?\d*)(?:\s*$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        # 回退：找最后一个数字
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        return None

    @staticmethod
    def check_answer(pred: Optional[float], target: float, tolerance: float = 0.01) -> bool:
        """检查答案是否正确"""
        if pred is None:
            return False
        return abs(pred - target) < tolerance

    @staticmethod
    def compute_accuracy_reward(pred: Optional[float], target: float) -> float:
        """基于准确率的奖励"""
        if pred is None:
            return 0.0
        diff = abs(pred - target)
        if diff < 0.01:
            return 5.0  # 完全正确
        elif diff < 0.1:
            return 2.0  # 接近
        elif diff < 1.0:
            return 0.5  # 有点接近
        else:
            return 0.0  # 错误


class MathReward:
    """
    对齐 MrlX-DeepResearch 原始设计的奖励函数

    核心原则：
    1. 二值制格式检查 — 格式错误 → 0.0（无论答案对错）
    2. Main Agent: 格式正确+答案正确→1.0，格式正确+答案错误→0.1
    3. Sub Agent: 格式错误→0.0，格式正确→继承Main分数
    4. 无平滑过渡 — 完全对齐原始设计
    """

    FORMAT_SCORE = 0.1  # 格式正确但答案错误的安慰分

    # ── 格式验证（状态机，对齐原始 MrlX） ──

    @staticmethod
    def _validate_main_format(text: str) -> bool:
        """
        Main Agent 状态机格式验证

        要求: <thinking> → [tool_call] → <result>  严格顺序
        标签成对，内容只能出现在标签内部，终态为 end
        """
        tags_to_check = ["thinking", "tool_call", "result"]
        has_alt_tool = ("[tool_call]" in text and "[/tool_call]" in text)

        for tag in tags_to_check:
            if tag == "tool_call":
                if has_alt_tool:
                    continue
                if f"<{tag}>" in text or f"</{tag}>" in text:
                    if text.count(f"<{tag}>") != text.count(f"</{tag}>"):
                        return False
            else:
                if text.count(f"<{tag}>") != text.count(f"</{tag}>"):
                    return False

        pattern = r"(</?(?:thinking|tool_call|result)>)"
        if has_alt_tool:
            pattern = r"(</?(?:thinking|tool_call|result)>|\[/?tool_call\])"

        parts = re.split(pattern, text)
        state = "start"

        for part in parts:
            if not part.strip():
                continue

            is_tag = re.match(r"</?(?:thinking|tool_call|result)>|\[/?tool_call\]", part)
            if is_tag:
                tag_str = part.strip()
                if tag_str == "<thinking>" and state in ("start",):
                    state = "in_thinking"
                elif tag_str == "</thinking>" and state == "in_thinking":
                    state = "after_thinking"
                elif tag_str in ("<tool_call>", "[tool_call]") and state in ("start", "after_thinking"):
                    state = "in_tool_call"
                elif tag_str in ("</tool_call>", "[/tool_call]") and state == "in_tool_call":
                    state = "after_tool_call"
                elif tag_str == "<result>" and state in ("after_thinking", "after_tool_call"):
                    state = "in_result"
                elif tag_str == "</result>" and state == "in_result":
                    state = "end"
                else:
                    return False
            else:
                if state not in ("in_thinking", "in_tool_call", "in_result"):
                    return False

        return state == "end"

    @staticmethod
    def _validate_sub_format(text: str) -> bool:
        """
        Sub Agent 状态机格式验证

        要求: <thinking> → <result>  严格顺序
        终态为 end
        """
        for tag in ["thinking", "result"]:
            if text.count(f"<{tag}>") != text.count(f"</{tag}>"):
                return False

        pattern = r"(</?(?:thinking|result)>)"
        parts = re.split(pattern, text)
        state = "start"

        for part in parts:
            if not part.strip():
                continue

            is_tag = re.match(r"</?(?:thinking|result)>", part)
            if is_tag:
                tag_str = part.strip()
                if tag_str == "<thinking>" and state == "start":
                    state = "in_thinking"
                elif tag_str == "</thinking>" and state == "in_thinking":
                    state = "after_thinking"
                elif tag_str == "<result>" and state in ("after_thinking", "start"):
                    state = "in_result"
                elif tag_str == "</result>" and state == "in_result":
                    state = "end"
                else:
                    return False
            else:
                if state not in ("in_thinking", "in_result"):
                    return False

        return state == "end"

    # ── 奖励计算（完全对齐原始 MrlX） ──

    @staticmethod
    def compute_main_reward(task: MathTask, main_response: str, sub_results: List[str]) -> float:
        """
        Main Agent 奖励 — 对齐原始 MrlX compute_score_em
        """
        # 截断到最后一个 </result>，忽略后面的垃圾文本
        last_result = main_response.rfind("</result>")
        if last_result != -1:
            main_response = main_response[:last_result + len("</result>")]

        is_valid_format = MathReward._validate_main_format(main_response)

        if not is_valid_format:
            return 0.0

        pred = MathEnvironment.extract_number(main_response)

        if pred is None:
            return MathReward.FORMAT_SCORE

        if MathEnvironment.check_answer(pred, task.answer):
            return 1.0
        else:
            return MathReward.FORMAT_SCORE

    @staticmethod
    def compute_sub_reward(subtask: str, sub_response: str, main_score: float, task_answer: float) -> float:
        """
        Sub Agent 奖励 — 对齐原始 MrlX sub_agent_multiturn.reward_func
        """
        last_result = sub_response.rfind("</result>")
        if last_result != -1:
            sub_response = sub_response[:last_result + len("</result>")]

        is_valid_format = MathReward._validate_sub_format(sub_response)

        if not is_valid_format:
            return 0.0

        if main_score == 0.0:
            return MathReward.FORMAT_SCORE
        elif main_score <= MathReward.FORMAT_SCORE:
            return MathReward.FORMAT_SCORE
        else:
            return 1.0
