"""
SFT 数据生成器
为 Main Agent 和 Sub Agent 生成高质量的监督微调数据
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

from math_environment import MathEnvironment


@dataclass
class SFTSample:
    """SFT 样本"""
    messages: List[Dict[str, str]]
    category: str  # "main" or "sub"


class SFTTaskGenerator:
    """生成 SFT 训练任务"""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.main_prompt_template = (
            "你是数学解题器。按如下格式输出，不要遗漏任何部分：\n\n"
            "<thinking>\n"
            "简要分析问题\n"
            "</thinking>\n"
            "[tool_call]\n"
            "计算的具体内容，如: 计算 61 × 19\n"
            "[/tool_call]\n"
            "<result>\n"
            "数字答案\n"
            "</result>\n\n"
            "示例 - 输入: 长方形的长6宽3，面积?\n"
            "示例 - 输出:\n"
            "<thinking>\n"
            "面积=长×宽\n"
            "</thinking>\n"
            "[tool_call]\n"
            "计算 6 × 3\n"
            "[/tool_call]\n"
            "<result>\n"
            "18\n"
            "</result>"
        )

        self.sub_prompt_template = (
            "你是一个计算执行器。你的任务是执行给定的计算子任务并返回精确的数字结果。\n\n"
            "格式要求：\n"
            "<thinking>简要写出你的计算过程</thinking>\n"
            "<result>数字答案</result>\n\n"
            "规则：\n"
            "1. 只执行被分配的具体计算，不要分析原问题\n"
            "2. <result> 中必须只包含数字，不要带单位或解释文字\n"
            "3. 如果子任务是乘法，直接算出来；是加减法，直接算出来"
        )

    @staticmethod
    def _format_number(value) -> str:
        value = float(value)
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.2f}".rstrip("0").rstrip(".")

    @staticmethod
    def _calc_from_question(question: str, answer) -> str:
        if question.startswith("计算:"):
            return question.replace("计算:", "", 1).strip()

        m = re.search(r"解方程:\s*(.+)", question)
        if m:
            return f"{m.group(1)} -> x = {SFTTaskGenerator._format_number(answer)}"

        nums = re.findall(r"\d+", question)
        if "一共有" in question and len(nums) >= 2:
            return f"{nums[0]} + {nums[1]}"
        if "每天看" in question and len(nums) >= 2:
            return f"{nums[0]} ÷ {nums[1]}"
        if "面积" in question and len(nums) >= 2:
            return f"{nums[0]} × {nums[1]}"
        if "每袋装" in question and len(nums) >= 2:
            return f"{nums[0]} // {nums[1]}"
        if "还剩" in question and len(nums) >= 2:
            return f"{nums[0]} - {nums[1]}"

        return f"计算得到 {SFTTaskGenerator._format_number(answer)}"

    def generate_environment_samples(self) -> List[SFTSample]:
        """把数学环境里的 50 道题也转成严格格式 SFT 样本。"""
        samples = []
        env = MathEnvironment(seed=42)
        for task in env.tasks:
            answer = self._format_number(task.answer)
            calc = self._calc_from_question(task.question, task.answer)
            thinking = task.steps[0] if task.steps else "识别题目中的运算关系"

            main_messages = [
                {"role": "system", "content": self.main_prompt_template},
                {"role": "user", "content": f"问题: {task.question}"},
                {"role": "assistant", "content": (
                    f"<thinking>\n{thinking}\n</thinking>\n"
                    f"[tool_call]\n{calc}\n[/tool_call]\n"
                    f"<result>\n{answer}\n</result>"
                )}
            ]
            samples.append(SFTSample(messages=main_messages, category="main"))

            sub_messages = [
                {"role": "system", "content": self.sub_prompt_template},
                {"role": "user", "content": f"计算任务: 计算 {calc}\n请执行并给出结果:"},
                {"role": "assistant", "content": (
                    f"<thinking>\n执行计算 {calc}，结果是 {answer}\n</thinking>\n"
                    f"<result>\n{answer}\n</result>"
                )}
            ]
            samples.append(SFTSample(messages=sub_messages, category="sub"))

        return samples

    def generate_main_samples(self) -> List[SFTSample]:
        """生成 Main Agent SFT 数据"""
        samples = []

        tasks = [
            ("一个长方形的长是 6 厘米，宽是 3 厘米，面积是多少平方厘米？", "6 × 3", 18, "面积=长×宽"),
            ("一个长方形的长是 8 米，宽是 5 米，面积是多少平方米？", "8 × 5", 40, "面积=长×宽"),
            ("一个长方形的长是 12 厘米，宽是 4 厘米，面积是多少平方厘米？", "12 × 4", 48, "面积=长×宽"),
            ("一个长方形的长是 10 米，宽是 7 米，面积是多少平方米？", "10 × 7", 70, "面积=长×宽"),
            ("计算: 25 + 17", "25 + 17", 42, "加法运算"),
            ("计算: 86 - 39", "86 - 39", 47, "减法运算"),
            ("计算: 7 × 8", "7 × 8", 56, "乘法运算"),
            ("计算: 72 ÷ 9", "72 ÷ 9", 8, "除法运算"),
            ("计算: 15 + 28 + 42", "15 + 28 + 42", 85, "连加运算"),
            ("计算: 100 - 37 - 28", "100 - 37 - 28", 35, "连减运算"),
            ("解方程: x + 5 = 12", "x = 12 - 5", 7, "移项求解"),
            ("解方程: x - 3 = 10", "x = 10 + 3", 13, "移项求解"),
            ("解方程: 2x = 16", "x = 16 ÷ 2", 8, "系数化1"),
            ("解方程: x ÷ 4 = 6", "x = 6 × 4", 24, "系数化1"),
            ("小明有 25 个苹果，小红有 18 个苹果，他们一共有多少个苹果？", "25 + 18", 43, "加法应用"),
            ("一本书有 100 页，小明每天看 20 页，需要多少天看完？", "100 ÷ 20", 5, "除法应用"),
            ("商店里有 48 个橙子，每袋装 6 个，可以装多少袋？", "48 ÷ 6", 8, "除法应用"),
            ("小明从家到学校有 150 米，他已经走了 75 米，还剩多少米？", "150 - 75", 75, "减法应用"),
            ("一个正方形的边长是 9 厘米，面积是多少平方厘米？", "9 × 9", 81, "正方形面积"),
            ("一个三角形的底是 8 厘米，高是 6 厘米，面积是多少平方厘米？", "8 × 6 ÷ 2", 24, "三角形面积"),
        ]

        for question, calc, answer, thinking in tasks:
            messages = [
                {"role": "system", "content": self.main_prompt_template},
                {"role": "user", "content": f"问题: {question}"},
                {"role": "assistant", "content": f"<thinking>\n{thinking}\n</thinking>\n[tool_call]\n{calc}\n[/tool_call]\n<result>\n{answer}\n</result>"}
            ]
            samples.append(SFTSample(messages=messages, category="main"))

        return samples

    def generate_sub_samples(self) -> List[SFTSample]:
        """生成 Sub Agent SFT 数据"""
        samples = []

        calculations = [
            ("计算 6 × 3", "6 × 3", "6乘以3等于18", "18"),
            ("计算 8 × 5", "8 × 5", "8乘以5等于40", "40"),
            ("计算 12 × 4", "12 × 4", "12乘以4等于48", "48"),
            ("计算 10 × 7", "10 × 7", "10乘以7等于70", "70"),
            ("计算 25 + 17", "25 + 17", "25加17等于42", "42"),
            ("计算 86 - 39", "86 - 39", "86减39等于47", "47"),
            ("计算 7 × 8", "7 × 8", "7乘以8等于56", "56"),
            ("计算 72 ÷ 9", "72 ÷ 9", "72除以9等于8", "8"),
            ("计算 15 + 28", "15 + 28", "15加28等于43", "43"),
            ("计算 100 - 37", "100 - 37", "100减37等于63", "63"),
            ("计算 9 × 9", "9 × 9", "9乘以9等于81", "81"),
            ("计算 8 × 6 ÷ 2", "8 × 6 ÷ 2", "先算8乘6等于48，再除以2等于24", "24"),
            ("计算 81 ÷ 9", "81 ÷ 9", "81除以9等于9", "9"),
            ("计算 45 + 55", "45 + 55", "45加55等于100", "100"),
            ("计算 64 - 28", "64 - 28", "64减28等于36", "36"),
            ("计算 13 × 4", "13 × 4", "13乘以4等于52", "52"),
            ("计算 96 ÷ 6", "96 ÷ 6", "96除以6等于16", "16"),
            ("计算 184 × 12", "184 × 12", "184乘以12等于2208", "2208"),
            ("计算 64 ÷ 4", "64 ÷ 4", "64除以4等于16", "16"),
            ("计算 98 ÷ 7", "98 ÷ 7", "98除以7等于14", "14"),
            ("计算 30 ÷ 3", "30 ÷ 3", "30除以3等于10", "10"),
            ("计算 69 - 39", "69 - 39", "69减39等于30", "30"),
            ("计算 81 - 17", "81 - 17", "81减17等于64", "64"),
            ("计算 113 - 15", "113 - 15", "113减15等于98", "98"),
            ("计算 36 ÷ 3", "36 ÷ 3", "36除以3等于12", "12"),
        ]

        for task, calc, thinking, result in calculations:
            messages = [
                {"role": "system", "content": self.sub_prompt_template},
                {"role": "user", "content": f"计算任务: {task}\n请执行并给出结果:"},
                {"role": "assistant", "content": f"<thinking>\n{thinking}\n</thinking>\n<result>\n{result}\n</result>"}
            ]
            samples.append(SFTSample(messages=messages, category="sub"))

        return samples

    def generate_all_samples(self) -> List[SFTSample]:
        """生成所有 SFT 数据"""
        main_samples = self.generate_main_samples()
        sub_samples = self.generate_sub_samples()
        env_samples = self.generate_environment_samples()
        return main_samples + sub_samples + env_samples

    def save_to_jsonl(self, output_path: str):
        """保存为 JSONL 格式"""
        samples = self.generate_all_samples()

        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                # 转换为标准格式
                item = {
                    "messages": sample.messages,
                    "category": sample.category
                }
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"[SFT] 已生成 {len(samples)} 条 SFT 数据")
        print(f"[SFT] Main Agent: {len([s for s in samples if s.category == 'main'])} 条")
        print(f"[SFT] Sub Agent: {len([s for s in samples if s.category == 'sub'])} 条")
        print(f"[SFT] 保存至: {output_path}")


if __name__ == "__main__":
    generator = SFTTaskGenerator(seed=42)

    # 生成 SFT 数据
    save_path = Path(__file__).parent / "sft_data.jsonl"
    generator.save_to_jsonl(str(save_path))

    # 打印示例
    samples = generator.generate_all_samples()
    print("\n" + "="*60)
    print("Main Agent 示例:")
    print("="*60)
    main_sample = [s for s in samples if s.category == "main"][0]
    for msg in main_sample.messages:
        print(f"\n[{msg['role']}]")
        print(msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content'])

    print("\n" + "="*60)
    print("Sub Agent 示例:")
    print("="*60)
    sub_sample = [s for s in samples if s.category == "sub"][0]
    for msg in sub_sample.messages:
        print(f"\n[{msg['role']}]")
        print(msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content'])
