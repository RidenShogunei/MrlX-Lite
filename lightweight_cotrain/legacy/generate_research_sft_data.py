"""Generate SFT data for the mini deep-research environment."""

import argparse
import json
from pathlib import Path

from research_environment import MiniResearchEnvironment, ResearchTask


MAIN_SYSTEM = (
    "You are the main research agent. Choose the next local tool call needed for research.\n"
    "Output exactly this format:\n"
    "<thinking>brief reason for the next action</thinking>\n"
    "[tool_call]search(\"query\") or read(\"DOCID\") or quote(\"DOCID|evidence text\")[/tool_call]\n"
    "Stop after [/tool_call]."
)

SUB_SYSTEM = (
    "You are the tool execution agent. Execute the given search/read/quote call over the local corpus.\n"
    "Output exactly this format:\n"
    "<thinking>execute tool</thinking>\n"
    "<result>tool observation</result>"
)

ANSWER_SYSTEM = (
    "You are the main research agent. Use the provided research history to answer the question.\n"
    "Output exactly this format:\n"
    "<thinking>brief evidence reasoning</thinking>\n"
    "<result>answer | evidence: DOCID, DOCID</result>\n"
    "Stop after </result>."
)


def _first_search_query(task: ResearchTask) -> str:
    if task.task_type == "author":
        return task.question.replace("Who introduced the ", "").replace(" system?", "")
    if task.task_type == "dataset_author":
        dataset = task.question.split(" evaluated on ", 1)[1].rstrip("?")
        return dataset
    return task.question.replace("Which system has the higher F1 score, ", "").rstrip("?")


def _main_tool_call(task: ResearchTask) -> str:
    return f'search("{_first_search_query(task)}")'


def _oracle_tool_calls(task: ResearchTask):
    calls = [_main_tool_call(task)]
    calls.extend(f'read("{doc_id}")' for doc_id in task.support_doc_ids)
    return calls


def _history_text(history):
    if not history:
        return "No observations yet."
    lines = []
    for idx, (call, observation) in enumerate(history, 1):
        lines.append(f"Step {idx} tool call: [tool_call]{call}[/tool_call]")
        lines.append(f"Step {idx} observation: {observation}")
    return "\n".join(lines)


def build_main_action_samples(task: ResearchTask):
    samples = []
    history = []
    calls = _oracle_tool_calls(task)
    for step, tool_call in enumerate(calls, 1):
        if step == 1:
            thinking = "Search for candidate documents before reading evidence."
        else:
            thinking = "Read the next supporting document to collect evidence."
        assistant = f"<thinking>{thinking}</thinking>[tool_call]{tool_call}[/tool_call]"
        samples.append({
            "messages": [
                {"role": "system", "content": MAIN_SYSTEM},
                {"role": "user", "content": f"Question: {task.question}\nResearch history:\n{_history_text(history)}"},
                {"role": "assistant", "content": assistant},
            ],
            "category": "main",
            "task_type": task.task_type,
            "stage": "action",
            "step": step,
        })
        ok, observation = MiniResearchEnvironment.execute_tool(task, f"[tool_call]{tool_call}[/tool_call]")
        assert ok
        history.append((tool_call, observation))
    return samples


def build_main_sample(task: ResearchTask):
    return build_main_action_samples(task)[0]


def build_main_answer_sample(task: ResearchTask):
    history = []
    for tool_call in _oracle_tool_calls(task):
        ok, observation = MiniResearchEnvironment.execute_tool(task, f"[tool_call]{tool_call}[/tool_call]")
        assert ok
        history.append((tool_call, observation))

    evidence = ", ".join(task.support_doc_ids)
    support_text = "; ".join(task.support_facts)
    assistant = (
        f"<thinking>The observations contain the answer and supporting document ids: {support_text}</thinking>"
        f"<result>{task.answer} | evidence: {evidence}</result>"
    )
    return {
        "messages": [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"Question: {task.question}\nResearch history:\n{_history_text(history)}"},
            {"role": "assistant", "content": assistant},
        ],
        "category": "main",
        "task_type": task.task_type,
        "stage": "answer",
    }


def build_sub_samples(task: ResearchTask):
    samples = []
    calls = _oracle_tool_calls(task)
    for doc_id, fact in zip(task.support_doc_ids, task.support_facts):
        calls.append(f'quote("{doc_id}|{fact}")')

    seen = set()
    for call in calls:
        if call in seen:
            continue
        seen.add(call)
        ok, result = MiniResearchEnvironment.execute_tool(task, f"[tool_call]{call}[/tool_call]")
        if not ok:
            continue
        samples.append({
            "messages": [
                {"role": "system", "content": SUB_SYSTEM},
                {"role": "user", "content": f"[tool_call]{call}[/tool_call]"},
                {"role": "assistant", "content": f"<thinking>execute tool</thinking><result>{result}</result>"},
            ],
            "category": "sub",
            "task_type": task.task_type,
        })
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Generate mini research SFT data.")
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--answer-fraction", type=float, default=0.25)
    parser.add_argument("--output", default="research_sft_data.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    env = MiniResearchEnvironment(seed=args.seed, num_tasks=args.tasks)
    samples = []
    answer_stride = max(int(1 / max(min(args.answer_fraction, 1.0), 1e-6)), 1)
    for idx, task in enumerate(env.tasks):
        samples.extend(build_main_action_samples(task))
        if args.answer_fraction >= 1.0 or idx % answer_stride == 0:
            samples.append(build_main_answer_sample(task))
        samples.extend(build_sub_samples(task))

    out = Path(__file__).parent / args.output
    with open(out, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[research-sft] wrote {len(samples)} samples to {out}")
    print(f"[research-sft] main={sum(1 for s in samples if s['category'] == 'main')}")
    print(f"[research-sft] sub={sum(1 for s in samples if s['category'] == 'sub')}")
    print(f"[research-sft] answer_fraction={args.answer_fraction}")


if __name__ == "__main__":
    main()
