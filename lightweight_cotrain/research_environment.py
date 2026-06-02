"""Local mini deep-research environment for Main/Sub agent experiments.

This is a small, deterministic environment shaped like the M-GRPO deep-research
setting: a main agent plans and answers, while a sub agent executes search/read
tool calls over a local document collection.
"""

import json
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ResearchDoc:
    doc_id: str
    title: str
    text: str
    facts: Dict[str, str]


@dataclass
class ResearchTask:
    task_id: str
    question: str
    answer: str
    support_doc_ids: List[str]
    support_facts: List[str]
    task_type: str
    difficulty: int
    docs: List[ResearchDoc]


class MiniResearchEnvironment:
    """Synthetic document-search environment with verifiable evidence."""

    METHODS = [
        ("AuroraRank", "Dr. Lina Cho", "CivicQA", "cross-encoder reranking", 71.4),
        ("BeaconReader", "Prof. Amir Patel", "MedFact", "evidence-aware summarization", 76.8),
        ("CedarSearch", "Dr. Mei Tan", "OpenClaims", "iterative query reformulation", 68.2),
        ("DeltaTrace", "Dr. Nora Singh", "SciDocs", "citation graph tracing", 73.5),
        ("EmberGraph", "Prof. Hugo Vale", "BioLinks", "graph-guided retrieval", 79.1),
        ("FluxMemo", "Dr. Iris Wang", "NewsLens", "memory-augmented retrieval", 65.9),
        ("GraniteQA", "Prof. Omar Diaz", "PolicyBench", "decomposition prompting", 74.2),
        ("HarborLM", "Dr. Eva Rossi", "FinQA", "table-grounded reasoning", 70.6),
    ]

    def __init__(self, seed: int = 42, num_tasks: int = 100):
        self.seed = seed
        self.rng = random.Random(seed)
        self.docs = self._build_docs()
        self.tasks = [self._generate_task(i) for i in range(num_tasks)]

    @classmethod
    def split(
        cls,
        train_n: int,
        val_n: int,
        test_n: int,
        seed: int = 42,
    ) -> Tuple[List[ResearchTask], List[ResearchTask], List[ResearchTask]]:
        total = train_n + val_n + test_n
        env = cls(seed=seed, num_tasks=total)
        return env.tasks[:train_n], env.tasks[train_n:train_n + val_n], env.tasks[train_n + val_n:]

    def sample_tasks(self, n: int) -> List[ResearchTask]:
        return self.rng.sample(self.tasks, min(n, len(self.tasks)))

    def _build_docs(self) -> List[ResearchDoc]:
        docs = []
        for idx, (method, author, dataset, technique, score) in enumerate(self.METHODS):
            doc_id = f"D{idx + 1:03d}"
            title = f"{method}: retrieval study"
            facts = {
                "method": method,
                "author": author,
                "dataset": dataset,
                "technique": technique,
                "score": f"{score:.1f}",
            }
            text = (
                f"{title}\n"
                f"The {method} system was introduced by {author}. "
                f"It was evaluated on the {dataset} benchmark. "
                f"The core technique is {technique}. "
                f"In the reported main result, {method} achieved an F1 score of {score:.1f}. "
                f"The paper emphasizes reproducible evidence extraction."
            )
            docs.append(ResearchDoc(doc_id, title, text, facts))

        for idx, (_, _, dataset, _, _) in enumerate(self.METHODS):
            doc_id = f"B{idx + 1:03d}"
            title = f"{dataset} benchmark card"
            text = (
                f"{title}\n"
                f"{dataset} is a benchmark for multi-hop information seeking. "
                f"It contains questions that require retrieving evidence from multiple documents. "
                f"The benchmark card lists the dataset name as {dataset}."
            )
            docs.append(ResearchDoc(doc_id, title, text, {"dataset": dataset}))
        return docs

    def _generate_task(self, idx: int) -> ResearchTask:
        task_type = self.rng.choice(["author", "dataset_author", "comparison"])
        a = self.rng.randrange(len(self.METHODS))
        b = self.rng.randrange(len(self.METHODS))
        while b == a:
            b = self.rng.randrange(len(self.METHODS))

        method, author, dataset, _, score = self.METHODS[a]
        doc = self.docs[a]
        bench_doc = self.docs[len(self.METHODS) + a]

        if task_type == "author":
            question = f"Who introduced the {method} system?"
            return ResearchTask(
                str(idx), question, author, [doc.doc_id],
                [f"{method} system was introduced by {author}"],
                "author", 1, self.docs,
            )

        if task_type == "dataset_author":
            question = f"Which researcher introduced the system evaluated on {dataset}?"
            return ResearchTask(
                str(idx), question, author, [bench_doc.doc_id, doc.doc_id],
                [f"{dataset} benchmark", f"{method} system was introduced by {author}"],
                "dataset_author", 2, self.docs,
            )

        method_b, _, _, _, score_b = self.METHODS[b]
        doc_b = self.docs[b]
        if score >= score_b:
            answer = method
            support = [f"{method} achieved an F1 score of {score:.1f}", f"{method_b} achieved an F1 score of {score_b:.1f}"]
        else:
            answer = method_b
            support = [f"{method} achieved an F1 score of {score:.1f}", f"{method_b} achieved an F1 score of {score_b:.1f}"]
        question = f"Which system has the higher F1 score, {method} or {method_b}?"
        return ResearchTask(
            str(idx), question, answer, [doc.doc_id, doc_b.doc_id],
            support, "comparison", 2, self.docs,
        )

    @staticmethod
    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9. ]+", " ", text.lower())).strip()

    @staticmethod
    def parse_tool_call(text: str) -> Optional[Tuple[str, str]]:
        m = re.search(r"\[tool_call\]\s*(.*?)\s*\[/tool_call\]", text, re.DOTALL)
        if not m:
            return None
        body = m.group(1).strip()
        call = re.match(r"(?is)(search|read|quote)\s*\((.*)\)\s*$", body)
        if not call:
            return None
        tool = call.group(1).lower()
        arg = call.group(2).strip().strip('"').strip("'")
        return tool, arg

    @staticmethod
    def search(docs: List[ResearchDoc], query: str, k: int = 3) -> str:
        terms = set(MiniResearchEnvironment.normalize(query).split())
        scored = []
        for doc in docs:
            hay = MiniResearchEnvironment.normalize(f"{doc.title} {doc.text}")
            score = sum(1 for term in terms if term and term in hay)
            if score:
                snippet = f"{doc.title}. Use read(\"{doc.doc_id}\") to inspect this document."
                scored.append((score, doc.doc_id, doc.title, snippet))
        scored.sort(key=lambda x: (-x[0], x[1]))
        rows = [
            {"doc_id": doc_id, "title": title, "snippet": snippet}
            for _, doc_id, title, snippet in scored[:k]
        ]
        return json.dumps(rows, ensure_ascii=False)

    @staticmethod
    def read(docs: List[ResearchDoc], doc_id: str) -> Tuple[bool, str]:
        clean = doc_id.strip()
        if not re.fullmatch(r"[A-Z]\d{3}", clean):
            return False, "Invalid doc_id"
        for doc in docs:
            if doc.doc_id == clean:
                return True, json.dumps({"doc_id": doc.doc_id, "title": doc.title, "text": doc.text}, ensure_ascii=False)
        return False, "Unknown doc_id"

    @staticmethod
    def quote(docs: List[ResearchDoc], arg: str) -> Tuple[bool, str]:
        parts = [p.strip() for p in arg.split("|", 1)]
        if len(parts) != 2:
            return False, "quote expects doc_id|text"
        ok, raw = MiniResearchEnvironment.read(docs, parts[0])
        if not ok:
            return False, raw
        data = json.loads(raw)
        needle = MiniResearchEnvironment.normalize(parts[1])
        text_norm = MiniResearchEnvironment.normalize(data["text"])
        if needle and needle in text_norm:
            return True, json.dumps({"doc_id": data["doc_id"], "quote": parts[1]}, ensure_ascii=False)
        return False, "Quote not found in document"

    @staticmethod
    def execute_tool(task: ResearchTask, tool_call_text: str) -> Tuple[bool, str]:
        parsed = MiniResearchEnvironment.parse_tool_call(tool_call_text)
        if parsed is None:
            return False, "No valid tool_call found"
        tool, arg = parsed
        if tool == "search":
            return True, MiniResearchEnvironment.search(task.docs, arg)
        if tool == "read":
            return MiniResearchEnvironment.read(task.docs, arg)
        if tool == "quote":
            return MiniResearchEnvironment.quote(task.docs, arg)
        return False, f"Unknown tool: {tool}"

    @staticmethod
    def extract_result(text: str) -> str:
        m = re.search(r"<result>\s*(.*?)\s*</result>", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def extract_evidence_doc_ids(text: str) -> List[str]:
        return sorted(set(re.findall(r"\b[A-Z]\d{3}\b", text)))

    @staticmethod
    def reward(task: ResearchTask, response: str) -> Dict[str, float]:
        result = MiniResearchEnvironment.extract_result(response)
        answer_ok = MiniResearchEnvironment.normalize(task.answer) in MiniResearchEnvironment.normalize(result)
        doc_ids = set(MiniResearchEnvironment.extract_evidence_doc_ids(response))
        needed = set(task.support_doc_ids)
        evidence_hit = len(doc_ids & needed) / max(len(needed), 1)
        tool_valid = 1.0 if MiniResearchEnvironment.parse_tool_call(response) else 0.0
        total = 0.6 * float(answer_ok) + 0.3 * evidence_hit + 0.1 * tool_valid
        return {
            "total": total,
            "answer": float(answer_ok),
            "evidence": evidence_hit,
            "tool_valid": tool_valid,
        }


if __name__ == "__main__":
    env = MiniResearchEnvironment(seed=42, num_tasks=3)
    for task in env.tasks:
        print("=" * 80)
        print(task.question)
        print("answer:", task.answer)
        print("support:", ", ".join(task.support_doc_ids))
