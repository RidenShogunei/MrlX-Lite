"""GRPO for HotpotQA using TRL's GRPOTrainer.

This uses the official TRL implementation which handles:
- Efficient log prob computation
- DeepSpeed ZeRO-3 integration
- Batched generation and reward computation
"""

import argparse
import re
import string
from pathlib import Path
from typing import List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from hotpotqa_environment import HotpotQAEnvironment


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[' + string.punctuation + ']', '', s)
    s = ' '.join(s.split())
    return s


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens) if pred_tokens else 0
    rec = len(common) / len(gold_tokens) if gold_tokens else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def extract_answer(text: str) -> str:
    """Extract answer from <result> block."""
    m = re.search(r'<result>(.*?)</result>', text, re.DOTALL)
    if not m:
        return text.strip()
    result = m.group(1).strip()
    parts = result.split('| evidence:')
    return parts[0].strip()


def extract_evidence(text: str) -> List[str]:
    m = re.search(r'evidence:\s*([A-Z0-9,\s]+)', text)
    if not m:
        return []
    return [x.strip() for x in m.group(1).split(',') if x.strip()]


def hotpotqa_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Reward function for TRL GRPOTrainer.
    
    TRL calls this with batches of prompts and completions.
    We need to extract the answer and compare with ground truth.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # Extract ground truth from prompt
        # Prompt format: "...Question: XXX\n\nDocuments:...\n\nAnswer:"
        q_match = re.search(r'Question:\s*(.+?)\n\nDocuments:', prompt, re.DOTALL)
        question = q_match.group(1).strip() if q_match else ""
        
        # Find the answer in the environment data
        # For simplicity, we parse the gold answer from the prompt's system message
        # In practice, you'd pass the gold answer through the dataset
        
        pred_answer = extract_answer(completion)
        pred_evidence = extract_evidence(completion)
        
        # Format reward
        has_result = '<result>' in completion and '</result>' in completion
        has_evidence = 'evidence:' in completion.lower()
        format_reward = 0.3 * has_result + 0.2 * has_evidence
        
        # We can't compute F1 without gold answer here
        # TRL's reward_func signature doesn't pass extra metadata
        # So we return format reward only, or use a wrapper dataset
        
        reward = format_reward
        rewards.append(reward)
    
    return rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the main coordinator agent. Use the research results to answer the question.\n"
    "Output exactly this format:\n"
    "<thinking>brief reasoning</thinking>\n"
    "<result>answer | evidence: DOCID, DOCID</result>\n"
    "Stop after </result>."
)


def build_prompt(task) -> str:
    docs_text = "\n\n".join([
        f"Document {d.doc_id} ({d.title}):\n{d.text[:500]}"
        for d in task.docs[:5]
    ])
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {task.question}\n\n"
        f"Documents:\n{docs_text}\n\n"
        f"Answer:"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="/home/jinxu/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B")
    parser.add_argument("--main-lora", default="./hotpotqa_qwen35_9b_sft_500x1/main_agent")
    parser.add_argument("--train-jsonl", default="./hotpotqa_data_enhanced/train.jsonl")
    parser.add_argument("--val-jsonl", default="./hotpotqa_data_enhanced/val.jsonl")
    parser.add_argument("--save-dir", default="./hotpotqa_qwen35_9b_grpo_trl")
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--val-tasks", type=int, default=20)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading data...")
    train_env = HotpotQAEnvironment.from_jsonl(args.train_jsonl, limit=args.tasks)
    val_env = HotpotQAEnvironment.from_jsonl(args.val_jsonl, limit=args.val_tasks)

    # Build dataset with prompts and gold answers
    train_data = []
    for task in train_env.tasks:
        train_data.append({
            "prompt": build_prompt(task),
            "gold_answer": task.answer,
            "gold_doc_ids": task.support_doc_ids,
        })
    
    val_data = []
    for task in val_env.tasks:
        val_data.append({
            "prompt": build_prompt(task),
            "gold_answer": task.answer,
            "gold_doc_ids": task.support_doc_ids,
        })

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # Reward function that uses gold answer from dataset
    def reward_func(prompts, completions, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            pred_answer = extract_answer(completion)
            pred_evidence = extract_evidence(completion)
            
            # Find matching task to get gold answer
            gold_answer = ""
            for item in train_data:
                if item["prompt"] == prompt:
                    gold_answer = item["gold_answer"]
                    gold_doc_ids = item["gold_doc_ids"]
                    break
            
            # Answer F1
            answer_f1 = f1_score(pred_answer, gold_answer)
            
            # Evidence recall
            pred_doc_ids = set(pred_evidence)
            gold_doc_ids_set = set(gold_doc_ids)
            evidence_recall = len(gold_doc_ids_set & pred_doc_ids) / len(gold_doc_ids_set) if gold_doc_ids_set else 0.0
            
            # Format reward
            has_result = '<result>' in completion and '</result>' in completion
            has_evidence = 'evidence:' in completion.lower()
            format_reward = 0.3 * has_result + 0.2 * has_evidence
            
            reward = 0.5 * answer_f1 + 0.3 * evidence_recall + format_reward
            rewards.append(reward)
        
        return rewards

    # GRPO Config
    grpo_config = GRPOConfig(
        output_dir=args.save_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        max_prompt_length=args.max_length,
        max_completion_length=args.max_new_tokens,
        num_generations=args.num_generations,
        logging_steps=1,
        save_steps=10,
        eval_strategy="steps",
        eval_steps=5,
        bf16=True,
        report_to="none",
    )

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Load LoRA if provided
    if args.main_lora:
        print(f"Loading LoRA from {args.main_lora}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.main_lora)

    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Training complete. Model saved to {args.save_dir}")


if __name__ == "__main__":
    main()
