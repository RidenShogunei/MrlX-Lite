# HotpotQA Dynamic MAS Experiments

This workspace is focused on reproducing and extending M-GRPO-style
multi-agent training on a local HotpotQA environment.

The current research line is:

```text
HotpotQA local search/read environment
-> fixed Main/Sub MAS baseline
-> dynamic Main routing with 0..N Sub agents
-> Main-only synthesis SFT
-> Main-only dynamic GRPO
-> joint GRPO, only after the staged system is stable
```

## Current Checkpoints

Local checkpoint directories are intentionally ignored by git.

```text
Current fixed MAS baseline:
  main = hotpotqa_mas_enhanced_mainonly_conservative_50x1/best/main
  sub  = hotpotqa_mas_enhanced_mainonly_conservative_50x1/best/sub

Current dynamic prototype:
  main = hotpotqa_dynamic_synthesis_mainonly_500x1/main_agent
  sub  = hotpotqa_dynamic_mixture_sft_300x1_v3/sub_agent
```

## Current Data

Tracked data used by the current line:

```text
hotpotqa_data_enhanced/train.jsonl
hotpotqa_data_enhanced/val.jsonl
hotpotqa_dynamic_mixture_sft_data_300_v3.jsonl
hotpotqa_dynamic_synthesis_sft_data_500.jsonl
```

Historical HotpotQA SFT data is still tracked because it is useful for baseline
reproduction:

```text
hotpotqa_sft_data.jsonl
hotpotqa_mas_sft_data.jsonl
hotpotqa_mas_sft_data_v2.jsonl
hotpotqa_dynamic_mas_sft_data.jsonl
```

## Main Scripts

Environment and data:

```text
hotpotqa_environment.py
prepare_hotpotqa_data.py
prepare_hotpotqa_enhanced_data.py
generate_hotpotqa_mas_sft_data.py
generate_hotpotqa_dynamic_mas_sft_data.py
generate_hotpotqa_dynamic_mixture_sft_data.py
generate_hotpotqa_dynamic_synthesis_sft_data.py
```

Training:

```text
sft_trainer.py
train_hotpotqa_sub_preferences.py
grpo_hotpotqa.py
grpo_hotpotqa_mas.py
```

Evaluation:

```text
analyze_hotpotqa_mas_results.py
analyze_hotpotqa_dynamic_mas_results.py
analyze_hotpotqa_sub_oracle.py
run_hotpotqa_eval_suite.py
run_hotpotqa_direct_eval_suite.py
run_hotpotqa_dynamic_eval_suite.py
```

## Reproduce Current Dynamic Synthesis SFT

Generate Main-only synthesis data:

```bash
python generate_hotpotqa_dynamic_synthesis_sft_data.py ^
  --train-jsonl .\hotpotqa_data_enhanced\train.jsonl ^
  --output .\hotpotqa_dynamic_synthesis_sft_data_500.jsonl ^
  --limit 500 ^
  --max-subtasks 2 ^
  --max-snippet-chars 360
```

Train Main only:

```bash
python sft_trainer.py ^
  --data-path .\hotpotqa_dynamic_synthesis_sft_data_500.jsonl ^
  --save-dir .\hotpotqa_dynamic_synthesis_mainonly_500x1 ^
  --epochs 1 ^
  --lr 3e-5 ^
  --max-length 1536 ^
  --main-lora .\hotpotqa_dynamic_mixture_sft_300x1_v3\main_agent ^
  --no-train-sub
```

Evaluate dynamic MAS:

```bash
python analyze_hotpotqa_dynamic_mas_results.py ^
  --val-jsonl .\hotpotqa_data_enhanced\val.jsonl ^
  --tasks 20 ^
  --samples 2 ^
  --main-lora .\hotpotqa_dynamic_synthesis_mainonly_500x1\main_agent ^
  --sub-lora .\hotpotqa_dynamic_mixture_sft_300x1_v3\sub_agent ^
  --max-subagents 2
```

Run multi-offset dynamic evaluation:

```bash
python run_hotpotqa_dynamic_eval_suite.py ^
  --val-jsonl .\hotpotqa_data_enhanced\val.jsonl ^
  --out-dir .\hotpotqa_dynamic_eval_suite_synthesis_offsets ^
  --offsets 0 20 40 ^
  --tasks 10 ^
  --samples 2 ^
  --max-subagents 2
```

## Reports

Current reports live in `docs/`:

```text
docs/ENHANCED_HOTPOTQA_EVAL_REPORT.md
docs/HOTPOTQA_MGRPO_REPORT.md
docs/STAGE_HOTPOTQA_MAS_REPORT.md
```

Most recent conclusion:

```text
Dynamic routing can now select useful evidence, but final answer synthesis is
still the main bottleneck. Main-only synthesis SFT improved dynamic reward on
the offset-0 validation slice, but larger multi-offset validation is still
needed before starting Main-only dynamic GRPO.
```
