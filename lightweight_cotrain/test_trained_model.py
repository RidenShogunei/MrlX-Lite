"""
测试训练前后模型效果对比
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from math_environment import MathEnvironment

def load_model(base_path):
    """加载基础模型"""
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        trust_remote_code=True,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_lora_model(base_path, lora_path):
    """加载 LoRA 模型"""
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        trust_remote_code=True,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    """生成响应"""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=0.9,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    prompt_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return response

def build_test_prompt(question):
    """构建测试 prompt"""
    system_msg = (
        "You are a math problem solver. Solve the problem step by step.\n"
        "Use the following format:\n"
        "<thinking>Your reasoning process</thinking>\n"
        "<result>Your final numerical answer</result>"
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Problem: {question}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def extract_number(text):
    """提取答案数字"""
    import re
    patterns = [
        r'(?:答案|answer|result)(?:是|:|=)\s*(-?\d+\.?\d*)',
        r'(?:=|等于)\s*(-?\d+\.?\d*)',
        r'<result>(-?\d+\.?\d*)</result>',
        r'(-?\d+\.?\d*)\s*(?:$|\n)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except:
                continue
    return None

def test_single_question(model, tokenizer, task, model_name):
    """测试单个问题"""
    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print(f"问题: {task.question}")
    print(f"正确答案: {task.answer}")
    print('-'*60)
    
    prompt = build_test_prompt(task.question)
    response = generate_response(model, tokenizer, prompt, max_new_tokens=512)
    pred = extract_number(response)
    
    print(f"生成的响应:\n{response}")
    print(f"\n提取的答案: {pred}")
    is_correct = abs(pred - task.answer) < 1e-3 if pred is not None else False
    print(f"是否正确: {'✅ 正确' if is_correct else '❌ 错误'}")
    return is_correct

if __name__ == "__main__":
    BASE_MODEL = "./models/qwen/Qwen2___5-0___5B-Instruct"
    LORA_PATH_MAIN = "./cotrain_checkpoints_math/lora_main_step_50/main"
    LORA_PATH_SUB = "./cotrain_checkpoints_math/lora_sub_step_50/sub"
    
    print("="*60)
    print("训练前后模型效果对比测试")
    print("="*60)
    
    env = MathEnvironment()
    test_tasks = env.sample_tasks(5)
    
    print(f"\n选择了 5 道测试题:")
    for i, task in enumerate(test_tasks):
        print(f"  {i+1}. {task.question} (答案: {task.answer})")
    
    print(f"\n{'='*60}")
    print("加载原始模型（未训练）...")
    base_model, tokenizer = load_model(BASE_MODEL)
    
    base_correct = 0
    for i, task in enumerate(test_tasks):
        is_correct = test_single_question(base_model, tokenizer, task, "原始模型")
        if is_correct:
            base_correct += 1
    
    del base_model
    torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print("加载训练后的 Main Agent LoRA 模型...")
    main_model, tokenizer = load_lora_model(BASE_MODEL, LORA_PATH_MAIN)
    
    main_correct = 0
    for i, task in enumerate(test_tasks):
        is_correct = test_single_question(main_model, tokenizer, task, "Main Agent (训练后)")
        if is_correct:
            main_correct += 1
    
    del main_model
    torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print("加载训练后的 Sub Agent LoRA 模型...")
    sub_model, tokenizer = load_lora_model(BASE_MODEL, LORA_PATH_SUB)
    
    sub_correct = 0
    for i, task in enumerate(test_tasks):
        is_correct = test_single_question(sub_model, tokenizer, task, "Sub Agent (训练后)")
        if is_correct:
            sub_correct += 1
    
    del sub_model
    torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print("总结")
    print('='*60)
    print(f"原始模型正确率: {base_correct}/{len(test_tasks)} ({base_correct/len(test_tasks)*100:.1f}%)")
    print(f"Main Agent 正确率: {main_correct}/{len(test_tasks)} ({main_correct/len(test_tasks)*100:.1f}%)")
    print(f"Sub Agent 正确率: {sub_correct}/{len(test_tasks)} ({sub_correct/len(test_tasks)*100:.1f}%)")
    print(f"\n{'='*60}")

