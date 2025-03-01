import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from scipy.stats import entropy, ttest_ind
import matplotlib.pyplot as plt
import argparse
def generate_with_template(prompt,tokenizer):
    messages = [
        {"role": "system", "content": "You are a specialist in reasoning analysis, do it well!"},
        {"role": "user", "content": prompt},
    ]
    chat_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return chat_template

def find_double_check_triggers(text):
    before_text = ''
    triggers = ['double-check', 'But wait', 'in case of']
    sentences = text.split('.')
    find_flag = False
    for i, sentence in enumerate(sentences):
        for trigger in triggers:
            if trigger.lower() in sentence.lower():
                find_flag = True
                break
        if find_flag:
            break
        else:
            before_text += sentences[i]+'.'
    return before_text

def find_thought_text(text):
    end_idx = text.find('</Thought>')
    if end_idx == -1:
        return text
    return text[:end_idx].strip()

def compute_entropy(model, tokenizer, text):
    """计算文本的熵值
    
    Args:
        model: 预训练模型
        tokenizer: 分词器
        text: 输入文本
    Returns:
        float: 平均熵值
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取最后一个token的logits
    # logits = outputs.logits[0]
    logits = outputs.logits[0, :-1] 
    # 计算softmax概率
    probs = torch.softmax(logits, dim=-1)
    # 将bfloat16转换为float32，然后再转换为numpy数组
    entropies = [entropy(p.cpu().float().numpy()) for p in probs]
    return np.mean(entropies)

def main():
    # 加载模型
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='navigate')
    args = parser.parse_args()
    model_path = 'O1-OPEN/OpenO1-LLama-8B-v0.1'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="cuda", 
        torch_dtype=torch.bfloat16
    )

    # Load Data
    double_check_data = json.load(open(f'./data/{args.data_path}/double_check_data.json','r'))
    Non_double_check_data = json.load(open(f'./data/{args.data_path}/Non_double_check_data.json','r'))

    dc_entropies = []
    non_dc_entropies = []
    before_dc_entropies = []
    after_dc_entropies = []
    print("Computing the double-check entropy...")
    for item in double_check_data:
        before_text = find_double_check_triggers(item['model_output'])
        formatted_prompt = generate_with_template(item['input'],tokenizer)
        input_text = formatted_prompt + before_text
        entropy_value = compute_entropy(model, tokenizer, input_text)
        before_dc_entropies.append(entropy_value)

        before_text = find_thought_text(item['model_output'])
        input_text = formatted_prompt + before_text
        entropy_value = compute_entropy(model, tokenizer, input_text)
        after_dc_entropies.append(entropy_value)
    plt.figure(figsize=(10, 6))
    plt.boxplot([before_dc_entropies, after_dc_entropies], 
                labels=['Before Double-check', 'After Double-check'])
    plt.title('Entropy Distribution in before Double-check vs after Double-check Samples')
    plt.ylabel('Entropy')
    plt.savefig(f'./{args.data_path}_entropy_distribution.png')
    plt.close()

if __name__ == "__main__":
    main()

