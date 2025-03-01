import torch
torch.set_grad_enabled(False)
from transformer_lens import HookedTransformer, ActivationCache
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens.loading_from_pretrained import get_official_model_name
import argparse
import os
from datasets import load_dataset

import json

def generate_response(prompt,model,tokenizer,max_new_tokens=512):
    messages = [
        {"role": "system", "content": "You are a helpful and harmless assistant. You should think step-by-step."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None    
    )
    generated_ids = [output_ids for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="boolean_expressions")
    parser.add_argument("--model_name", type=str, default="O1-OPEN/OpenO1-LLama-8B-v0.1")
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_dataset("maveriq/bigbenchhard", args.dataset_name)
    generated_dataset = []
    for i in range(len(dataset['train'])):
        sample = dataset['train'][i]
        response = generate_response(sample['input'],model,tokenizer,max_new_tokens=1000)
        generated_dataset.append({
            'input': sample['input'],
            'target': sample['target'],
            'model_output': response
        })

    model_name = args.model_name.split('/')[-1]
    output_path = f'./{model_name}'
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, f'{args.dataset_name}_dataset_generation.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(generated_dataset, f, ensure_ascii=False, indent=2)
    print(f"Save output in {output_path}")

    critical_words = open('critical_words.txt').read().split('\n')
    task_names = ['web_of_lies','word_sorting','boolean']
    for task_name in task_names:
        print(f'{task_name} begining')
        data = json.load(open(f'{task_name}_dataset_generation.json'))
        double_check_data = []
        Non_double_check_data = []
        for item in data:
            if 'model_output' in item and any(critical_word.lower() in item['model_output'].lower() for critical_word in critical_words):
                double_check_data.append(item)
            else:
                Non_double_check_data.append(item)
        import os
        os.makedirs(f'./data/{task_name}', exist_ok=True)
        with open(f'./data/{task_name}/double_check_data.json', 'w') as f:
            json.dump(double_check_data, f, ensure_ascii=False, indent=2)
        with open(f'./data/{task_name}/Non_double_check_data.json', 'w') as f:
            json.dump(Non_double_check_data, f, ensure_ascii=False, indent=2)

