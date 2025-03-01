from patch_generation import generate_with_patching, get_source_representation, feed_source_representation
import torch
from transformer_lens import HookedTransformer
from functools import partial
import numpy as np
import nltk
from nltk.corpus import stopwords
import json
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def check_hidden_states(original_text, patch_text, model):
    stop_words = set(stopwords.words('english'))
    stop_words.update({',', '.', ':', '"', "'", ' ', '`', '-', '(', ')', '[', ']', ''})
    tokens1 = set(token.strip().lower() for token in model.to_str_tokens(original_text,prepend_bos=False) 
                 if token.strip().lower() not in stop_words)
    tokens2 = set(token.strip().lower() for token in model.to_str_tokens(patch_text,prepend_bos=False) 
                 if token.strip().lower() not in stop_words)
    
    common_tokens = tokens1.intersection(tokens2)
    
    if len(common_tokens) > 0:
        print(f"Common tokens: {common_tokens}")
    return len(common_tokens) > 0

LLAMA_2_7B_CHAT_PATH = 'meta-llama/Meta-Llama-3-8B-Instruct'
from transformers import LlamaForCausalLM

model = HookedTransformer.from_pretrained(LLAMA_2_7B_CHAT_PATH, device="cuda", fold_ln=False, center_writing_weights=False, center_unembed=False)
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True


def generate_with_patching_all_layers(model, target_prompt,cache):
    def identity_function(source_rep: torch.Tensor) -> torch.Tensor:
        return source_rep
    last_pos_id = len(model.to_tokens(target_prompt)[0]) - 1
    generations = np.ndarray((model.cfg.n_layers, model.cfg.n_layers), dtype=object)
    for source_layer_id in range(model.cfg.n_layers, model.cfg.n_layers):
        source_rep = get_source_representation(
            cache,
            layer_id=source_layer_id,
            pos_id=-1
        )
        for target_layer_id in range(model.cfg.n_layers):
            target_f = partial(
                feed_source_representation,
                source_rep=source_rep,
                pos_id=last_pos_id,
                f=identity_function,
                model=model,
                layer_id=target_layer_id
            )
            gen = generate_with_patching(model, target_prompt, target_f, max_new_tokens=10)
            generations[source_layer_id, target_layer_id] = gen[len(target_prompt):]
    
    return generations

def generate_hidden_thoughts(prompts,model,target_prompt):
    tokens = model.to_tokens(prompts, prepend_bos=True)
    logits, cache = model.run_with_cache(tokens)
    if target_prompt == "description":
        target_prompt = (
            "Syria: Syria is a country in the Middle East, " +
            "Leonardo DiCaprio: Leonardo DiCaprio is an American actor, " +
            "Samsung: Samsung is a South Korean multinational corporation, " +
            "x"
        )
    else:
        target_prompt = (
            "Syria: Syria, " +
            "Leonardo DiCaprio: Leonardo DiCaprio, " +
            "Samsung: Samsung, " +
            "x"
        )
    generations = generate_with_patching_all_layers(model, target_prompt,cache)
    

data = json.load(open('reasoning_chains.json','r'))


def check_with_patching_all_layers(model, target_prompt,cache, original_text):
    def identity_function(source_rep: torch.Tensor) -> torch.Tensor:
        return source_rep
    last_pos_id = len(model.to_tokens(target_prompt)[0]) - 1
    for source_layer_id in range(model.cfg.n_layers):
        source_rep = get_source_representation(
            cache,
            layer_id=source_layer_id,
            pos_id=-1
        )
        for target_layer_id in range(model.cfg.n_layers):
            target_f = partial(
                feed_source_representation,
                source_rep=source_rep,
                pos_id=last_pos_id,
                f=identity_function,
                model=model,
                layer_id=target_layer_id
            )
            gen = generate_with_patching(model, target_prompt, target_f, max_new_tokens=10)
            if check_hidden_states(original_text, gen[len(target_prompt):], model):
                return True
    return False


final_results = []
for item in data[:100]:
    for step in item['reasoning_steps']:
        tokens = model.to_tokens([step['input']], prepend_bos=True)
        logits, cache = model.run_with_cache(tokens)
        target_prompt = (
            "Syria: Syria, " +
            "Leonardo DiCaprio: Leonardo DiCaprio, " +
            "Samsung: Samsung, " +
            "x"
        )
        final_results.append(check_with_patching_all_layers(model,target_prompt, cache,step['output']))
json.dump(final_results, open('patching_results.json', 'w'))

accuracy = sum(final_results) / len(final_results)
print(f"Preknow Accuracy: {accuracy:.2%}")


