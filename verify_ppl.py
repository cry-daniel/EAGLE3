import json
import argparse
import torch
from transformers import AutoTokenizer, AutoConfig
from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from eagle.model.modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from eagle.model.modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from os import path as osp
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="gsm8k")
    parser.add_argument("--input-file", type=str, 
                        default="eagle/data/mt_bench/reference_answer/gpt-4.jsonl")
    parser.add_argument("--judge-model", type=str, 
                        default="/home/ruiyang.chen/hfd/Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    if args.bench_name == "mt_bench":
        raise NotImplementedError

    base_model_path = args.judge_model

    Type = AutoConfig.from_pretrained(base_model_path).architectures[0]
    if Type == 'LlamaForCausalLM':
        model = KVLlamaForCausalLM.from_pretrained(
            base_model_path
        )
    elif Type == 'Qwen2ForCausalLM':
        base_model = KVQwen2ForCausalLM.from_pretrained(
            base_model_path
        )
    else:
        base_model = KVMixtralForCausalLM.from_pretrained(
            base_model_path
        )

    max_length = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = []
    root_dir = osp.dirname(__file__)

    file_path = osp.join(root_dir, "eagle", "data", args.bench_name, "question.jsonl")

    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"open file in {file_path}")
        for line in f:
            obj = json.loads(line)
            full_text = obj.get("turns")[0] + "\n" + obj.get("reference")[0]
            samples.append(full_text)

    total_loss = 0.0
    total_tokens = 0

    for text in tqdm(samples, desc="Evaluating PPL"):
        # print(text)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024, padding=True).input_ids
        input_ids = torch.as_tensor(inputs).cuda()

        local_loss = 0.0
        local_tokens = 0

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            num_tokens = input_ids.numel()

        local_loss = loss.item() * num_tokens
        local_tokens = num_tokens
        avg_nll = local_loss / local_tokens
        ppl = torch.exp(torch.tensor(avg_nll))
        print(text)
        print(f"Local Perplexity: {ppl.item():.2f}")

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    avg_nll = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_nll))
    print(f"\nAverage Perplexity: {ppl.item():.2f}")