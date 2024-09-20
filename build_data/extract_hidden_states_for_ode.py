import torch
# from transformers import AutoTokenizer, OPTForCausalLM
import pandas as pd
import numpy as np
from typing import Dict, List
import json
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
import json
import os
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoConfig
# from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='embedding_extraction.log')

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def init_model(model_name: str, device: str, num_gpus: int, max_gpu_memory: int):
    """
    Initializes and returns the model and tokenizer.
    """
    if device == "cuda":
        kwargs = {"torch_dtype": torch.bfloat16, "offload_folder": f"{model_name}/offload"}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: f"{max_gpu_memory}GiB" for i in range(num_gpus)},
                })
    elif device == "cpu":
        kwargs = {}
    else:
        raise ValueError(f"Invalid device: {device}")
    try:
        config = AutoConfig.from_pretrained("huggyllama/llama-" + model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-" + model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-" + model_name, trust_remote_code=True,
                                                     low_cpu_mem_usage=True, config=config, **kwargs)
    except Exception as e:
        print(f"An error occurred when initializing the model: {str(e)}")
        return None, None

    if device == "cuda" and num_gpus == 1:
        model.cuda()

    return model, tokenizer


def eval_model(args):
    try:
        with open("build_data/config.json") as config_file:
            config_parameters = json.load(config_file)
    except FileNotFoundError:
        logging.error("Configuration file not found. Please ensure the file exists and the path is correct.")
        return
    except PermissionError:
        logging.error("Permission denied. Please check your file permissions.")
        return
    except json.JSONDecodeError:
        logging.error("Configuration file is not valid JSON. Please check the file's contents.")
        return

    # Model
    model_name = args.model if args.model is not None else config_parameters["model"]
    device = args.device if args.device is not None else config_parameters["device"]
    num_gpus = args.num_gpus if args.num_gpus is not None else config_parameters["num_gpus"]
    max_gpu_memory = args.max_gpu_memory if args.max_gpu_memory is not None else config_parameters["max_gpu_memory"]

    model, tokenizer = init_model(model_name, device, num_gpus, max_gpu_memory)

    if model is None or tokenizer is None:
        logging.error("Model or tokenizer initialization failed.")
        return

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    responses = {}
    # system_prompt = "You are a helpful, respectful and honest assistant with a deep knowledge of natural language processing. Always answer as helpfully as possible. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
    system_prompt = "Always answer as helpfully as possible."
    for line in tqdm(questions):
        idx = line["question_id"]
        question = line['question']
        response = line['response']
        label = line['label']
        combined_hidden_states = {}
        combined_tokens = {}

        prompt = system_prompt + question
        # prompt = "Q: {}".format(question) + "A: {}".format(response)
        tokenizer.pad_token = tokenizer.eos_token
        encoded_input = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']

        try:
            with torch.inference_mode():
                model_outputs = model(
                    input_ids,
                    output_hidden_states=True,
                    output_attentions=True,
                    attention_mask=attention_mask,
                )
            hidden_layer = 32
            hidden_states = model_outputs['hidden_states'][hidden_layer][0]  # model_outputs['hidden_states']的类型是tuple(33), 里面每一个元素是（1, 30, 4096）
            # print(hidden_states)
            # print(hidden_states.shape)
            total_tokens = []
            for t in range(input_ids.shape[1]):
                total_gen_tok_id = input_ids[:, t]
                total_gen_tok = tokenizer.decode(total_gen_tok_id)
                total_tokens.append(total_gen_tok)
            question_tf = "".join(line["question"].split(" "))
            xarr = [i for i in range(len(total_tokens))]
            for i1 in xarr:
                mystring = "".join(total_tokens[i1:])
                if question_tf not in mystring:
                    break
            i1 = i1 - 1
            for i2 in xarr[::-1]:
                mystring = "".join(total_tokens[i1:i2 + 1])
                if question_tf not in mystring:
                    break
            i2 = i2 + 1

            question_start_idx = i1
            question_end_idx = i2
            ques_tokens_len = i2 - i1 + 1
            ques_hidden_states = hidden_states[question_start_idx:question_end_idx + 1, :]
            combined_hidden_states["ques"] = ques_hidden_states.to(torch.float32).detach().cpu().numpy().tolist()
            combined_tokens["ques"] = total_tokens[question_start_idx:question_end_idx + 1]

            if (question_start_idx == 9) and (question_end_idx == len(total_tokens) - 1) and (not torch.isnan(ques_hidden_states).any()) and (len(combined_hidden_states['ques']) < 350):
                output = {"question_id": idx,
                          "question": question,
                          "response": response,
                          "hidden_states": combined_hidden_states,
                          "tokens": combined_tokens,
                          "label": label
                          }
                responses[idx] = output
            else:
                print("idx")
                print(question_start_idx == 9)
                print(question_end_idx == len(total_tokens) - 1)
                print(not torch.isnan(ques_hidden_states).any())
                print((len(combined_hidden_states['ques']) < 350))
        except torch.cuda.OutOfMemoryError:
            print("there is not enough memory")
            print("idx:".format(idx))
            torch.cuda.empty_cache()
            continue



    with open(answers_file, 'wb') as file:
        pickle.dump(responses, file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate new csv with embeddings.")
    parser.add_argument("--model",
                        help="Name of the language model to use: '6.7b', '2.7b', '1.3b', '350m'")
    parser.add_argument("--question-file", type=str, default="build_data/datasets_local/company.json")
    parser.add_argument("--answers-file", type=str, default="build_data/datasets_local/answer_company.bin")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--layers", nargs='*',
                        help="List of layers of the LM to save embeddings from indexed negatively from the end")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=80)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")

    args = parser.parse_args()

    eval_model(args)
