import torch
import pandas as pd
import numpy as np
from typing import Dict, List
import json
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoConfig
import spacy
from datasets import load_dataset

def generate_wiki_data():
    with open("datasets/dataset_v3.json", "r") as f:
        content = f.read()
        dataset = json.loads(content)

    # Load the "wiki_bio" dataset
    wiki_bio = load_dataset("wiki_bio")
    split_name = "test"  # Replace with the desired split name
    wiki_bio_test = wiki_bio[split_name]

    res_list = []
    for article in dataset[0:50]:
        wiki_bio_test_idx = article['wiki_bio_test_idx']
        wiki_bio_test_table = wiki_bio_test[wiki_bio_test_idx]['input_text']['table']
        if "name" in wiki_bio_test_table['column_header']:
            name = wiki_bio_test_table['content'][wiki_bio_test_table['column_header'].index('name')]
            prompt = "This is a Wikipedia passage about" + " " + name + '.'
            res_list.append([article['wiki_bio_text'], article['wiki_bio_test_idx'], name, prompt])

    df = pd.DataFrame(data=res_list, columns=['wiki_bio_text', 'wiki_bio_test_idx', 'name', 'prompt'])
    df.to_csv('datasets/wiki_bio_info.csv', index=False)


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
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-" + model_name, trust_remote_code=True,
            low_cpu_mem_usage=True, config=config, **kwargs)
    except Exception as e:
        print(f"An error occurred when initializing the model: {str(e)}")
        return None, None

    if device == "cuda" and num_gpus == 1:
        model.cuda()

    return model, tokenizer


def split_data(list_ori, p):
    list_new = []
    list_short = []
    for i in list_ori:
        list_short.append(i)
        if i == p:
            list_new.append(list_short)
            list_short = []
    list_new.append(list_short)
    return list_new


def split_str(str, p):
    return str.split(p)


# Still not convinced this function works 100% correctly, but it's much faster than process_row.
def process_batch(batch_prompts: List[str], model, tokenizer, remove_period: bool, device):
    """
    Processes a batch of data and returns the embeddings for each statement.
    """
    if remove_period:
        batch_prompts = [prompt.rstrip(". ") for prompt in batch_prompts]
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=200, do_sample=True, temperature=0.5, top_p=0.95, top_k=5)
        gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    token_idx_seqs = [split_data(ele.detach().cpu().numpy().tolist(), 29889) for ele in outputs]
    # text_seqs = [[item + '.' for item in split_str(ele, '.')] for ele in gen_text]
    nlp = spacy.load("en_core_web_sm")
    text_seqs = []
    for ele in gen_text:
        text_seqs.append([item.text.strip() for item in nlp(ele).sents])

    return token_idx_seqs, text_seqs


def load_data(dataset_path: Path, dataset_name: str):
    filename_suffix = ""
    dataset_file = dataset_path / f"{dataset_name}{filename_suffix}.csv"
    try:
        df = pd.read_csv(dataset_file)
    except FileNotFoundError as e:
        print(f"Dataset file {dataset_file} not found: {str(e)}")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing the CSV file {dataset_file}: {str(e)}")
        return None
    except pd.errors.EmptyDataError as e:
        print(f"No data in CSV file {dataset_file}: {str(e)}")
        return None
    return df

def save_data(df, output_path: Path, dataset_name: str, model_name: str, remove_period: bool):
    """
    Saves the processed data to a CSV file.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    filename_suffix = "_rmv_period" if remove_period else ""
    output_file = output_path / f"{dataset_name}_{model_name}{filename_suffix}_token_text_seqs.csv"
    try:
        df.to_csv(output_file, index=False)
    except PermissionError:
        print(f"Permission denied when trying to write to {output_file}. Please check your file permissions.")
    except Exception as e:
        print(f"An unexpected error occurred when trying to write to {output_file}: {e}")

def generate_token_text_seqs():

    model, tokenizer = init_model(model_name, device, num_gpus, max_gpu_memory)
    if model is None or tokenizer is None:
        logging.error("Model or tokenizer initialization failed.")
        return

    for dataset_name in tqdm(dataset_names, desc="Processing datasets"):

        dataset = load_data(dataset_path, dataset_name)

        output = dataset.copy()
        output['token_idx_seqs'] = pd.Series(dtype='object')
        output['text_seqs'] = pd.Series(dtype='object')

        num_batches = len(dataset) // BATCH_SIZE + (len(dataset) % BATCH_SIZE != 0)

        for batch_num in tqdm(range(num_batches), desc=f"Processing batches in {dataset_name}"):
            start_idx = batch_num * BATCH_SIZE
            actual_batch_size = min(BATCH_SIZE, len(dataset) - start_idx)
            end_idx = start_idx + actual_batch_size
            batch = dataset.iloc[start_idx:end_idx]
            batch_prompts = batch['prompt'].tolist()
            token_idx_seqs, text_seqs = process_batch(batch_prompts, model, tokenizer, should_remove_period, device)

            for i, idx in enumerate(range(start_idx, end_idx)):
                output.at[idx, 'token_idx_seqs'] = token_idx_seqs[i]
                output.at[idx, 'text_seqs'] = text_seqs[i]

            if batch_num % 10 == 0:
                logging.info(f"Processing batch {batch_num}")

        save_data(output, output_path, dataset_name, model_name, should_remove_period)


def generate_statement():
    for dataset_name in tqdm(dataset_names, desc="Processing datasets"):
        filename_suffix = ""
        dataset_file = output_path / f"{dataset_name}_{model_name}{filename_suffix}_token_text_seqs.csv"
        df = pd.read_csv(dataset_file)
        res = []
        for index, row in df.iterrows():
            token_idx_seqs = eval(row['token_idx_seqs'])
            text_seqs = eval(row['text_seqs'])
            token_idx_len = len(token_idx_seqs)
            text_len = len(text_seqs)
            if token_idx_len == text_len:
                for i in range(1, token_idx_len-1):
                    res.append([row['wiki_bio_text'], row['wiki_bio_test_idx'], row['name'], row['prompt'], row['text_seqs'], token_idx_seqs[i],
                                text_seqs[i]])

            df = pd.DataFrame(data=res, columns=['wiki_bio_text', 'wiki_bio_test_idx', 'name', 'prompt', "text_seqs", 'token_idx_seqs', 'statement'])
            output_file = output_path / f"{dataset_name}{filename_suffix}_sentences_level.csv"
            df.to_csv(output_file, index=False)

if __name__ == "__main__":
    try:
        with open("config.json") as config_file:
            config_parameters = json.load(config_file)
    except FileNotFoundError:
        logging.error("Configuration file not found. Please ensure the file exists and the path is correct.")
    except PermissionError:
        logging.error("Permission denied. Please check your file permissions.")
    except json.JSONDecodeError:
        logging.error("Configuration file is not valid JSON. Please check the file's contents.")

    parser = argparse.ArgumentParser(description="Generate new csv with embeddings.")
    parser.add_argument("--model",
                        help="Name of the language model to use: '6.7b', '2.7b', '1.3b', '350m'")
    parser.add_argument("--layers", nargs='*',
                        help="List of layers of the LM to save embeddings from indexed negatively from the end")
    parser.add_argument("--dataset_names", nargs='*',
                        help="List of dataset names without csv extension. Can leave off 'true_false' suffix if true_false flag is set to True")
    parser.add_argument("--true_false", type=bool, help="Do you want to append 'true_false' to the dataset name?")
    parser.add_argument("--batch_size", type=int, help="Batch size for processing.")
    parser.add_argument("--remove_period", type=bool, help="True if you want to extract embedding for last token before the final period.")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=80)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")

    args = parser.parse_args()

    model_name = args.model if args.model is not None else config_parameters["model"]
    should_remove_period = args.remove_period if args.remove_period is not None else config_parameters["remove_period"]
    dataset_names = args.dataset_names if args.dataset_names is not None else config_parameters["list_of_datasets"]
    true_false = args.true_false if args.true_false is not None else config_parameters["true_false"]
    BATCH_SIZE = args.batch_size if args.batch_size is not None else config_parameters["batch_size"]
    device = args.device if args.device is not None else config_parameters["device"]
    num_gpus = args.num_gpus if args.num_gpus is not None else config_parameters["num_gpus"]
    max_gpu_memory = args.max_gpu_memory if args.max_gpu_memory is not None else config_parameters["max_gpu_memory"]
    dataset_path = Path(config_parameters["dataset_path"])
    output_path = Path(config_parameters["processed_dataset_path"])

    # 生成原始的wiki数据
    # generate_wiki_data()

    # 生成token and text seqs
    # generate_token_text_seqs()

    # 生成最终需要的statement
    # generate_statement()






