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
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoConfig
from datasets import load_dataset


if __name__ == "__main__":
    data_dict = load_dataset("truthful_qa", "generation")["validation"]
    new_file_name = "datasets_local/truthful_qa.json"
    new_file = open(new_file_name, "w")
    idx = 0
    # 统计correct_answers and incorrect_answers 里面的每一个答案
    # for ele in tqdm(data_dict):
    #     question = ele["question"]
    #     # best_answer = ele["best_answer"]
    #     # label = 'ACCURATE'
    #     # new_file.write(json.dumps({
    #     #     "question_id": idx,
    #     #     "question": question,
    #     #     "response": best_answer,
    #     #     "label": label
    #     # }) + "\n")
    #     # idx += 1
    #     for ans in ele['correct_answers']:
    #         # label = 'ACCURATE'
    #         label = 0
    #         new_file.write(json.dumps({
    #             "question_id": idx,
    #             "question": question,
    #             "response": ans + '.',
    #             "label": label
    #         }) + "\n")
    #         idx += 1
    #
    #     for ans in ele['incorrect_answers']:
    #         # label = 'INACCURATE'
    #         label = 1
    #         new_file.write(json.dumps({
    #             "question_id": idx,
    #             "question": question,
    #             "response": ans + '.',
    #             "label": label
    #         }) + "\n")
    #         idx += 1
    #     new_file.flush()

    # 统计correct_answers and incorrect_answers 里面的第一个答案
    for ele in tqdm(data_dict):
        question = ele["question"]

        label = 0
        new_file.write(json.dumps({
            "question_id": idx,
            "question": question,
            "response": ele['correct_answers'][0] + '.',
            "label": label
        }) + "\n")
        idx += 1

        label = 1
        new_file.write(json.dumps({
            "question_id": idx,
            "question": question,
            "response": ele['incorrect_answers'][0] + '.',
            "label": label
        }) + "\n")
        idx += 1

        new_file.flush()

    new_file.close()
