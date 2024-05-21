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
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoConfig
from datasets import load_dataset
import json
import pandas as pd


def main(dataset_name):
    if dataset_name == "company":
        origin_file_name = 'datasets_local/neg_companies_true_false.csv'
        ques_file_name = 'datasets_local/company.json'
    else:
        origin_file_name = 'aaa'
        ques_file_name = 'bbb'

    df = pd.read_csv(origin_file_name)
    new_file = open(ques_file_name, "w")
    idx = 0
    for index, row in df.iterrows():
        question = row["statement"]
        label = row["label"]
        if label == 0:
            new_label = 1
        else:
            new_label = 0

        new_file.write(json.dumps({
            "question_id": idx,
            "question": question,
            "response": question,
            "label": new_label
        }) + "\n")
        idx += 1

        if idx > 3:
            break

        new_file.flush()

    new_file.close()

if __name__ == "__main__":
   main(dataset_name="company")
