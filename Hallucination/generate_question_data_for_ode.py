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
# from negate import Negator
import json
import pandas as pd


def main(dataset_name):
    # company
    if dataset_name == "company":
        origin_file_name = 'datasets_local/companies_true_false.csv'
        ques_file_name = 'datasets_local/company.json'
    elif dataset_name == "neg_company":
        origin_file_name = 'datasets_local/neg_companies_true_false.csv'
        ques_file_name = 'datasets_local/neg_company.json'
    elif dataset_name == "conj_neg_company":
        origin_file_name = 'datasets_local/conj_neg_companies_true_false.csv'
        ques_file_name = 'datasets_local/conj_neg_company.json'
    # fact
    elif dataset_name == "fact":
        origin_file_name = 'datasets_local/facts_true_false.csv'
        ques_file_name = 'datasets_local/fact.json'
    elif dataset_name == "neg_fact":
        origin_file_name = 'datasets_local/neg_facts_true_false.csv'
        ques_file_name = 'datasets_local/neg_fact.json'
    elif dataset_name == "conj_neg_fact":
        origin_file_name = 'datasets_local/conj_neg_facts_true_false.csv'
        ques_file_name = 'datasets_local/conj_neg_fact.json'
    # animal
    elif dataset_name == "animal":
        origin_file_name = 'datasets_local/animals_true_false.csv'
        ques_file_name = 'datasets_local/animal.json'
    # city
    elif dataset_name == "city":
        origin_file_name = 'datasets_local/cities_true_false.csv'
        ques_file_name = 'datasets_local/city.json'
    elif dataset_name == "neg_city":
        origin_file_name = 'datasets_local/neg_cities_true_false.csv'
        ques_file_name = 'datasets_local/neg_city.json'
    # elements
    elif dataset_name == "element":
        origin_file_name = 'datasets_local/elements_true_false.csv'
        ques_file_name = 'datasets_local/element.json'
    # invention
    elif dataset_name == "invention":
        origin_file_name = 'datasets_local/inventions_true_false.csv'
        ques_file_name = 'datasets_local/invention.json'
    # invention_fact
    elif dataset_name == "neg_invention_fact":
        origin_file_name = 'datasets_local/neg_inventions_facts_true_false.csv'
        ques_file_name = 'datasets_local/neg_invention_fact.json'
    # capital
    elif dataset_name == "capital":
        origin_file_name = 'datasets_local/capitals_true_false.csv'
        ques_file_name = 'datasets_local/capital.json'


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

        #if idx > 20:
            #break

        new_file.flush()

    new_file.close()

if __name__ == "__main__":
    main(dataset_name="capital")
   #main(dataset_name="company")
   #main(dataset_name="neg_company")
   #main(dataset_name="fact")
   #main(dataset_name="neg_fact")
   #main(dataset_name="animal")
   #main(dataset_name="city")
   #main(dataset_name="neg_city")
   #main(dataset_name="element")
   #main(dataset_name="invention")
   # main(dataset_name="neg_invention_fact")



