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
import random
import os
import nltk
from nltk.tokenize import sent_tokenize

# Download the necessary resources for sentence tokenization
nltk.download('punkt')


# 生成数据company 那一类的数据
def main(dataset_name):
    # company
    if dataset_name == "company":
        origin_file_name = 'build_data/datasets_local/companies_true_false.csv'
        ques_file_name = 'build_data/datasets_local/company.json'
    elif dataset_name == "neg_company":
        origin_file_name = 'build_data/datasets_local/neg_companies_true_false.csv'
        ques_file_name = 'build_data/datasets_local/neg_company.json'
    elif dataset_name == "conj_neg_company":
        origin_file_name = 'build_data/datasets_local/conj_neg_companies_true_false.csv'
        ques_file_name = 'build_data/datasets_local/conj_neg_company.json'
    # fact
    elif dataset_name == "fact":
        origin_file_name = 'build_data/datasets_local/facts_true_false.csv'
        ques_file_name = 'build_data/datasets_local/fact.json'
    elif dataset_name == "neg_fact":
        origin_file_name = 'build_data/datasets_local/neg_facts_true_false.csv'
        ques_file_name = 'build_data/datasets_local/neg_fact.json'
    elif dataset_name == "conj_neg_fact":
        origin_file_name = 'build_data/datasets_local/conj_neg_facts_true_false.csv'
        ques_file_name = 'build_data/datasets_local/conj_neg_fact.json'
    # animal
    elif dataset_name == "animal":
        origin_file_name = 'build_data/datasets_local/animals_true_false.csv'
        ques_file_name = 'build_data/datasets_local/animal.json'
    # city
    elif dataset_name == "city":
        origin_file_name = 'build_data/datasets_local/cities_true_false.csv'
        ques_file_name = 'build_data/datasets_local/city.json'
    elif dataset_name == "neg_city":
        origin_file_name = 'build_data/datasets_local/neg_cities_true_false.csv'
        ques_file_name = 'build_data/datasets_local/neg_city.json'
    # elements
    elif dataset_name == "element":
        origin_file_name = 'build_data/datasets_local/elements_true_false.csv'
        ques_file_name = 'build_data/datasets_local/element.json'
    # invention
    elif dataset_name == "invention":
        origin_file_name = 'build_data/datasets_local/inventions_true_false.csv'
        ques_file_name = 'build_data/datasets_local/invention.json'
    # invention_fact
    elif dataset_name == "neg_invention_fact":
        origin_file_name = 'build_data/datasets_local/neg_inventions_facts_true_false.csv'
        ques_file_name = 'build_data/datasets_local/neg_invention_fact.json'
    # capital
    elif dataset_name == "capital":
        origin_file_name = 'build_data/datasets_local/capitals_true_false.csv'
        ques_file_name = 'build_data/datasets_local/capital.json'
    # ani_cap_ele_fact_inv
    elif dataset_name == "ani_cap_ele_fact_inv":
        origin_file_name = 'build_data/datasets_local/ani_cap_ele_fact_inv_true_false.csv'
        ques_file_name = 'build_data/datasets_local/ani_cap_ele_fact_inv.json'

    df = pd.read_csv(origin_file_name)
    new_file = open(ques_file_name, "w")
    idx = 0
    for index, row in df.iterrows():
        question = row["statement"]
        label = row["label"]
        if label == 0:    # 原数据集中， 正确的是1， 错误的是0
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

        if idx > 40:
            break

        new_file.flush()

    new_file.close()


def main_truthful_qa():
    data_dict = load_dataset("truthful_qa", "generation")["validation"]
    new_file_name = "build_data/datasets_local/truthful_qa.json"
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
            "question": "Q: {}".format(question) + " A: {}".format(ele['correct_answers'][0] + '.'),  # 将question 和 response 放在一起
            "response": ele['correct_answers'][0] + '.',
            "label": label
        }) + "\n")
        idx += 1

        label = 1
        new_file.write(json.dumps({
            "question_id": idx,
            "question": "Q: {}".format(question) + " A: {}".format(ele['incorrect_answers'][0] + '.'),
            "response": ele['incorrect_answers'][0] + '.',
            "label": label
        }) + "\n")
        idx += 1

        new_file.flush()

    new_file.close()

# Split the generated file into two parts: one for training and another for testing
def split_json_file(filename, ratio=0.8):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            try:
                json_data = json.loads(line)
                data.append(json_data)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from line: {line}, error: {e}")

    random.shuffle(data)

    split_point = int(len(data) * ratio)

    part1 = data[:split_point]
    part2 = data[split_point:]


    with open('build_data/datasets_local/truthful_qa_train.json', 'w') as file1:
        for item in part1:
            json.dump(item, file1)
            file1.write('\n')
    with open('build_data/datasets_local/truthful_qa_test.json', 'w') as file2:
        for item in part2:
            json.dump(item, file2)
            file2.write('\n')

def main_human_machine_text():
    # origin_data_path = "build_data/datasets_local/reddit_chatGPT.jsonl"
    # new_file_name = "build_data/datasets_local/reddit_chatGPT_data.json"
    origin_data_path = "build_data/datasets_local/wikipedia_chatgpt.jsonl"
    new_file_name = "build_data/datasets_local/wikipedia_chatgpt_data.json"
    new_file = open(new_file_name, "w")
    idx = 0
    questions = [json.loads(q) for q in open(os.path.expanduser(origin_data_path), "r")]

    for line in tqdm(questions):
        prompt = line['prompt']
        human_text = line['human_text']
        label = 0
        sentences = sent_tokenize(human_text)
        # human_text_update = " ".join(sentences[:3])
        human_text_update = ""
        for i, sentence in enumerate(sentences, 1):
            if len(human_text_update.split()) < 85:
                human_text_update += sentence
            # print(f"Sentence {i}: {sentence}")

        new_file.write(json.dumps({
            "question_id": idx,
            "question": prompt + human_text_update,
            "response": prompt + human_text_update,
            "label": label
        }) + "\n")
        idx += 1

        label = 1
        machine_text = line['machine_text']
        sentences = sent_tokenize(machine_text)
        # machine_text_update = " ".join(sentences[:3])
        machine_text_update = ""
        for i, sentence in enumerate(sentences, 1):
            if len(machine_text_update.split()) < 85:
                machine_text_update += sentence
            # print(f"Sentence {i}: {sentence}")
        new_file.write(json.dumps({
            "question_id": idx,
            "question": prompt + machine_text_update,
            "response": prompt + machine_text_update,
            "label": label
        }) + "\n")
        idx += 1
        new_file.flush()

        if idx > 1000:  # 1000:
            break

    new_file.close()



if __name__ == "__main__":
   # main(dataset_name="ani_cap_ele_fact_inv")
   #main(dataset_name="capital")
   # main(dataset_name="company")
   # main(dataset_name="neg_company")
   #main(dataset_name="fact")
   # main(dataset_name="neg_fact")
   #main(dataset_name="animal")
   #main(dataset_name="city")
   #main(dataset_name="neg_city")
   #main(dataset_name="element")
   #main(dataset_name="invention")
   # main(dataset_name="neg_invention_fact")
   # main_truthful_qa()
   # file_name = "build_data/datasets_local/truthful_qa.json"
   # split_json_file(file_name)
   main_human_machine_text()



