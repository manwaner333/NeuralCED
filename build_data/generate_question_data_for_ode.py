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
import openai
import time
from openai import OpenAI
from resource import openai_key
# Download the necessary resources for sentence tokenization
nltk.download('punkt')

instruction = """I want you act as a hallucination answer generator. Given a question, the right answer, and the link of question source, your objective is to write a hallucinated answer that sounds plausible but is factually incorrect. You SHOULD write the hallucinated answer using the following method (each with some examples):

You are trying to answer a question but you misunderstand the question context and intention.
#Link of Question Source#: https://www.europetnet.org/pet-resources/dog-breeds/item/1465-american-hairless-terrier.html
#Question#: What is a rare breed of dog that was derived as a variant of Rat Terrier, Shiloh Shepherd dog or American Hairless Terrier?
#Right Answer#: American Hairless Terrier
#Hallucinated Answer#: One rare breed of dog that was derived as a variant of Rat Terrier, Shiloh Shepherd dog or American Hairless Terrier is the Teddy Roosevelt Terrier.
"""

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
    elif dataset_name == "neg_invention":
        origin_file_name = 'build_data/datasets_local/neg_inventions_true_false.csv'
        ques_file_name = 'build_data/datasets_local/neg_invention.json'
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

        if idx > 20:
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

# instruction +
def get_qa_res(knowledge, question, answer, instruction):
    if isinstance(instruction, str):
        message = [
            {"role": "user", "content": "You are now a mature hallucination generator. Please generate hallucinated answer for the following question. You can use any method you have learned that is suitable for the given question.  Hallucinated answer as short as possible." +
                                        "\n\n#Link of Question Source#: " + knowledge +
                                        "\n#Question#: " + question +
                                        "\n#Right Answer#: " + answer +
                                        "\n#Hallucinated Answer#: "}
        ]
    elif isinstance(instruction, list):
        mes = [{"role": "user",
                "content": "You are now a mature hallucination generator. Please generate hallucinated answer for the following question. You can use any method you have learned that is suitable for the given question." +
                           "\n\n#Knowledge#: " + knowledge +
                           "\n#Question#: " + question +
                           "\n#Right Answer#: " + answer +
                           "\n#Hallucinated Answer#: "}]
        message = instruction + mes
    else:
        raise TypeError("The instruction must be str or list!")

    while True:
        try:
            client = openai.OpenAI(api_key=openai_key)
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=message,
                temperature=1,
                max_tokens=100,
                top_p=1
            )
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)

    # print(res['choices'][0]['message']['content'])
    return res.choices[0].message.content    # res['choices'][0]['message']['content']


def get_nq_qa_correct_res(knowledge, question, instruction):
    if isinstance(instruction, str):
        message = [
            {"role": "user", "content": "You are now a mature correct generator. Please generate hallucinated answer for the following question. You can use any method you have learned that is suitable for the given question.  Correct answer as short as possible." +
                                        "\n\n#Link of Question Source#: " + knowledge +
                                        "\n#Question#: " + question +
                                        "\n#Right Answer#: "
                                        }]
    elif isinstance(instruction, list):
        mes = [{"role": "user",
                "content": "You are now a mature hallucination generator. Please generate hallucinated answer for the following question. You can use any method you have learned that is suitable for the given question." +
                           "\n\n#Knowledge#: " + knowledge +
                           "\n#Question#: " + question +
                           "\n#Right Answer#: "
                }]
        message = instruction + mes
    else:
        raise TypeError("The instruction must be str or list!")

    while True:
        try:
            client = openai.OpenAI(api_key=openai_key)
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=message,
                temperature=1,
                max_tokens=100,
                top_p=1
            )
            break
        except Exception as e:
            print(e)

    # print(res['choices'][0]['message']['content'])
    return res.choices[0].message.content    # res['choices'][0]['message']['content']


def get_nq_qa_wrong_res(knowledge, question, instruction):
    if isinstance(instruction, str):
        message = [
            {"role": "user", "content": "You are now a mature hallucination generator. Please generate hallucinated answer for the following question. You can use any method you have learned that is suitable for the given question.  Hallucinated answer as short as possible." +
                                        "\n\n#Link of Question Source#: " + knowledge +
                                        "\n#Question#: " + question +
                                        "\n#Hallucinated Answer#: "}
        ]
    elif isinstance(instruction, list):
        mes = [{"role": "user",
                "content": "You are now a mature hallucination generator. Please generate hallucinated answer for the following question. You can use any method you have learned that is suitable for the given question." +
                           "\n\n#Knowledge#: " + knowledge +
                           "\n#Question#: " + question +
                           "\n#Hallucinated Answer#: "}]
        message = instruction + mes
    else:
        raise TypeError("The instruction must be str or list!")

    while True:
        try:
            client = openai.OpenAI(api_key=openai_key)
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=message,
                temperature=1,
                max_tokens=100,
                top_p=1
            )
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)

    # print(res['choices'][0]['message']['content'])
    return res.choices[0].message.content    # res['choices'][0]['message']['content']



def main_NQ_qa():
    # data_dict = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")["validation"]
    # lines = [json.loads(q) for q in open(os.path.expanduser(r"C:\Users\Administrator\Downloads\v1.0-simplified_nq-dev-all.jsonl\v1.0-simplified_nq-dev-all.jsonl"), "r", encoding="utf-8")]
    count = 0
    lines = []
    for q in open(os.path.expanduser(r"C:\Users\Administrator\Downloads\v1.0-simplified_nq-dev-all.jsonl\v1.0-simplified_nq-dev-all.jsonl"), "r", encoding="utf-8"):
        a = json.loads(q)
        lines.append(a)
        count += 1
        if count > 600:
            break
    # data_dict = load_dataset("google-research-datasets/natural_questions", "dev")["validation"]
    # dataset = load_dataset("google-research-datasets/natural_questions", split="dev")

    # Function to extract question and answers from each example
    # def extract_answers(example):
    #     question = example['question']
    #     answers = example['annotations']['short_answers']
    #     # Convert answer indices to strings using the context
    #     extracted_answers = []
    #     for answer in answers:
    #         start_token, end_token = answer['start_token'], answer['end_token']
    #         text_answer = " ".join(example['document_text'].split()[start_token:end_token])
    #         extracted_answers.append(text_answer)
    #     return {
    #         "question": question,
    #         "answers": extracted_answers
    #     }
    #
    # # Extract answers from the dataset
    # extracted_data = dataset.map(extract_answers, remove_columns=dataset.column_names)
    #
    # # Convert to a Pandas DataFrame
    # df = pd.DataFrame(extracted_data)
    #
    # # Keep only the first 600 rows
    # df_first_600 = df.head(600)

    new_file_name = "build_data/datasets_local/nq_qa.json"
    directory = os.path.dirname(new_file_name)
    os.makedirs(directory, exist_ok=True)
    new_file = open(new_file_name, "w")
    idx = 0
    for ele in tqdm(lines):
        question = ele["question_text"]
        question_source = ele['document_url']
        document_title = ele['document_title']
        answer = get_nq_qa_correct_res(question_source, question, instruction)
        wrong_answer = get_nq_qa_wrong_res(question_source, question, instruction)

        label = 0
        new_file.write(json.dumps({
            "question_id": idx,
            "document_title": document_title,
            "question": "Q: {}".format(question) + " A: {}".format(answer + '.'),  # 将question 和 response 放在一起
            "response": answer,
            "label": label
        }) + "\n")
        idx += 1

        label = 1
        new_file.write(json.dumps({
            "question_id": idx,
            "document_title": document_title,
            "question": "Q: {}".format(question) + " A: {}".format(wrong_answer),
            "response": wrong_answer,
            "label": label
        }) + "\n")
        idx += 1

        new_file.flush()
        print(idx)
        if idx > 1000:
            break


def main_trivia_qa():
    data_dict = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")["validation"]
    new_file_name = "build_data/datasets_local/trivia_qa.json"
    directory = os.path.dirname(new_file_name)
    os.makedirs(directory, exist_ok=True)
    new_file = open(new_file_name, "w")
    idx = 0

    for ele in tqdm(data_dict):
        question = ele["question"]
        question_source = ele['question_source']
        answer = ele['answer']['aliases'][0]
        wrong_answer = get_qa_res(question_source, question, answer, instruction)

        label = 0
        new_file.write(json.dumps({
            "question_id": idx,
            "question": "Q: {}".format(question) + " A: {}".format(answer + '.'),  # 将question 和 response 放在一起
            "response": answer,
            "label": label
        }) + "\n")
        idx += 1

        label = 1
        new_file.write(json.dumps({
            "question_id": idx,
            "question": "Q: {}".format(question) + " A: {}".format(wrong_answer),
            "response": wrong_answer,
            "label": label
        }) + "\n")
        idx += 1

        new_file.flush()
        print(idx)
        if idx > 1000:
            break


def main_HaluEval_qa():

    new_file_name = "build_data/datasets_local/HaluEval_qa.json"

    # if os.path.exists(new_file_name):
    #     os.remove(new_file_name)

    # directory = os.path.dirname(new_file_name)
    # os.makedirs(directory)
    #
    new_file = open(new_file_name, "w")
    idx = 0

    lines = [json.loads(q) for q in open(os.path.expanduser('build_data/datasets_local/HaluEval_qa_data_original.json'), "r", encoding="utf-8")]
    for ele in lines:
        question = ele["question"]

        label = 0
        new_file.write(json.dumps({
            "question_id": idx,
            "question": "Q: {}".format(question) + " A: {}".format(ele['right_answer'] + '.'),
            # 将question 和 response 放在一起
            "response": ele['right_answer'],
            "label": label
        }) + "\n")
        idx += 1

        label = 1
        new_file.write(json.dumps({
            "question_id": idx,
            "question": "Q: {}".format(question) + " A: {}".format(ele['hallucinated_answer'] + '.'),
            "response": ele['hallucinated_answer'],
            "label": label
        }) + "\n")
        idx += 1
        new_file.flush()

        if idx > 1000:
            break

    qingli = 3



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
            if len(human_text_update.split()) < 65:
                human_text_update += sentence
            # print(f"Sentence {i}: {sentence}")

        new_file.write(json.dumps({
            "question_id": idx,
            "question": prompt + ". " + human_text_update,
            "response": prompt + ". " + human_text_update,
            "label": label
        }) + "\n")
        idx += 1

        label = 1
        machine_text = line['machine_text']
        sentences = sent_tokenize(machine_text)
        # machine_text_update = " ".join(sentences[:3])
        machine_text_update = ""
        for i, sentence in enumerate(sentences, 1):
            if len(machine_text_update.split()) < 65:
                machine_text_update += sentence
            # print(f"Sentence {i}: {sentence}")
        new_file.write(json.dumps({
            "question_id": idx,
            "question": prompt + ". " + machine_text_update,
            "response": prompt + ". " + machine_text_update,
            "label": label
        }) + "\n")
        idx += 1
        new_file.flush()

        if idx > 40:  # 1000:
            break

    new_file.close()



if __name__ == "__main__":
   # main(dataset_name="ani_cap_ele_fact_inv")
   #main(dataset_name="capital")
   # main(dataset_name="company")
   # main(dataset_name="neg_company")
   # main(dataset_name="neg_fact")
   # main(dataset_name="neg_fact")
   #main(dataset_name="animal")
   #main(dataset_name="city")
   # main(dataset_name="neg_city")
   #main(dataset_name="element")
   #main(dataset_name="invention")
   # main(dataset_name="neg_invention")
   # main(dataset_name="neg_invention_fact")
   # main_truthful_qa()
   # file_name = "build_data/datasets_local/truthful_qa.json"
   # split_json_file(file_name)
   # main_human_machine_text()
   # main_trivia_qa()
   # main_HaluEval_qa()
   main_NQ_qa()



