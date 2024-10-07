import pickle
import torch
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
import os
import json
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, precision_recall_curve
import matplotlib.pyplot as plt
import random
import pandas as pd
import csv
import joblib
from sklearn.metrics import roc_auc_score



def extract_info_from_answers(file_path, data_flag, use_tfidf_weight=False, use_attention_weight=False):

    with open(file_path, "rb") as f:
        responses = pickle.load(f)

    human_label_detect_False = {}
    human_label_detect_True = {}
    average_logprob_scores = {}  # average_logprob
    average_entropy_scores = {}  # lowest_logprob
    lowest_logprob_scores = {}  # average_entropy5
    highest_entropy_scores = {}
    sentences_info = {}
    images_info = {}
    sentences_idx_info = {}
    token_and_logprobs_info = {}
    labels_info = {}
    idx_info = {}
    logprob_avg_response_scores = {}
    logprob_min_response_scores = {}
    entropy_avg_response_scores = {}
    entropy_max_response_scores = {}
    label_True_response = {}
    label_False_response = {}
    tfidf_weight_scores = {}
    attention_weight_scores = {}

    if data_flag == "neg_city":
        filter_file = "experiments_cde/datasets/processed_data/neg_city_nostaticintensity_notimeintensity/test_question_ids.pt"
    elif data_flag == "neg_company":
        filter_file = "experiments_cde/datasets/processed_data/neg_company_nostaticintensity_notimeintensity/test_question_ids.pt"
    elif data_flag == "neg_fact":
        filter_file = "experiments_cde/datasets/processed_data/neg_fact_nostaticintensity_notimeintensity/test_question_ids.pt"
    elif data_flag == "neg_invention":
        filter_file = "experiments_cde/datasets/processed_data/neg_invention_nostaticintensity_notimeintensity/test_question_ids.pt"

    keys_val = torch.load(filter_file)

    for idx, response in responses.items():

        if idx not in keys_val:
            continue

        question_id = response["question_id"]
        log_probs = response["logprobs"]
        combined_token_logprobs = log_probs["combined_token_logprobs"]
        combined_token_entropies = log_probs["combined_token_entropies"]
        labels = response["label"]
        sentences_len = 1
        tokens = response['logprobs']['combined_tokens']


        average_logprob_sent_level = [None for _ in range(sentences_len)]  # [None for _ in range(sentences_len)]
        lowest_logprob_sent_level = [None for _ in range(sentences_len)]
        average_entropy_sent_level = [None for _ in range(sentences_len)]
        highest_entropy_sent_level = [None for _ in range(sentences_len)]
        label_True_sent_level = [None for _ in range(sentences_len)]
        label_False_sent_level = [None for _ in range(sentences_len)]
        sentence_sent_level = [None for _ in range(sentences_len)]
        image_sent_level = [None for _ in range(sentences_len)]
        sentence_idx_sent_level = [None for _ in range(sentences_len)]
        token_and_logprob_sent_level = [None for _ in range(sentences_len)]
        label_sent_level = [None for _ in range(sentences_len)]
        idx_sent_level = [None for _ in range(sentences_len)]
        tfidf_weight_sent_level = [None for _ in range(sentences_len)]
        attention_weight_sent_level = [None for _ in range(sentences_len)]

        response_log_probs = []
        response_entropies = []

        sentence_log_probs = combined_token_logprobs['ques']  # combined_token_logprobs[i]
        sentence_entropies = combined_token_entropies['ques']
        label = labels
        average_logprob = np.mean(sentence_log_probs)
        lowest_logprob = np.min(sentence_log_probs)
        average_entropy = np.mean(sentence_entropies)
        highest_entropy = np.max(sentence_entropies)

        average_logprob_sent_level[0] = average_logprob
        lowest_logprob_sent_level[0] = lowest_logprob
        average_entropy_sent_level[0] = average_entropy
        highest_entropy_sent_level[0] = highest_entropy
        label_sent_level[0] = label


        if label == 1:
            true_score = 1.0
            false_score = 0.0
        elif label == 0:
            true_score = 0.0
            false_score = 1.0

        label_True_sent_level[0] = true_score
        label_False_sent_level[0] = false_score

        # response
        response_log_probs.extend(sentence_log_probs)
        response_entropies.extend(sentence_entropies)


        # sentence level
        average_logprob_scores[question_id] = average_logprob_sent_level
        lowest_logprob_scores[question_id] = lowest_logprob_sent_level
        average_entropy_scores[question_id] = average_entropy_sent_level
        highest_entropy_scores[question_id] = highest_entropy_sent_level
        human_label_detect_True[question_id] = label_True_sent_level
        human_label_detect_False[question_id] = label_False_sent_level
        labels_info[question_id] = label_sent_level

    return (average_logprob_scores, lowest_logprob_scores, average_entropy_scores, highest_entropy_scores
            , human_label_detect_True, human_label_detect_False, sentences_info, images_info, sentences_idx_info
            , token_and_logprobs_info, labels_info, idx_info, logprob_avg_response_scores, logprob_min_response_scores
            , entropy_avg_response_scores, entropy_max_response_scores, label_True_response, label_False_response
            , tfidf_weight_scores, attention_weight_scores)



def form_dataframe_from_extract_info(average_logprob_scores, lowest_logprob_scores, average_entropy_scores, highest_entropy_scores, human_label_detect_True, human_label_detect_False, sentences_info, images_info
                                     , sentences_idx_info, token_and_logprobs_info, labels_info, idx_info, tfidf_weight_scores, attention_weight_scores):

    average_logprob_pd = []
    lowest_logprob_pd = []
    average_entropy_pd = []
    highest_entropy_pd = []
    human_label_detect_True_pd = []
    human_label_detect_False_pd = []
    labels_pd = []


    for dic_idx in list(average_logprob_scores.keys()):

        average_logprob_pd.extend(average_logprob_scores[dic_idx])
        lowest_logprob_pd.extend(lowest_logprob_scores[dic_idx])
        average_entropy_pd.extend(average_entropy_scores[dic_idx])
        highest_entropy_pd.extend(highest_entropy_scores[dic_idx])
        human_label_detect_True_pd.extend(human_label_detect_True[dic_idx])
        human_label_detect_False_pd.extend(human_label_detect_False[dic_idx])
        labels_pd.extend(labels_info[dic_idx])

    data = {
        'average_logprob': average_logprob_pd,
        'lowest_logprob': lowest_logprob_pd,
        'average_entropy': average_entropy_pd,
        'highest_entropy': highest_entropy_pd,
        'human_label_detect_True': human_label_detect_True_pd,
        'human_label_detect_False': human_label_detect_False_pd,
        'labels': labels_pd,
    }
    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by='average_logprob', ascending=False)  # , ascending=False
    return df_sorted




if __name__ == "__main__":
    uncertainty = True
    self_check = False
    data_flag = "neg_city"
    if uncertainty:
        # 计算相关的指标， 并保存数据
        model_version = 'llama_7b'  # 'llava_v16_7b'    # 'llava_v16_mistral_7b'
        print(model_version)

        if data_flag == 'neg_city':
            path = f"uncertainty/result/{data_flag}/{model_version}_answer_neg_city_uncertainty_infor.bin"
        elif data_flag == 'neg_company':
            path = f"uncertainty/result/{data_flag}/{model_version}_answer_neg_company_uncertainty_infor.bin"
        elif data_flag == 'neg_fact':
            path = f"uncertainty/result/{data_flag}/{model_version}_answer_neg_fact_uncertainty_infor.bin"
        elif data_flag == 'neg_invention':
            path = f"uncertainty/result/{data_flag}/{model_version}_answer_neg_invention_uncertainty_infor.bin"

        (average_logprob_scores, lowest_logprob_scores, average_entropy_scores, highest_entropy_scores
         , human_label_detect_True, human_label_detect_False, sentences_info, images_info
         , sentences_idx_info, token_and_logprobs_info, labels_info, idx_info
         , logprob_avg_response_scores, logprob_min_response_scores, entropy_avg_response_scores, entropy_max_response_scores
         , label_True_response, label_False_response, tfidf_weight_scores, attention_weight_scores) = extract_info_from_answers(path, data_flag)

        # 将数据存储在dataframe中
        df = form_dataframe_from_extract_info(average_logprob_scores, lowest_logprob_scores, average_entropy_scores, highest_entropy_scores
                                              , human_label_detect_True, human_label_detect_False, sentences_info, images_info
                                              , sentences_idx_info, token_and_logprobs_info, labels_info, idx_info
                                              , tfidf_weight_scores, attention_weight_scores)

        average_logprob_scores = df['average_logprob'].tolist()
        lowest_logprob_scores = df['lowest_logprob'].tolist()
        average_entropy_scores = df['average_entropy'].tolist()
        highest_entropy_scores = df['highest_entropy'].tolist()
        human_label_detect_True = df['human_label_detect_True'].tolist()
        human_label_detect_False = df['human_label_detect_False'].tolist()

        # 分析准确率
        total_num = len(human_label_detect_True)
        print("The total number of sentences is: {}; The ratio of true values is: {}".format(total_num, human_label_detect_True.count(1.0) / total_num))
        print("The total number of sentences is: {}; The ratio of false values is: {}".format(total_num, human_label_detect_True.count(0.0) / total_num))


        print("Detect hallucination")
        # average_logprob
        average_logprob_scores_1 = [-ele for ele in average_logprob_scores]
        average_logprob_accuracy = 0  #  accuracy_score(human_label_detect_True, average_logprob_scores_1)
        average_logprob_precision = 0  # precision_score(human_label_detect_True, average_logprob_scores_1, pos_label=1)
        average_logprob_recall = 0  # recall_score(human_label_detect_True, average_logprob_scores_1, pos_label=1)
        average_logprob_f1 = 0  # f1_score(human_label_detect_True, average_logprob_scores_1, pos_label=1)
        avg_logprob_roc_auc = roc_auc_score(human_label_detect_True, average_logprob_scores_1)
        print(f"average_logprob_Accuracy: {average_logprob_accuracy:.2f} average_logprob_Precision: {average_logprob_precision:.2f} average_logprob_Recall: {average_logprob_recall:.2f} average_logprob_F1: {average_logprob_f1:.2f} average_logprob_AUC_ROC:{avg_logprob_roc_auc:.2f}")

        # average_entropy
        avg_entropy_accuracy = 0  #  accuracy_score(human_label_detect_True, average_entropy_scores)
        avg_entropy_precision = 0  #  precision_score(human_label_detect_True, average_entropy_scores, pos_label=1)
        avg_entropy_recall = 0  #  recall_score(human_label_detect_True, average_entropy_scores, pos_label=1)
        avg_entropy_f1 = 0  #  f1_score(human_label_detect_True, average_entropy_scores, pos_label=1)
        avg_entropy_roc_auc = roc_auc_score(human_label_detect_True, average_entropy_scores)
        print(f"average_entropy_Accuracy: {avg_entropy_accuracy:.2f} average_entropy_Precision: {avg_entropy_precision:.2f} average_entropy_Recall: {avg_entropy_recall:.2f} average_entropy_F1: {avg_entropy_f1:.2f} average_entropy_AUC_ROC:{avg_entropy_roc_auc:.2f}")


        # lowest_logprob
        lowest_logprob_scores_1 = [-ele for ele in lowest_logprob_scores]
        lowest_logprob_accuracy = 0  #  accuracy_score(human_label_detect_True, lowest_logprob_scores_1)
        lowest_logprob_precision = 0  #  precision_score(human_label_detect_True, lowest_logprob_scores_1, pos_label=1)
        lowest_logprob_recall = 0  #  recall_score(human_label_detect_True, lowest_logprob_scores_1, pos_label=1)
        lowest_logprob_f1 = 0  #  f1_score(human_label_detect_True, lowest_logprob_scores_1, pos_label=1)
        lowest_logprob_roc_auc = roc_auc_score(human_label_detect_True, lowest_logprob_scores_1)
        print(f"lowest_logprob_Accuracy: {lowest_logprob_accuracy:.2f} lowest_logprob_Precision: {lowest_logprob_precision:.2f} lowest_logprob_Recall: {lowest_logprob_recall:.2f} lowest_logprob_F1: {lowest_logprob_f1:.2f} lowest_logprob_AUC_ROC:{lowest_logprob_roc_auc:.2f}")


        # highest_entropy
        highest_entropy_accuracy = 0  #  accuracy_score(human_label_detect_True, highest_entropy_scores)
        highest_entropy_precision = 0  #  precision_score(human_label_detect_True, highest_entropy_scores, pos_label=1)
        highest_entropy_recall = 0  #  recall_score(human_label_detect_True, highest_entropy_scores, pos_label=1)
        highest_entropy_f1 = 0  #  f1_score(human_label_detect_True, highest_entropy_scores, pos_label=1)
        highest_entropy_roc_auc = roc_auc_score(human_label_detect_True, highest_entropy_scores)
        print(f"highest_entropy_Accuracy: {highest_entropy_accuracy:.2f} highest_entropy_Precision: {highest_entropy_precision:.2f} highest_entropy_Recall: {highest_entropy_recall:.2f} highest_entropy_F1: {highest_entropy_f1:.2f} highest_entropy_AUC_ROC:{highest_entropy_roc_auc:.2f}")


