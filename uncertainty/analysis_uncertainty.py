import pickle
import torch
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
import os
import json
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import random
import pandas as pd
import csv
import joblib
from sklearn.metrics import roc_auc_score


def unroll_pred(scores, indices):
    unrolled = []
    for idx in indices:
        unrolled.extend(scores[idx])
    return unrolled


# def get_PR_with_human_labels(preds, human_labels, pos_label=1, oneminus_pred=False):
#     indices = [k for k in human_labels.keys()]
#     unroll_preds = unroll_pred(preds, indices)
#     if oneminus_pred:
#         unroll_preds = [1.0-x for x in unroll_preds]
#     unroll_labels = unroll_pred(human_labels, indices)
#     assert len(unroll_preds) == len(unroll_labels)
#     print("len:", len(unroll_preds))
#     P, R, thre = precision_recall_curve(unroll_labels, unroll_preds, pos_label=pos_label)
#     # auroc = roc_auc_score(unroll_labels, unroll_preds)
#     return P, R

def get_PR_with_human_labels(preds, human_labels, pos_label=1, oneminus_pred=False):
    unroll_preds = preds
    if oneminus_pred:
        unroll_preds = [1.0 - x for x in unroll_preds]
    unroll_labels = human_labels
    assert len(unroll_preds) == len(unroll_labels)
    print("len:", len(unroll_preds))
    P, R, thre = precision_recall_curve(unroll_labels, unroll_preds, pos_label=pos_label)
    # auroc = roc_auc_score(unroll_labels, unroll_preds)
    return P, R


def print_AUC(P, R):
    print("AUC: {:.2f}".format(auc(R, P) * 100))


def detect_hidden_states(combined_hidden_states):
    for k, v in combined_hidden_states.items():
        if len(v) == 0:
            return False
    return True


def tfidf_encode(vectorizer, sent):
    tfidf_matrix = vectorizer.transform([sent])

    # Convert the TF-IDF matrix for the sentence to a dense format
    dense_tfidf = tfidf_matrix.todense()

    # Get the feature names (vocabulary) from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Tokenize the sentence
    tokenized_sentence = sent.split()

    token_weights = []

    # For each token in the input sentence, get its weight from the TF-IDF model
    for token in tokenized_sentence:
        # Check if the token is in the TF-IDF model's vocabulary
        if token in feature_names:
            # Find the index of the token in the feature names
            token_index = list(feature_names).index(token)
            # Append the weight of the token to the list
            token_weights.append(dense_tfidf[0, token_index])
        else:
            # If the token is not found in the model's vocabulary, assign a weight of 0
            token_weights.append(0)

    return token_weights


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
    sentences_pd = []
    images_pd = []
    sentences_idx_pd = []
    token_and_logprobs_pd = []
    labels_pd = []
    idxs_pd = []
    label_True_response_pd = []
    label_False_response_pd = []
    tfidf_weight_pd = []
    attention_weight_pd = []

    for dic_idx in list(average_logprob_scores.keys()):

        average_logprob_pd.extend(average_logprob_scores[dic_idx])
        lowest_logprob_pd.extend(lowest_logprob_scores[dic_idx])
        average_entropy_pd.extend(average_entropy_scores[dic_idx])
        highest_entropy_pd.extend(highest_entropy_scores[dic_idx])
        human_label_detect_True_pd.extend(human_label_detect_True[dic_idx])
        human_label_detect_False_pd.extend(human_label_detect_False[dic_idx])
        # sentences_pd.extend(sentences_info[dic_idx])
        # images_pd.extend(images_info[dic_idx])
        # sentences_idx_pd.extend(sentences_idx_info[dic_idx])
        # token_and_logprobs_pd.extend(token_and_logprobs_info[dic_idx])
        labels_pd.extend(labels_info[dic_idx])
        # idxs_pd.extend(idx_info[dic_idx])
        # tfidf_weight_pd.extend(tfidf_weight_scores[dic_idx])
        # attention_weight_pd.extend(attention_weight_scores[dic_idx])

    data = {
        'average_logprob': average_logprob_pd,
        'lowest_logprob': lowest_logprob_pd,
        'average_entropy': average_entropy_pd,
        'highest_entropy': highest_entropy_pd,
        'human_label_detect_True': human_label_detect_True_pd,
        'human_label_detect_False': human_label_detect_False_pd,
        # 'sentences': sentences_pd,
        # 'images': images_pd,
        # 'sentences_idx': sentences_idx_pd,
        # 'token_and_logprobs': token_and_logprobs_pd,
        'labels': labels_pd,
        # 'idx_info': idxs_pd,
        # 'tfidf_weight': tfidf_weight_pd,
        # 'attention_weight': attention_weight_pd
    }
    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by='average_logprob', ascending=False)  # , ascending=False
    return df_sorted


def analysis_sentence_level_info(average_logprob_scores, average_entropy_scores, lowest_logprob_scores,
                                 highest_entropy_scores
                                 , human_label, average_logprob_flag, average_entropy_flag,
                                 lowest_logprob_flag, highest_entropy_flag, save_path, name):
    # True
    # uncertainty
    Pb_average_logprob, Rb_average_logprob = get_PR_with_human_labels(average_logprob_scores,
                                                                      human_label, pos_label=1,
                                                                      oneminus_pred=average_logprob_flag)
    Pb_average_entropy, Rb_average_entropy = get_PR_with_human_labels(average_entropy_scores,
                                                                      human_label, pos_label=1,
                                                                      oneminus_pred=average_entropy_flag)
    Pb_lowest_logprob, Rb_lowest_logprob = get_PR_with_human_labels(lowest_logprob_scores, human_label,
                                                                    pos_label=1, oneminus_pred=lowest_logprob_flag)
    Pb_highest_entropy, Rb_highest_entropy = get_PR_with_human_labels(highest_entropy_scores,
                                                                      human_label, pos_label=1,
                                                                      oneminus_pred=highest_entropy_flag)

    print("-----------------------")
    print("Baseline1: Avg(logP)")
    print_AUC(Pb_average_logprob, Rb_average_logprob)
    print("-----------------------")
    print("Baseline2: Avg(H)")
    print_AUC(Pb_average_entropy, Rb_average_entropy)
    print("-----------------------")
    print("Baseline3: Max(logP)")
    print_AUC(Pb_lowest_logprob, Rb_lowest_logprob)
    print("-----------------------")
    print("Baseline4: Max(H)")
    print_AUC(Pb_highest_entropy, Rb_highest_entropy)

    random_baseline = np.mean(human_label)

    # with human label, Detecting Non-factual*
    if average_logprob_flag == True and average_entropy_flag == False and lowest_logprob_flag == True and highest_entropy_flag == False:
        label_average_logprob = '-Avg(logP)'
        label_average_entropy = 'Avg(H)'
        label_lowest_logprob = '-Max(logP)'
        label_highest_logprob = 'Max(H)'

    if average_logprob_flag == False and average_entropy_flag == True and lowest_logprob_flag == False and highest_entropy_flag == True:
        label_average_logprob = 'Avg(logP)'
        label_average_entropy = '-Avg(H)'
        label_lowest_logprob = 'Min(logP)'
        label_highest_logprob = '-Max(H)'

    fig = plt.figure(figsize=(5.5, 4.5))
    plt.hlines(y=random_baseline, xmin=0, xmax=1.0, color='grey', linestyles='dotted', label='Random')
    plt.plot(Rb_average_logprob, Pb_average_logprob, '-', label=label_average_logprob)
    plt.plot(Rb_average_entropy, Pb_average_entropy, '-', label=label_average_entropy)
    plt.plot(Rb_lowest_logprob, Pb_lowest_logprob, '-', label=label_lowest_logprob)
    plt.plot(Rb_highest_entropy, Pb_highest_entropy, '-', label=label_highest_logprob)
    plt.legend()
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.tight_layout()
    # plt.show()

    # save

    save_dir = save_path + '/figures/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + name + '.png'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


def analysis_sentence_level_self_check_info(total_mqag_scores, total_bert_scores, total_ngram_scores,
                                     total_nli_scores, total_true_labels, mqag_flag,
                                     bert_flag, ngram_flag, nli_flag, save_path, name):
    # True
    # uncertainty
    Pb_mqag, Rb_mqag = get_PR_with_human_labels(total_mqag_scores, total_true_labels, pos_label=1,
                                                oneminus_pred=mqag_flag)
    Pb_bert, Rb_bert = get_PR_with_human_labels(total_bert_scores, total_true_labels, pos_label=1,
                                                oneminus_pred=bert_flag)
    Pb_ngram, Rb_ngram = get_PR_with_human_labels(total_ngram_scores, total_true_labels, pos_label=1,
                                                oneminus_pred=ngram_flag)
    Pb_nli, Rb_nli = get_PR_with_human_labels(total_nli_scores, total_true_labels, pos_label=1,
                                              oneminus_pred=nli_flag)

    print("-----------------------")
    print("Baseline1: SelfCk-QA")
    print_AUC(Pb_mqag, Rb_mqag)
    print("-----------------------")
    print("Baseline2: SelfCk-BERTScore")
    print_AUC(Pb_bert, Rb_bert)
    print("-----------------------")
    print("Baseline3: SelfCk-Unigram")
    print_AUC(Pb_ngram, Rb_ngram)
    print("-----------------------")
    print("Baseline4: SelfCk-NLI")
    print_AUC(Pb_nli, Rb_nli)

    random_baseline = np.mean(total_true_labels)

    mqag = 'SelfCk-QA'
    bert = 'SelfCk-BERTScore'
    ngram = 'SelfCk-Unigram'
    nli = 'SelfCk-NLI'



    fig = plt.figure(figsize=(5.5, 4.5))
    plt.hlines(y=random_baseline, xmin=0, xmax=1.0, color='grey', linestyles='dotted', label='Random')
    plt.plot(Rb_mqag, Pb_mqag, '-', label=mqag)
    plt.plot(Rb_bert, Pb_bert, '-', label=bert)
    plt.plot(Rb_ngram, Pb_ngram, '-', label=ngram)
    plt.plot(Rb_nli, Pb_nli, '-', label=nli)
    plt.legend()
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.tight_layout()
    # plt.show()

    # save
    save_dir = save_path + '/figures/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + name + '.png'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


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
        # # 存储数据
        # df.to_csv(f"result/m_hal/{model_version}_m_hal_df.csv")
        #
        # # 读取数据
        # df = pd.read_csv(f"result/m_hal/{model_version}_m_hal_df.csv")

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

        # # 分析sentence level的相关数据
        print("False")
        average_logprob_flag = True  # False
        average_entropy_flag = False  # True
        lowest_logprob_flag = True  # False
        highest_entropy_flag = False  # True

        if data_flag == "neg_city":
            save_path = 'uncertainty/result/neg_city'
            name = f'{model_version}_neg_city_false'
        elif data_flag == "neg_company":
            save_path = 'uncertainty/result/neg_company'
            name = f'{model_version}_neg_company_false'
        elif data_flag == "neg_fact":
            save_path = 'uncertainty/result/neg_fact'
            name = f'{model_version}_neg_fact_false'

        analysis_sentence_level_info(average_logprob_scores, average_entropy_scores, lowest_logprob_scores,
                                     highest_entropy_scores, human_label_detect_False, average_logprob_flag,
                                     average_entropy_flag, lowest_logprob_flag, highest_entropy_flag, save_path, name)

        print("True")
        average_logprob_flag = False
        average_entropy_flag = True
        lowest_logprob_flag = False
        highest_entropy_flag = True

        if data_flag == "neg_city":
            save_path = 'uncertainty/result/neg_city'
            name = f'{model_version}_neg_city_false'
        elif data_flag == "neg_company":
            save_path = 'uncertainty/result/neg_company'
            name = f'{model_version}_neg_company_false'
        elif data_flag == "neg_fact":
            save_path = 'uncertainty/result/neg_fact'
            name = f'{model_version}_neg_fact_false'

        analysis_sentence_level_info(average_logprob_scores, average_entropy_scores, lowest_logprob_scores,
                                     highest_entropy_scores, human_label_detect_True, average_logprob_flag,
                                     average_entropy_flag, lowest_logprob_flag, highest_entropy_flag, save_path, name)
