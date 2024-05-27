import csv
import math
import os
import pathlib
import torch
import urllib.request
import zipfile
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from . import common
import pickle
import numpy as np

here = pathlib.Path(__file__).resolve().parent

# 暂时用一下
def download(dataset_name, num_dim):
    pca = PCA(n_components=num_dim)
    file_name = "datasets/data/" + dataset_name + "/" + "answer_" + dataset_name + ".bin"

    with open(file_name, 'rb') as f:
        keys_val = pickle.load(f)

    hidden_states = []
    labels = []
    question_ids = []
    tokens = []
    for idx, ele in keys_val.items():
        question_id = ele['question_id']
        extract_hidden_states = ele["logprobs"]['combined_hidden_states']
        question_hidden_states = extract_hidden_states['ques']
        if np.isnan(question_hidden_states).any():
            print("question_hidden_states contains NaN values")
        figure_hidden_states = extract_hidden_states['fig']
        sentence_length = len(ele["sentences"])
        for i in range(sentence_length):
            sub_hidden_states = []
            sentence_hidden_states = extract_hidden_states[i]
            sub_hidden_states.extend(sentence_hidden_states)
            hidden_states.append(sub_hidden_states)

        for i in range(sentence_length):
            label = ele['labels'][i]
            if label in ['ACCURATE', 'ANALYSIS']:
                labels.append(0)
            else:
                labels.append(1)
            question_ids.append(question_id)

    x_all = hidden_states
    y = labels
    x_all_flattened = []
    length = []
    for ele in x_all:
        length.append(len(ele))
        for o in ele:
            x_all_flattened.append(o)
    x_all_flattened_reduce = pca.fit_transform(x_all_flattened)
    max_length = max(length)
    x = []
    start = 0
    for le in length:
        sub_x = []
        end = start + le
        for idx in range(start, end):
            sub_x.append(x_all_flattened_reduce[idx, :])
        x.append(sub_x)
        start = end
    y = np.array(y)
    print("The size of train data {}; The number of False cases {}".format(len(x), y.tolist().count(1)))
    return x, y, question_ids, tokens, max_length

def _process_data(X_times, y, question_ids, max_length, time_intensity):
    final_indices = []
    for time in X_times:
        final_indices.append(len(time) - 1)
    # maxlen = max(final_indices) + 1
    maxlen = max_length
    time_values = len(X_times[0][0])
    for time in X_times:
        for _ in range(maxlen - len(time)):
            time.append([float('nan') for value in range(time_values)])
    X_times = torch.tensor(np.array(X_times)).float()
    y = torch.tensor(y).float()
    final_indices = torch.tensor(final_indices)
    question_ids = torch.tensor(question_ids)
    times = torch.linspace(1, X_times.size(1), X_times.size(1))

    (times, coeffs, y, final_index, question_ids, X, _) = common.preprocess_data(times, X_times, y, final_indices, question_ids, append_times=True,
                                                   append_intensity=time_intensity)
    return (times, coeffs, y, final_index, question_ids, X)


def get_data(train_dataset_name, test_dataset_name, static_intensity, time_intensity, batch_size, num_dim):
    base_base_loc = here / 'processed_data'
    loc = base_base_loc / (train_dataset_name + ('_staticintensity' if static_intensity else '_nostaticintensity') + ('_timeintensity' if time_intensity else '_notimeintensity'))

    if os.path.exists(loc):
        tensors = common.load_data(loc)
        times = tensors['times']
        train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
        test_coeffs = tensors['test_a'], tensors['test_b'], tensors['test_c'], tensors['test_d']
        train_X = tensors['train_X']
        test_X = tensors['test_X']
        train_y = tensors['train_y']
        test_y = tensors['test_y']
        train_final_index = tensors['train_final_index']
        test_final_index = tensors['test_final_index']
        train_question_ids = tensors['train_question_ids']
        test_question_ids = tensors['test_question_ids']
    else:
        train_x, train_y, train_question_ids, train_tokens, train_max_length = download(train_dataset_name, num_dim)
        test_x,  test_y,  test_question_ids,  test_tokens, test_max_length = download(test_dataset_name, num_dim)
        if train_max_length >= test_max_length:
            max_length = train_max_length
        else:
            max_length = test_max_length

        times, train_coeffs, train_y, train_final_index, train_question_ids, train_X = _process_data(train_x, train_y, train_question_ids, max_length, time_intensity)
        times, test_coeffs, test_y, test_final_index, test_question_ids, test_X = _process_data(test_x, test_y, test_question_ids, max_length, time_intensity)

        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        common.save_data(loc, times=times,
                         train_a=train_coeffs[0], train_b=train_coeffs[1], train_c=train_coeffs[2], train_d=train_coeffs[3],
                         test_a=test_coeffs[0], test_b=test_coeffs[1], test_c=test_coeffs[2], test_d=test_coeffs[3],
                         train_y=train_y, test_y=test_y, train_final_index=train_final_index, test_final_index=test_final_index,
                         train_question_ids=train_question_ids, test_question_ids=test_question_ids,
                         train_X=train_X, test_X=test_X)
    print(len(train_X), len(test_X))
    times, train_dataloader, test_dataloader = common.wrap_data(times, train_coeffs, test_coeffs, train_X, test_X, train_y, test_y,
                                                                                train_final_index, test_final_index, train_question_ids, test_question_ids,
                                                                                'cpu', batch_size=batch_size)

    return times, train_dataloader, test_dataloader




