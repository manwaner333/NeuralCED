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
    if dataset_name == "company":
        file_name = "datasets/data/company/answer_company.bin"
    else:
        file_name = "aaaaa"

    with open(file_name, 'rb') as f:
        keys_val = pickle.load(f)

    hidden_states = []
    labels = []
    question_ids = []
    tokens = []
    for idx, ele in keys_val.items():
        sub_hidden_states = []
        token_sub = []
        question_id = ele['question_id']
        token_sub.extend(ele['tokens']['ques'])
        question_hidden_states = ele['hidden_states']['ques']
        if np.isnan(question_hidden_states).any():
            print("question_hidden_states contains NaN values")
        sub_hidden_states.extend(question_hidden_states)
        hidden_states.append(sub_hidden_states)
        label = ele['label']
        labels.append(label)
        question_ids.append(question_id)
        tokens.append(token_sub)

    x_all = hidden_states
    y = labels
    x_all_flattened = []
    length = []
    for ele in x_all:
        length.append(len(ele))
        for o in ele:
            x_all_flattened.append(o)
    x_all_flattened_reduce = pca.fit_transform(x_all_flattened)

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
    return x, y, question_ids, tokens

def _process_data(X_times, y, question_ids, time_intensity):
    final_indices = []
    for time in X_times:
        final_indices.append(len(time) - 1)
    maxlen = max(final_indices) + 1
    time_values = len(X_times[0][0])
    for time in X_times:
        for _ in range(maxlen - len(time)):
            time.append([float('nan') for value in range(time_values)])
    X_times = torch.tensor(X_times).float()
    y = torch.tensor(y).float()
    final_indices = torch.tensor(final_indices)
    question_ids = torch.tensor(question_ids)
    times = torch.linspace(1, X_times.size(1), X_times.size(1))

    (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, train_question_ids, val_question_ids, test_question_ids, train_X, val_X, test_X, _) = common.preprocess_data(times, X_times, y, final_indices, question_ids, append_times=True,
                                                   append_intensity=time_intensity)
    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index, train_question_ids, val_question_ids, test_question_ids, train_X, val_X, test_X)


def get_data(dataset_name, static_intensity, time_intensity, batch_size, num_dim):
    base_base_loc = here / 'processed_data'
    if dataset_name == "company":
        loc = base_base_loc / ('company' + ('_staticintensity' if static_intensity else '_nostaticintensity') + ('_timeintensity' if time_intensity else '_notimeintensity'))
    else:
        loc = "aaa"
    if os.path.exists(loc):
        tensors = common.load_data(loc)
        times = tensors['times']
        train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
        val_coeffs = tensors['val_a'], tensors['val_b'], tensors['val_c'], tensors['val_d']
        test_coeffs = tensors['test_a'], tensors['test_b'], tensors['test_c'], tensors['test_d']
        train_X = tensors['train_X']
        val_X = tensors['val_X']
        test_X = tensors['test_X']
        train_y = tensors['train_y']
        val_y = tensors['val_y']
        test_y = tensors['test_y']
        train_final_index = tensors['train_final_index']
        val_final_index = tensors['val_final_index']
        test_final_index = tensors['test_final_index']
        train_question_ids = tensors['train_question_ids']
        val_question_ids = tensors['val_question_ids']
        test_question_ids = tensors['test_question_ids']
    else:
        x, y, question_ids, tokens = download(dataset_name, num_dim)
        (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
         test_final_index, train_question_ids, val_question_ids, test_question_ids, train_X, val_X, test_X) = _process_data(x, y, question_ids, time_intensity)
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        common.save_data(loc, times=times,
                         train_a=train_coeffs[0], train_b=train_coeffs[1], train_c=train_coeffs[2], train_d=train_coeffs[3],
                         val_a=val_coeffs[0], val_b=val_coeffs[1], val_c=val_coeffs[2], val_d=val_coeffs[3],
                         test_a=test_coeffs[0], test_b=test_coeffs[1], test_c=test_coeffs[2], test_d=test_coeffs[3],
                         train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
                         val_final_index=val_final_index, test_final_index=test_final_index,
                         train_question_ids=train_question_ids, val_question_ids=val_question_ids,
                         train_X=train_X, val_X=val_X, test_X=test_X, test_question_ids=test_question_ids)
    print(len(train_X), len(val_X), len(test_X))
    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data(times, train_coeffs, val_coeffs,
                                                                                test_coeffs, train_X, val_X, test_X, train_y, val_y, test_y,
                                                                                train_final_index, val_final_index,
                                                                                test_final_index, train_question_ids, val_question_ids, test_question_ids,
                                                                                'cpu', batch_size=batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader




