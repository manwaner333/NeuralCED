import csv
import math
import os
import pathlib
from numpy.lib.function_base import append
import torch
import urllib.request
import zipfile
import numpy as np
from . import truthful_qa_common
# import time_dataset
from sklearn.decomposition import PCA
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

here = pathlib.Path(__file__).resolve().parent


def download():
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    pca = PCA(n_components=20)
    with open("datasets/data/truthful_qa/answer_truthful_qa_300.bin", 'rb') as f:
        keys_val = pickle.load(f)

    question_ids = []
    labels = []
    question_lengths = []
    answer_lengths = []
    x_y_all_flattened = []
    for idx, ele in keys_val.items():
        question_hidden_states = ele['hidden_states']['ques']
        answer_hidden_states = ele['hidden_states']['answer']
        if np.isnan(question_hidden_states).any():
            print("question_hidden_states contains NaN values")
        if np.isnan(answer_hidden_states).any():
            print("answer_hidden_states contains NaN values")
        question_id = ele['question_id']
        label = ele['label']
        if label == 'ACCURATE':
            flag = 1
        else:
            flag = 0
        labels.append(flag)
        question_ids.append(question_id)
        ques_len = len(question_hidden_states)
        question_lengths.append(ques_len)
        answer_len = len(answer_hidden_states)
        answer_lengths.append(answer_len)
        for o1 in question_hidden_states:
            x_y_all_flattened.append(o1)
        for o2 in answer_hidden_states:
            x_y_all_flattened.append(o2)
    x_y_all_flattened_0_1 = scaler.fit_transform(x_y_all_flattened)
    x_y_all_flattened_reduce = pca.fit_transform(x_y_all_flattened_0_1)

    x = []
    y = []
    xy = []
    ques_start = 0
    for idx in range(len(question_lengths)):
        ques_len = question_lengths[idx]
        answer_len = answer_lengths[idx]
        sub_x = []
        sub_y = []
        sub_xy = []
        ques_end = ques_start + ques_len - 1
        for ele in range(ques_start, ques_end + 1):
            sub_x.append(x_y_all_flattened_reduce[ele, :])
            sub_xy.append(x_y_all_flattened_reduce[ele, :])
        answer_start = ques_end + 1
        answer_end = answer_start + answer_len - 1
        for ele in range(answer_start, answer_end + 1):
            sub_y.append(x_y_all_flattened_reduce[ele, :])
            sub_xy.append(x_y_all_flattened_reduce[ele, :])
        x.append(sub_x)
        y.append(sub_y)
        xy.append(sub_xy)
        ques_start = answer_end + 1
    labels = np.array(labels)
    x_lengths = np.array(question_lengths)
    y_lengths = np.array(answer_lengths)
    print(len(x), len(y), labels.tolist().count(1))
    return x, y, xy, question_ids, labels, x_lengths, y_lengths


def _process_data(X_times, y, xy, question_ids, labels, time_intensity=False):
    x_lengths = []
    for time in X_times:
        x_lengths.append(len(time))
    maxlen = max(x_lengths)
    time_values = len(X_times[0][0])
    for time in X_times:
        for _ in range(maxlen - len(time)):
            time.append([float('nan') for value in range(time_values)])

    y_lengths = []
    for sub_y in y:
        y_lengths.append(len(sub_y))
    y_maxlen = max(y_lengths)
    time_values = len(y[0][0])
    for sub_y in y:
        for _ in range(y_maxlen - len(sub_y)):
            sub_y.append([float('nan') for value in range(time_values)])

    xy_lengths = []
    for sub_xy in xy:
        xy_lengths.append(len(sub_xy))
    total_length = maxlen + y_maxlen
    time_values = len(xy[0][0])
    for sub_xy in xy:
        for _ in range(total_length - len(sub_xy)):
            sub_xy.append([float('nan') for value in range(time_values)])

    train_x = []
    train_y = []
    train_xy = []
    train_x_lengths = []
    train_y_lengths = []
    test_x = []
    test_y = []
    test_xy = []
    test_x_lengths = []
    test_y_lengths = []
    train_question_ids = []
    test_question_ids = []
    train_labels = []
    test_labels = []
    true_index = np.where(labels == 1)[0]
    false_index = np.where(labels == 0)[0]
    for idx in true_index[0:200]:
        train_x.append(X_times[idx])
        train_y.append(y[idx])
        train_xy.append(xy[idx])
        train_x_lengths.append(x_lengths[idx])
        train_y_lengths.append(y_lengths[idx])
        train_question_ids.append(question_ids[idx])
        train_labels.append(labels[idx])

    for idx in true_index[200:]:
        test_x.append(X_times[idx])
        test_y.append(y[idx])
        test_xy.append(xy[idx])
        test_x_lengths.append(x_lengths[idx])
        test_y_lengths.append(y_lengths[idx])
        test_question_ids.append(question_ids[idx])
        test_labels.append(labels[idx])

    for idx in false_index:
        test_x.append(X_times[idx])
        test_y.append(y[idx])
        test_xy.append(xy[idx])
        test_x_lengths.append(x_lengths[idx])
        test_y_lengths.append(y_lengths[idx])
        test_question_ids.append(question_ids[idx])
        test_labels.append(labels[idx])
    train_x_np = np.array(train_x)
    train_y_np = np.array(train_y)
    train_xy_np = np.array(train_xy)
    train_x = torch.tensor(train_x_np).float()
    train_y = torch.tensor(train_y_np).float()
    train_xy = torch.tensor(train_xy_np).float()
    train_x_lengths = torch.tensor(train_x_lengths)
    train_y_lengths = torch.tensor(train_y_lengths)
    train_question_ids = torch.tensor(train_question_ids)
    train_labels = torch.tensor(train_labels)

    test_x_np = np.array(test_x)
    test_y_np = np.array(test_y)
    test_xy_np = np.array(test_xy)
    test_x = torch.tensor(test_x_np).float()
    test_y = torch.tensor(test_y_np).float()
    test_xy = torch.tensor(test_xy_np).float()
    test_x_lengths = torch.tensor(test_x_lengths)
    test_y_lengths = torch.tensor(test_y_lengths)
    test_question_ids = torch.tensor(test_question_ids)
    test_labels = torch.tensor(test_labels)


    times = torch.linspace(0, total_length - 1, total_length)
    x_times = torch.linspace(0, maxlen - 1, maxlen)
    x_times, times, train_coeffs, train_X, train_y, train_xy, train_x_lengths, train_y_lengths, train_question_ids, train_labels, _ = truthful_qa_common.preprocess_data(x_times, times, train_x, train_y, train_xy, train_x_lengths, train_y_lengths, train_question_ids, train_labels, append_times=False, append_intensity=time_intensity)
    x_times, times, test_coeffs, test_X, test_y, test_xy, test_x_lengths, test_y_lengths, test_question_ids, test_labels, _ = truthful_qa_common.preprocess_data(x_times, times, test_x, test_y, test_xy, test_x_lengths, test_y_lengths, test_question_ids, test_labels, append_times=False, append_intensity=time_intensity)
    x_times, times, val_coeffs, val_X, val_y, val_xy, val_x_lengths, val_y_lengths, val_question_ids, val_labels, _ = truthful_qa_common.preprocess_data(x_times, times, test_x, test_y, test_xy, test_x_lengths, test_y_lengths, test_question_ids, test_labels, append_times=False, append_intensity=time_intensity)
    return (x_times, times, train_coeffs, val_coeffs, test_coeffs, train_X, val_X, test_X, train_y, val_y, test_y, train_xy, val_xy, test_xy, train_x_lengths, val_x_lengths,
            test_x_lengths, train_y_lengths, val_y_lengths, test_y_lengths, train_question_ids, val_question_ids, test_question_ids, train_labels, val_labels, test_labels)


def get_data(batch_size, missing_rate, append_time, time_seq, y_seq):
    base_base_loc = here / 'processed_data'

    if append_time:
        loc = base_base_loc / ('truthful_qa' + str(time_seq) + '_' + str(y_seq) + '_' + str(missing_rate) + '_time_aug')
    else:
        loc = base_base_loc / ('truthful_qa' + str(time_seq) + '_' + str(y_seq) + '_' + str(missing_rate))
    if os.path.exists(loc):
        tensors = truthful_qa_common.load_data(loc)
        x_times = tensors['x_times']
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
        train_xy = tensors['train_xy']
        val_xy = tensors['val_xy']
        test_xy = tensors['test_xy']
        train_x_lengths = tensors['train_x_lengths']
        val_x_lengths = tensors['val_x_lengths']
        test_x_lengths = tensors['test_x_lengths']
        train_y_lengths = tensors['train_y_lengths']
        val_y_lengths = tensors['val_y_lengths']
        test_y_lengths = tensors['test_y_lengths']
        train_question_ids = tensors['train_question_ids']
        val_question_ids = tensors['val_question_ids']
        test_question_ids = tensors['test_question_ids']
        train_labels = tensors['train_labels']
        val_labels = tensors['val_labels']
        test_labels = tensors['test_labels']
    else:
        x, y,  xy, question_ids, labels, x_lengths, y_lengths = download()
        (x_times, times, train_coeffs, val_coeffs, test_coeffs, train_X, val_X, test_X, train_y, val_y, test_y, train_xy, val_xy, test_xy, train_x_lengths, val_x_lengths,
         test_x_lengths, train_y_lengths, val_y_lengths, test_y_lengths, train_question_ids, val_question_ids, test_question_ids, train_labels, val_labels, test_labels) = _process_data(x, y, xy, question_ids, labels)

        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        truthful_qa_common.save_data(loc, x_times=x_times, times=times,
                         train_a=train_coeffs[0], train_b=train_coeffs[1], train_c=train_coeffs[2],
                         train_d=train_coeffs[3],
                         val_a=val_coeffs[0], val_b=val_coeffs[1], val_c=val_coeffs[2], val_d=val_coeffs[3],
                         test_a=test_coeffs[0], test_b=test_coeffs[1], test_c=test_coeffs[2], test_d=test_coeffs[3],
                         train_X=train_X, val_X=val_X, test_X=test_X,
                         train_y=train_y, val_y=val_y, test_y=test_y,
                         train_xy=train_xy, val_xy=val_xy, test_xy=test_xy, train_x_lengths=train_x_lengths,
                         val_x_lengths=val_x_lengths, test_x_lengths=test_x_lengths,
                         train_y_lengths=train_y_lengths, val_y_lengths=val_y_lengths, test_y_lengths=test_y_lengths,
                         train_question_ids=train_question_ids, val_question_ids=val_question_ids, test_question_ids=test_question_ids,
                        train_labels=train_labels, val_labels=val_labels, test_labels=test_labels)

    x_times, times, train_dataloader, val_dataloader, test_dataloader = truthful_qa_common.wrap_data(x_times, times, train_coeffs, val_coeffs,
                                                                                test_coeffs, train_X, val_X, test_X, train_y, val_y, test_y,
                                                                                train_xy, val_xy, test_xy,
                                                                                train_x_lengths, val_x_lengths, test_x_lengths,
                                                                                train_y_lengths, val_y_lengths, test_y_lengths,
                                                                                train_question_ids, val_question_ids, test_question_ids,
                                                                                train_labels, val_labels, test_labels,
                                                                                'cpu', batch_size=batch_size)

    return x_times, times, train_dataloader, val_dataloader, test_dataloader
