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

base_base_loc = here / 'data'
base_loc = base_base_loc / 'ihad'
loc_Azip = base_loc / 'training_setA.zip'
loc_Bzip = base_loc / 'training_setB.zip'
data_loc = base_base_loc / 'ihad/training_setA/training'

# def download():
#     # pca = PCA(n_components=50)
#     # tsne = TSNE(n_components=50, perplexity=10, n_iter=1000)
#     with open("datasets/data/ihad/test_data_for_ode.bin", 'rb') as f:
#         keys_val = pickle.load(f)
#     x_all = keys_val['x']
#     x_pca = []
#     for ele in x_all:
#         ele_reduce = PCA.fit_transform(np.array(ele))
#         x_pca.append(ele_reduce)
#     x = []
#     for ele in x_pca:
#         sub = []
#         for i in range(len(ele)):
#             sub.append(ele[i])
#         x.append(sub)
#     y = np.array(keys_val['y'])
#     return x, y


def download():
    pca = PCA(n_components=10)
    with open("datasets/data/ihad/test_data_for_ode.bin", 'rb') as f:
        keys_val = pickle.load(f)
    x_all = keys_val['x']
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
    y = np.array(keys_val['y'])
    return x, y


def _process_data(X_times, y, time_intensity):
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
    times = torch.linspace(1, X_times.size(1), X_times.size(1))
    (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, _) = common.preprocess_data(times, X_times, y, final_indices, append_times=True,
                                                   append_intensity=time_intensity)
    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index)


def get_data(static_intensity, time_intensity, batch_size):
    base_base_loc = here / 'processed_data'
    loc = base_base_loc / ('ihad' + ('_staticintensity' if static_intensity else '_nostaticintensity') + ('_timeintensity' if time_intensity else '_notimeintensity'))
    if os.path.exists(loc):
        tensors = common.load_data(loc)
        times = tensors['times']
        train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
        val_coeffs = tensors['val_a'], tensors['val_b'], tensors['val_c'], tensors['val_d']
        test_coeffs = tensors['test_a'], tensors['test_b'], tensors['test_c'], tensors['test_d']
        train_y = tensors['train_y']
        val_y = tensors['val_y']
        test_y = tensors['test_y']
        train_final_index = tensors['train_final_index']
        val_final_index = tensors['val_final_index']
        test_final_index = tensors['test_final_index']
    else:
        x, y = download()
        (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
         test_final_index) = _process_data(x, y, time_intensity)
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        common.save_data(loc, times=times,
                         train_a=train_coeffs[0], train_b=train_coeffs[1], train_c=train_coeffs[2], train_d=train_coeffs[3],
                         val_a=val_coeffs[0], val_b=val_coeffs[1], val_c=val_coeffs[2], val_d=val_coeffs[3],
                         test_a=test_coeffs[0], test_b=test_coeffs[1], test_c=test_coeffs[2], test_d=test_coeffs[3],
                         train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
                         val_final_index=val_final_index, test_final_index=test_final_index)

    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data(times, train_coeffs, val_coeffs,
                                                                                test_coeffs, train_y, val_y, test_y,
                                                                                train_final_index, val_final_index,
                                                                                test_final_index, 'cpu',
                                                                                batch_size=batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader




