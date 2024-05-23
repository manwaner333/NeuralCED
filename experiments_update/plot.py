import csv
import math
import os
import pathlib
import torch
import urllib.request
import zipfile
from sklearn.decomposition import PCA
import pickle
import numpy as np
import matplotlib.pyplot as plt


# def download():
#     pca = PCA(n_components=10)
#     with open("datasets/data/ihad/test_data_for_ode.bin", 'rb') as f:
#         keys_val = pickle.load(f)
#     x_all = keys_val['x']
#     x_pca = []
#     for ele in x_all:
#         ele_reduce = pca.fit_transform(ele)
#         x_pca.append(ele_reduce)
#     x = []
#     for ele in x_pca:
#         sub = []
#         for i in range(len(ele)):
#             sub.append(ele[i])
#         x.append(sub)
#     y = np.array(keys_val['y'])
#     return x, y
#
#
# x, y = download()
#
# with open('list.pkl', 'wb') as file:
#     pickle.dump(x, file)

with open('list.pkl', 'rb') as file:
    x = pickle.load(file)

fig, ax = plt.subplots(figsize=(6, 6))
x_sub = range(0, 10)
y_sub0 = x[0][0]
y_sub1 = x[0][1]
y_sub2 = x[0][2]
y_sub3 = x[0][3]
y_sub4 = x[0][4]
y_sub5 = x[0][5]
ax.plot(x_sub, y_sub0, linewidth=1.2, linestyle='solid', label='Clear Image')
ax.plot(x_sub, y_sub1, linewidth=1.2, linestyle='solid', label='Clear Image')
ax.plot(x_sub, y_sub2, linewidth=1.2, linestyle='solid', label='Clear Image')
ax.plot(x_sub, y_sub3, linewidth=1.2, linestyle='solid', label='Clear Image')
ax.plot(x_sub, y_sub4, linewidth=1.2, linestyle='solid', label='Clear Image')
ax.plot(x_sub, y_sub5, linewidth=1.2, linestyle='solid', label='Clear Image')
plt.show()