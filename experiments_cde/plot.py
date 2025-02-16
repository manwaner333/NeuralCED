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


# with open('list.pkl', 'rb') as file:
#     x = pickle.load(file)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# x_sub = range(0, 10)
# y_sub0 = x[0][0]
# y_sub1 = x[0][1]
# y_sub2 = x[0][2]
# y_sub3 = x[0][3]
# y_sub4 = x[0][4]
# y_sub5 = x[0][5]
# ax.plot(x_sub, y_sub0, linewidth=1.2, linestyle='solid', label='Clear Image')
# ax.plot(x_sub, y_sub1, linewidth=1.2, linestyle='solid', label='Clear Image')
# ax.plot(x_sub, y_sub2, linewidth=1.2, linestyle='solid', label='Clear Image')
# ax.plot(x_sub, y_sub3, linewidth=1.2, linestyle='solid', label='Clear Image')
# ax.plot(x_sub, y_sub4, linewidth=1.2, linestyle='solid', label='Clear Image')
# ax.plot(x_sub, y_sub5, linewidth=1.2, linestyle='solid', label='Clear Image')
# plt.show()


"""
# 画出不同模型下每个数据集的表现
models = ['LLama_7b', 'LLama_13b', 'Alpaca_13b', 'Vicuna_13b']
company = [19.8, 20.2, 16.0, 11.6]
fact = [12.0, 12.4, 13.1, 11.1]
city = [19.1, 20.5, 14.9, 18.0]
invention = [16.7, 18.0, 21.8, 19.6]

# Number of groups
n_groups = len(models)

# Create the bar plot
fig, ax = plt.subplots(figsize=(15, 10))

index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8

# Plot each set of bars for the different methods
# rects1 = ax.bar(index[0], vanilla[0], bar_width, alpha=opacity, label='trained on Adv') # Independent Classifier    color='darkorange',
# rects2 = ax.bar(index[1], vanilla[1], bar_width, alpha=opacity, label='trained on Ran')
# rects3 = ax.bar(index[2], vanilla[2], bar_width, alpha=opacity, label='trained on GQA')
# rects4 = ax.bar(index[3], vanilla[3], bar_width, alpha=opacity,  label='trained on M-Hal')
# rects5 = ax.bar(index[4], vanilla[4], bar_width, alpha=opacity,  label='trained on IHAD')
# rects6 = ax.bar(index+bar_width, freeCheck, bar_width, alpha=opacity,  label='trained on Pop')

company = ax.bar(index, company, bar_width, alpha=opacity, label=r'Company$^{*}$')
fact = ax.bar(index+bar_width, fact, bar_width, alpha=opacity, label=r'Fact$^{*}$')
city = ax.bar(index+bar_width+bar_width, city, bar_width, alpha=opacity, label=r'City$^{*}$')
invention = ax.bar(index+bar_width+bar_width+bar_width, invention, bar_width, alpha=opacity, label=r'Invention$^{*}$')

# rects2 = ax.bar(0 + bar_width, freeCheck[0], bar_width, alpha=opacity,  label='trained on Adv')  # color='lightseagreen',
# rects3 = ax.bar(1 + bar_width, freeCheck[1], bar_width, alpha=opacity,  label='trained on Ran')
# rects4 = ax.bar(2 + bar_width, freeCheck[2], bar_width, alpha=opacity,  label='trained on GQA')
# rects5 = ax.bar(3 + bar_width, freeCheck[3], bar_width, alpha=opacity,  label='trained on M-Hal')  # color='lightseagreen',
# rects6 = ax.bar(4 + bar_width, freeCheck[4], bar_width, alpha=opacity,  label='trained on IHAD')


# Add some text for labels, title, and axes ticks
ax.set_xlabel('Data', fontdict={'fontsize': 30})  #, 'fontweight': 'bold'
ax.set_ylabel(r'$\Delta\text{AUC-ROC}$', fontdict={'fontsize': 30})  #, 'fontweight': 'bold'
# ax.set_title('Accuracy by model and method')
ax.set_xticks(index + 2*bar_width)
ax.set_xticklabels(models, )  # fontweight='bold'
ax.set_yticklabels(ax.get_yticks()) #, fontweight='bold'
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)

# ax.legend()
ax.legend(loc='best', ncol=4, prop={'size': 20}, framealpha=0.3) #, 'weight': 'bold'
# Display the plot   bbox_to_anchor=(0.5, 1.13)
plt.tight_layout()
# plt.show()

fig.savefig("different_datasets_same_model.png", bbox_inches='tight', pad_inches=0.5)
"""

"""
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 26})

# Data from the table
tasks = ['Fauxtography', '$\mathrm{MR}^2$'] # 'LatestSnopes',
evidence_pairs = ["Company", "Fact", "City", "Invention"]
data = {
    'llama_7b': {
        'CDE': [65.9, 67.5, 75.7, 75.9],
        'SDE': [73.8, 70.3, 79.1, 68.7],
        'ODE': [59.7, 58.6, 73.0, 60.3],
        'Probing': [54.0, 58.3, 60.0, 59.2],
    },
    'llama_13b': {
        'CDE': [72.8, 74.8, 80.6, 88.3],
        'SDE': [78.4, 73.1, 89.8, 79.6],
        'ODE': [65.3, 66.9, 82.3, 80.9],
        'Probing': [58.2, 62.4, 69.3, 66.0],
    },
    'alpaca_13b': {
        'CDE': [75.3, 72.9, 72.1, 73.8],
        'SDE': [70.5, 70.3, 74.3, 74.2],
        'ODE': [67.8, 64.3, 71.2, 69.7],
        'Probing': [59.3, 59.8, 59.4, 52.4],
    },
    'vicuna_13b': {
        'CDE': [79.8, 76.7, 80.1, 81.2],
        'SDE': [72.3, 78.6, 82.5, 85.9],
        'ODE': [72.9, 70.4, 73.2, 80.4],
        'Probing': [68.2, 65.5, 64.5, 69.3],
    }
}

fontsize = 30
linewidth=3.0
# Plotting 3 subplots in one row
fig, axes = plt.subplots(2, 2, figsize=(24, 20), sharey=False)
axes[0][0].plot(evidence_pairs, data['llama_7b']['CDE'], marker='o', label="Llama_7b", linewidth=linewidth)
axes[0][0].plot(evidence_pairs, data['llama_13b']['CDE'], marker='o', label="Llama_13b", linewidth=linewidth)
axes[0][0].plot(evidence_pairs, data['alpaca_13b']['CDE'], marker='o', label="Alpaca_13b", linewidth=linewidth)
axes[0][0].plot(evidence_pairs, data['vicuna_13b']['CDE'], marker='o', label="Vicuna_13b", linewidth=linewidth)
axes[0][0].set_xlabel(r'NeuralCDE', fontsize=fontsize, fontweight='bold')
axes[0][0].set_ylabel(r'$\mathbf{AUC-ROC}$ using NeuralCDE', fontsize=fontsize, fontweight='bold')
axes[0][0].set_xticklabels(axes[0][0].get_xticklabels(), fontweight='bold')
axes[0][0].set_yticklabels(axes[0][0].get_yticklabels(), fontweight='bold')
axes[0][0].legend(prop={'weight': 'bold'})

axes[0][1].plot(evidence_pairs, data['llama_7b']['SDE'], marker='o', label="Llama_7b", linewidth=linewidth)
axes[0][1].plot(evidence_pairs, data['llama_13b']['SDE'], marker='o', label="Llama_13b", linewidth=linewidth)
axes[0][1].plot(evidence_pairs, data['alpaca_13b']['SDE'], marker='o', label="Alpaca_13b", linewidth=linewidth)
axes[0][1].plot(evidence_pairs, data['vicuna_13b']['SDE'], marker='o', label="Vicuna_13b", linewidth=linewidth)
axes[0][1].set_xlabel(r'NeuralSDE', fontsize=fontsize, fontweight='bold')
axes[0][1].set_ylabel(r'$\mathbf{AUC-ROC}$ using NeuralSDE', fontsize=fontsize, fontweight='bold')
axes[0][1].set_xticklabels(axes[0][1].get_xticklabels(), fontweight='bold')
axes[0][1].set_yticklabels(axes[0][1].get_yticklabels(), fontweight='bold')
axes[0][1].legend(prop={'weight': 'bold'})

axes[1][0].plot(evidence_pairs, data['llama_7b']['ODE'], marker='o', label="Llama_7b", linewidth=linewidth)
axes[1][0].plot(evidence_pairs, data['llama_13b']['ODE'], marker='o', label="Llama_13b", linewidth=linewidth)
axes[1][0].plot(evidence_pairs, data['alpaca_13b']['ODE'], marker='o', label="Alpaca_13b", linewidth=linewidth)
axes[1][0].plot(evidence_pairs, data['vicuna_13b']['ODE'], marker='o', label="Vicuna_13b", linewidth=linewidth)
axes[1][0].set_xlabel(r'NeuralODE', fontsize=fontsize, fontweight='bold')
axes[1][0].set_ylabel(r'$\mathbf{AUC-ROC}$ using NeuralODE', fontsize=fontsize, fontweight='bold')
axes[1][0].set_xticklabels(axes[1][0].get_xticklabels(), fontweight='bold')
axes[1][0].set_yticklabels(axes[1][0].get_yticklabels(), fontweight='bold')
axes[1][0].legend(prop={'weight': 'bold'})



axes[1][1].plot(evidence_pairs, data['llama_7b']['Probing'], marker='o', label="Llama_7b", linewidth=linewidth)
axes[1][1].plot(evidence_pairs, data['llama_13b']['Probing'], marker='o', label="Llama_13b", linewidth=linewidth)
axes[1][1].plot(evidence_pairs, data['alpaca_13b']['Probing'], marker='o', label="Alpaca_13b", linewidth=linewidth)
axes[1][1].plot(evidence_pairs, data['vicuna_13b']['Probing'], marker='o', label="Vicuna_13b", linewidth=linewidth)
axes[1][1].set_xlabel(r'Probing', fontsize=fontsize, fontweight='bold')
axes[1][1].set_ylabel(r'$\mathbf{AUC-ROC}$ using Probing', fontsize=fontsize, fontweight='bold')
axes[1][1].set_xticklabels(axes[1][1].get_xticklabels(), fontweight='bold')
axes[1][1].set_yticklabels(axes[1][1].get_yticklabels(), fontweight='bold')
axes[1][1].legend(prop={'weight': 'bold'})

spine_size = 2.0
for i in range(0, 2):
    for j in range(0, 2):
        for spine in axes[i][j].spines.values():
            spine.set_linewidth(spine_size)
            spine.set_edgecolor('black')


plt.tight_layout()  # rect=[0.0, -0.05, 1, 0.94]

# plt.show()

fig.savefig("different_models.png", bbox_inches='tight', pad_inches=0.5)


"""


"""
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
data = {
    'ODE': [49.7, 53.1, 59.8, 65.3, 66.1, 65.0, 64.8],
    'CDE': [60.1, 65.3, 72.8, 73.0, 73.2, 73.5, 73.1],
    'SDE': [58.1, 64.7, 70.1, 78.4, 78.1, 78.9, 78.6]
}
evidence_pairs = [2, 4, 6, 8, 10, 12, 14]
fontsize = 20
linewidth=2.5
# Plotting 3 subplots in one row
fig, axe = plt.subplots(1, 1, figsize=(8, 6), sharey=False)
axe.plot(evidence_pairs, data['ODE'], marker='o', label="ODE", linewidth=linewidth)
axe.plot(evidence_pairs, data['CDE'], marker='o', label="CDE", linewidth=linewidth)
axe.plot(evidence_pairs, data['SDE'], marker='o', label="SDE", linewidth=linewidth)
axe.set_xlabel(r'The number of hidden layer', fontsize=fontsize)
axe.set_ylabel(r'$\text{AUC-ROC}$(%)', fontsize=fontsize)
axe.set_xticks(evidence_pairs)
axe.set_xticklabels(evidence_pairs, fontsize=20)
axe.legend()
plt.tight_layout()  # rect=[0.0, -0.05, 1, 0.94]
# plt.show()

fig.savefig("layer_number.png", bbox_inches='tight', pad_inches=0.5)

"""


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 26})

# Data from the table
evidence_pairs = ['4', '6', '8', '10', '12', '14']
# vicuna
data = {
    'Company': {
        'ODE': [49.7, 56.9, 65.7, 72.9, 73.9, 75.1],
        'CDE': [59.8, 70.2, 79.8, 81.1, 82.9, 83.6],
        'SDE': [55.7, 60.9, 64.1, 72.3, 72.5, 73.9],
    },
    'Truthful_qa': {
        'ODE': [58.7, 65.3, 79.6, 83.8, 84.5, 85.1],
        'CDE': [61.6, 70.5, 89.2, 90.7, 91.9, 92.2],
        'SDE': [64.8, 71.9, 81.7, 89.5, 89.9, 91.4],
    }
}

fontsize = 25
linewidth=3.0
# Plotting 3 subplots in one row
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)
axes[0].set_facecolor('#f0f0f0')
axes[0].plot(evidence_pairs, data['Company']['ODE'], marker='o', label="Neural ODEs", linewidth=linewidth)
axes[0].plot(evidence_pairs, data['Company']['CDE'], marker='o', label="Neural CDEs", linewidth=linewidth)
axes[0].plot(evidence_pairs, data['Company']['SDE'], marker='o', label="Neural SDEs", linewidth=linewidth)
axes[0].set_xlabel(r'Number of Hidden Layer', fontsize=fontsize,)  # fontweight='bold'
axes[0].set_ylabel(r'$\text{AUC-ROC}$(%)', fontsize=fontsize,)
axes[0].set_xticks(evidence_pairs)
axes[0].set_xticklabels(evidence_pairs, )
axes[0].set_xticklabels(evidence_pairs, )
axes[0].set_yticklabels(axes[0].get_yticklabels(),)
axes[0].set_title(r'$\text{Company}^{*}$', fontsize=30, loc='center')
axes[0].grid(True, which='both', linestyle='-', color='white', linewidth=2.0)
axes[0].legend(frameon=True, framealpha=0.3)  #prop={'weight': 'bold'}

axes[1].set_facecolor('#f0f0f0')
axes[1].plot(evidence_pairs, data['Truthful_qa']['ODE'], marker='o', label="Neural ODEs", linewidth=linewidth)
axes[1].plot(evidence_pairs, data['Truthful_qa']['CDE'], marker='o', label="Neural CDEs", linewidth=linewidth)
axes[1].plot(evidence_pairs, data['Truthful_qa']['SDE'], marker='o', label="Neural SDEs", linewidth=linewidth)
axes[1].set_xlabel(r'Number of Hidden Layer', fontsize=fontsize,)
axes[1].set_ylabel(r'$\text{AUC-ROC}$(%)', fontsize=fontsize,)  # (r'$\mathbf{AUC-ROC}$(%)'
axes[1].set_xticks(evidence_pairs)
axes[1].set_xticklabels(evidence_pairs, )
axes[1].set_xticklabels(evidence_pairs,)
axes[1].set_yticklabels(axes[1].get_yticklabels(),)
axes[1].set_title('TruthfulQA', fontsize=30, loc='center')
axes[1].grid(True, which='both', linestyle='-', color='white', linewidth=2.0)
axes[1].legend(frameon=True, framealpha=0.3)  # prop={'weight': 'bold'}


spine_size = 2.0
for i in range(0, 2):
    for spine in axes[i].spines.values():
        spine.set_linewidth(spine_size)
        spine.set_edgecolor('black')


plt.tight_layout()  # rect=[0.0, -0.05, 1, 0.94]

# plt.show()

fig.savefig("layer_number_update.png", bbox_inches='tight', pad_inches=0.5)

"""


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 26})

# Data from the table
evidence_pairs = ["256", "512", "1024", "2048"]
# vicuna
data = {
    'Company': {
        'ODE': [60.1, 66.3, 72.9, 73.0],
        'CDE': [63.2, 70.1, 79.8, 80.0],
        'SDE': [61.5, 67.8, 72.3, 72.4],
    },
    'Truthful_qa': {
        'ODE': [68.4, 76.5, 83.8, 84.0],
        'CDE': [70.3, 77.9, 89.2, 89.0],
        'SDE': [69.8, 80.1, 89.5, 90.0],
    }
}

fontsize = 25
linewidth=3.0
# Plotting 3 subplots in one row
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)

axes[0].plot(evidence_pairs, data['Company']['ODE'], marker='o', label="Neural ODEs", linewidth=linewidth)
axes[0].plot(evidence_pairs, data['Company']['CDE'], marker='o', label="Neural CDEs", linewidth=linewidth)
axes[0].plot(evidence_pairs, data['Company']['SDE'], marker='o', label="Neural SDEs", linewidth=linewidth)
axes[0].set_xlabel(r'Number of PCA Dimension', fontsize=fontsize,)  # fontweight='bold'
axes[0].set_ylabel(r'$\text{AUC-ROC}$(%)', fontsize=fontsize,)
axes[0].set_xticks(evidence_pairs)
axes[0].set_xticklabels(evidence_pairs, )
axes[0].set_yticklabels(axes[0].get_yticklabels(),)
axes[0].grid(True, which='both', linestyle='-', color='white', linewidth=2.0)
axes[0].set_facecolor('#f0f0f0')
axes[0].set_title(r'$\text{Company}^{*}$', fontsize=30, loc='center')
axes[0].legend(frameon=True, framealpha=0.3)  #prop={'weight': 'bold'}

axes[1].plot(evidence_pairs, data['Truthful_qa']['ODE'], marker='o', label="Neural ODEs", linewidth=linewidth)
axes[1].plot(evidence_pairs, data['Truthful_qa']['CDE'], marker='o', label="Neural CDEs", linewidth=linewidth)
axes[1].plot(evidence_pairs, data['Truthful_qa']['SDE'], marker='o', label="Neural SDEs", linewidth=linewidth)
axes[1].set_xlabel(r'Number of PCA Dimension', fontsize=fontsize,)
axes[1].set_ylabel(r'$\text{AUC-ROC}$(%)', fontsize=fontsize,)  # (r'$\mathbf{AUC-ROC}$(%)'
axes[1].set_xticks(evidence_pairs)
axes[1].set_xticklabels(evidence_pairs,)

axes[1].set_yticklabels(axes[1].get_yticklabels(),)
axes[1].set_title('TruthfulQA', fontsize=30, loc='center')
axes[1].legend(frameon=True, framealpha=0.3)  # prop={'weight': 'bold'}
axes[1].set_facecolor('#f0f0f0')
# ax.set_facecolor('#f0f0f0')
axes[1].grid(True, which='both', linestyle='-', color='white', linewidth=2.0)
# plt.xscale('log')
spine_size = 2.0
for i in range(0, 2):
    for spine in axes[i].spines.values():
        spine.set_linewidth(spine_size)
        spine.set_edgecolor('black')


plt.tight_layout()  # rect=[0.0, -0.05, 1, 0.94]

# plt.show()

fig.savefig("rnn_dimension.png", bbox_inches='tight', pad_inches=0.5)

"""
























