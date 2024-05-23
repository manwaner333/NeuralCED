import json
import pickle
import os
import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import datasets
import matplotlib.pyplot as plt
device ="cuda"



if __name__ == "__main__":
    batch_size = 1  # 1024
    static_intensity = False
    time_intensity = False

    times, train_dataloader, val_dataloader, test_dataloader = datasets.truthful_qa.get_data(static_intensity,
                                                                                             time_intensity,
                                                                                             batch_size)

    index = 2
    res = []
    for batch in train_dataloader:
        batch = tuple(b.to(device) for b in batch)
        *train_coeffs, X, train_y, lengths, question_idxs = batch
        if question_idxs[0] not in [73, 74]:  # [71, 72, 73, 74, 75, 76]:
            continue
        print(question_idxs)
        length = lengths[0]
        real_x_filter_np = X[0, 0:length + 1, index]
        real_y_1 = np.array(real_x_filter_np.detach().cpu().numpy())
        res.append(real_y_1)
        print(len(real_y_1))

    for batch in val_dataloader:
        batch = tuple(b.to(device) for b in batch)
        *train_coeffs, X, train_y, lengths, question_idxs = batch
        if question_idxs[0] not in [73, 74]:  # [71, 72, 73, 74, 75, 76]:
            continue
        print(question_idxs)
        length = lengths[0]
        real_x_filter_np = X[0, 0:length + 1, index]
        real_y_1 = np.array(real_x_filter_np.detach().cpu().numpy())
        res.append(real_y_1)
        print(len(real_y_1))

    for batch in test_dataloader:
        batch = tuple(b.to(device) for b in batch)
        *train_coeffs, X, train_y, lengths, question_idxs = batch
        if question_idxs[0] not in [73, 74]:  # [71, 72, 73, 74, 75, 76]:
            continue
        print(question_idxs)
        length = lengths[0]
        real_x_filter_np = X[0, 0:length + 1, index]
        real_y_1 = np.array(real_x_filter_np.detach().cpu().numpy())
        res.append(real_y_1)
        print(len(real_y_1))

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    x = [x for x in range(29)]
    ticks_size = 17
    bold = None
    label_size = 20
    title_size = 20
    spine_size = 2.5
    linewidth = 5.0
    ax1.plot(x, res[0], marker='*', markersize=10, label='Ex.1', linestyle="-", linewidth=linewidth)  # 第一条折线
    ax1.plot(x, res[1], marker='o', markersize=10, label='Ex.2', linestyle="-.", linewidth=linewidth)  # 第一条折线
    x_labels = ['Q', ':', 'What', 'U', '.', 'S', '.', 'state', 'produces', 'the', 'most', 'pe', 'aches', '?', 'A', ':', 'South(New)', 'Carolina(Jersey)', 'produces', 'the', 'most', 'pe', 'aches', 'in', 'the', 'U', '.', 'S', '.']
    ax1.set_xticks(x[::len(x) // len(x_labels)])  # Set x-ticks to be evenly spaced
    ax1.set_xticklabels(x_labels, fontsize=ticks_size, fontweight=bold,
                        rotation=90)  # Set x-ticks labels to the strings
    for y_tick in ax1.get_yticklabels():
        y_tick.set_fontsize(ticks_size)
        y_tick.set_fontweight(bold)
    for spine in ax1.spines.values():
        spine.set_linewidth(spine_size)
        spine.set_edgecolor('black')
    plt.tight_layout()
    plt.legend()
    plt.show()


    """
    # similar hidden states case                                                                                        
    res = []
    for batch in train_dataloader:
        batch = tuple(b.to(device) for b in batch)
        *train_coeffs, X, train_y, lengths, question_idxs = batch
        if question_idxs[0] not in [0, 3]:   #  [71, 72, 73, 74, 75, 76]:
            continue
        print(question_idxs)
        index = 1
        length = lengths[0]
        real_x_filter_np = X[0, 0:length+1, index]
        real_y_1 = np.array(real_x_filter_np.detach().cpu().numpy())
        res.append(real_y_1)
        print(len(real_y_1))

    for batch in val_dataloader:
        batch = tuple(b.to(device) for b in batch)
        *train_coeffs, X, train_y, lengths, question_idxs = batch
        if question_idxs[0] not in [0,3]:   #  [71, 72, 73, 74, 75, 76]:
            continue
        print(question_idxs)
        index = 1
        length = lengths[0]
        real_x_filter_np = X[0, 0:length+1, index]
        real_y_1 = np.array(real_x_filter_np.detach().cpu().numpy())
        res.append(real_y_1)
        print(len(real_y_1))

    for batch in test_dataloader:
        batch = tuple(b.to(device) for b in batch)
        *train_coeffs, X, train_y, lengths, question_idxs = batch
        if question_idxs[0] not in [0, 3]:   #  [71, 72, 73, 74, 75, 76]:
            continue
        print(question_idxs)
        index = 1
        length = lengths[0]
        real_x_filter_np = X[0, 0:length+1, index]
        real_y_1 = np.array(real_x_filter_np.detach().cpu().numpy())
        res.append(real_y_1)
        print(len(real_y_1))



    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    x = [x for x in range(29)]
    ticks_size = 17
    bold = None
    label_size = 20
    title_size = 20
    spine_size = 2.5
    linewidth = 5.0
    ax1.plot(x, res[0], marker='*', markersize=10, label='Ex.1', linestyle="-", linewidth=linewidth)  # 第一条折线
    ax1.plot(x, res[1], marker='o', markersize=10, label='Ex.2', linestyle="-.", linewidth=linewidth)  # 第一条折线
    x_labels = ['Q', ':', 'Who', 'is', 'the', 'author', 'of', 'the', 'Harry', 'Pot', 'ter', 'series', '?', 'A', ':', 'J(George)', '.(R)', 'K(.)', '.(R)', 'Row(.)', 'ling(Martin)', 'author', 'ed', 'the', 'Harry', 'Pot', 'ter', 'series', '.']
    ax1.set_xticks(x[::len(x)//len(x_labels)])  # Set x-ticks to be evenly spaced
    ax1.set_xticklabels(x_labels, fontsize=ticks_size, fontweight=bold, rotation=90)  # Set x-ticks labels to the strings
    for y_tick in ax1.get_yticklabels():
        y_tick.set_fontsize(ticks_size)
        y_tick.set_fontweight(bold)
    for spine in ax1.spines.values():
        spine.set_linewidth(spine_size)
        spine.set_edgecolor('black')
    plt.tight_layout()
    plt.legend()
    # plt.show()
    save_path = "similar_hidden_states.png"
    plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight', pad_inches=0.1, )
    """
