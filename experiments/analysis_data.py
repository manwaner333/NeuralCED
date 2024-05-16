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
    filter_file = 'datasets/processed_data/truthful_qa_nostaticintensity_notimeintensity/train_X.pt'
    keys_val = torch.load(filter_file).numpy()

    batch_size = 1  # 1024
    static_intensity = False
    time_intensity = False

    times, train_dataloader, val_dataloader, test_dataloader = datasets.truthful_qa.get_data(static_intensity,
                                                                                             time_intensity,
                                                                                             batch_size)
    res = []
    for batch in train_dataloader:
        batch = tuple(b.to(device) for b in batch)
        *train_coeffs, X, train_y, lengths, question_idxs = batch
        if question_idxs[0] not in [71, 72]:   #  [71, 72, 73, 74, 75, 76]:
            continue
        print(question_idxs)
        index = 1
        length = lengths[0]
        real_x_filter_np = X[0, 0:length+1, index]
        real_y_1 = np.array(real_x_filter_np.detach().cpu().numpy())
        res.append(real_y_1)
        print(len(real_y_1))


    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for real_y_1 in res:
        x = range(len(real_y_1))
        ax.plot(x, real_y_1, label='real', color='blue', marker='o')  # 第一条折线
    plt.show()


    qingli = 3
        # batch_size = y.size(0)
        # if question_ids[0] not in [254]:
        #     # print(question_ids[0])
        #     continue