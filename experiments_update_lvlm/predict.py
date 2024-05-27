# import uea
# import sepsis
import os
# import ihad
import numpy as np
import torch
import copy
import json
import math
import numpy as np
import os
import pathlib
import sklearn.metrics
import torch
import tqdm

import models
np.random.seed(42)
torch.manual_seed(42)
import torch
import common
import datasets
import argparse
import pickle


class _SqueezeEnd(torch.nn.Module):
    def __init__(self, model):
        super(_SqueezeEnd, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs).squeeze(-1)



def _evaluate_metrics(train_dataloader, val_dataloader, test_dataloader, model, times, device, need_train, need_val, need_test, kwargs):
    with torch.no_grad():
        true_y_cpus = []
        pred_y_cpus = []
        thresholded_ys = []
        ques_ids = []

        if need_train:
            for batch in train_dataloader:
                batch = tuple(b.to(device) for b in batch)
                *coeffs, X, true_y, lengths, question_idxs = batch
                pred_y = model(times, coeffs, lengths, **kwargs)


                thresholded_y = (pred_y >= 0.5).to(true_y.dtype).detach().cpu().tolist()
                true_y_cpu = true_y.detach().cpu().tolist()
                pred_y_cpu = pred_y.detach().cpu().tolist()    # 给出的概率值
                true_y_cpus.extend(true_y_cpu)
                pred_y_cpus.extend(pred_y_cpu)
                thresholded_ys.extend(thresholded_y)
                ques_ids.extend(question_idxs.detach().cpu().numpy())

        if need_val:
            for batch in val_dataloader:
                batch = tuple(b.to(device) for b in batch)
                *coeffs, X, true_y, lengths, question_idxs = batch
                pred_y = model(times, coeffs, lengths, **kwargs)

                thresholded_y = (pred_y >= 0.5).to(true_y.dtype).detach().cpu().tolist()
                true_y_cpu = true_y.detach().cpu().tolist()
                pred_y_cpu = pred_y.detach().cpu().tolist()  # 给出的概率值
                true_y_cpus.extend(true_y_cpu)
                pred_y_cpus.extend(pred_y_cpu)
                thresholded_ys.extend(thresholded_y)
                ques_ids.extend(question_idxs.detach().cpu().numpy())

        if need_test:
            for batch in test_dataloader:
                batch = tuple(b.to(device) for b in batch)
                *coeffs, X, true_y, lengths, question_idxs = batch
                pred_y = model(times, coeffs, lengths, **kwargs)

                thresholded_y = (pred_y >= 0.5).to(true_y.dtype).detach().cpu().tolist()
                true_y_cpu = true_y.detach().cpu().tolist()
                pred_y_cpu = pred_y.detach().cpu().tolist()  # 给出的概率值
                true_y_cpus.extend(true_y_cpu)
                pred_y_cpus.extend(pred_y_cpu)
                thresholded_ys.extend(thresholded_y)
                ques_ids.extend(question_idxs.detach().cpu().numpy())


        accuracy = sklearn.metrics.accuracy_score(true_y_cpus, thresholded_ys)
        precision = sklearn.metrics.precision_score(true_y_cpus, thresholded_ys)
        recall = sklearn.metrics.recall_score(true_y_cpus, thresholded_ys)
        f1 = sklearn.metrics.f1_score(true_y_cpus, thresholded_ys)
        precision_curve, recall_curve, _ = sklearn.metrics.precision_recall_curve(true_y_cpus, pred_y_cpus)
        pr_auc = sklearn.metrics.auc(recall_curve, precision_curve)

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_y_cpus, pred_y_cpus)
        auc_roc = sklearn.metrics.auc(fpr, tpr)

        # res = {"question_idxs": ques_ids, "true_y_cpus": true_y_cpus, "thresholded_ys": thresholded_ys, "pred_y_cpus": pred_y_cpus}
        # with open('results/data.pkl', 'wb') as f:
        #     pickle.dump(res, f)

        print('Test accuracy: {:.3} Test precision: {:.3} Test recall: {:.3} Test f1: {:.3} Test auc: {:.3} Test auc-roc: {:.3}'
              .format(accuracy, precision, recall, f1, pr_auc, auc_roc))





def main(intensity, device='cuda', max_epochs=50, pos_weight=10, *,
         model_name, model_path, dataset_name, hidden_channels, hidden_hidden_channels, num_hidden_layers, batch_size, num_dim,
         **kwargs):

    if intensity == "False":
        intensity = False
    else:
        intensity = True

    batch_size = batch_size  # 1024
    lr = 0.0001 * (batch_size / 32)
    num_dim = num_dim

    static_intensity = intensity
    # these models use the intensity for their evolution. They won't explicitly use it as an input unless we include it
    # via the use_intensity parameter, though.
    time_intensity = intensity or (model_name in ('odernn', 'dt', 'decay'))


    times, train_dataloader, val_dataloader, test_dataloader = datasets.main.get_data(dataset_name, static_intensity,
                                                                                    time_intensity,
                                                                                    batch_size, num_dim)

    input_channels = 1 + (1 + time_intensity) * num_dim
    make_model = common.make_model(model_name, input_channels, 1, hidden_channels,
                                   hidden_hidden_channels, num_hidden_layers, use_intensity=intensity, initial=True)  # False

    def new_make_model():
        model, regularise = make_model()
        model.linear.weight.register_hook(lambda grad: 100 * grad)
        model.linear.bias.register_hook(lambda grad: 100 * grad)
        # return InitialValueNetwork(intensity, hidden_channels, model), regularise
        return model, regularise


    intensity_str = '_intensity' if intensity else '_nointensity'
    name = dataset_name + '_' + model_name + "_7b" + intensity_str


    if model_name == "ncde":
        model, regularise_parameters = new_make_model()
    elif model_name in ["naivesde", "odernn"]:
        model, regularise_parameters = make_model()


    load_model_sde = torch.load(model_path)
    model = _SqueezeEnd(model)
    model.load_state_dict(load_model_sde)
    model.to(device)

    times = times.to(device)

    # 切换到评估模式
    model.eval()
    need_train = True
    need_val = True
    need_test = True
    _evaluate_metrics(train_dataloader, val_dataloader, test_dataloader, model, times, device, need_train, need_val, need_test, kwargs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralED4Hal")
    parser.add_argument("--intensity", type=str, default="False")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--model_name", type=str, default="ncde")  # ncde, odernn
    parser.add_argument("--model_path", type=str, default='results/fact/model_state_dict.pth')
    parser.add_argument("--dataset_name", type=str, default="fact")
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--hidden_hidden_channels", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_dim", type=int, default=10)
    args = parser.parse_args()

    intensity = args.intensity
    device = args.device
    max_epochs = args.max_epochs
    model_name = args.model_name
    model_path = args.model_path
    dataset_name = args.dataset_name
    hidden_channels = args.hidden_channels
    hidden_hidden_channels = args.hidden_hidden_channels
    num_hidden_layers = args.num_hidden_layers
    batch_size = args.batch_size
    num_dim = args.num_dim

    main(intensity=intensity, device=device, model_name=model_name, model_path=model_path,
              dataset_name=dataset_name, hidden_channels=hidden_channels, hidden_hidden_channels=hidden_hidden_channels,
              num_hidden_layers=num_hidden_layers, batch_size=batch_size, num_dim=num_dim)



    # save data
    # with open('results/data.pkl', 'rb') as f:
    #     loaded_data = pickle.load(f)
    #
    #
    # question_idxs = loaded_data['question_idxs']
    # true_y_cpus = loaded_data['true_y_cpus']
    # thresholded_ys = loaded_data['thresholded_ys']
    # pred_y_cpus = loaded_data['pred_y_cpus']
    # for i in range(len(question_idxs)):
    #     if (true_y_cpus[i] != thresholded_ys[i]):
    #         print(question_idxs[i], true_y_cpus[i], thresholded_ys[i], pred_y_cpus[i])
