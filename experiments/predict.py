# import uea
# import sepsis
import os
# import ihad
import truthful_qa
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
import common_truthful
import datasets
import pickle


class _SqueezeEnd(torch.nn.Module):
    def __init__(self, model):
        super(_SqueezeEnd, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs).squeeze(-1)



def _evaluate_metrics(dataloader, model, times, loss_fn, device, kwargs):
    with torch.no_grad():
        true_y_cpus = []
        pred_y_cpus = []
        thresholded_ys = []
        ques_ids = []
        for batch in dataloader:
            batch = tuple(b.to(device) for b in batch)
            *coeffs, X, true_y, lengths, question_idxs = batch
            pred_y = model(times, coeffs, lengths, **kwargs)


            thresholded_y = (pred_y >= 0.5).to(true_y.dtype)
            true_y_cpu = true_y.detach().cpu()
            pred_y_cpu = pred_y.detach().cpu()    # 给出的概率值
            true_y_cpus.append(true_y_cpu)
            pred_y_cpus.append(pred_y_cpu)
            thresholded_ys.append(thresholded_y)
            ques_ids.extend(question_idxs.detach().cpu().numpy())


        true_y_cpus = torch.cat(true_y_cpus, dim=0).detach().cpu().numpy()
        thresholded_ys = torch.cat(thresholded_ys, dim=0).detach().cpu().numpy()
        pred_y_cpus = torch.cat(pred_y_cpus, dim=0).detach().cpu().numpy()
        accuracy = sklearn.metrics.accuracy_score(true_y_cpus, thresholded_ys)
        precision = sklearn.metrics.precision_score(true_y_cpus, thresholded_ys)
        recall = sklearn.metrics.recall_score(true_y_cpus, thresholded_ys)
        f1 = sklearn.metrics.f1_score(true_y_cpus, thresholded_ys)
        precision_curve, recall_curve, _ = sklearn.metrics.precision_recall_curve(true_y_cpus, pred_y_cpus)
        pr_auc = sklearn.metrics.auc(recall_curve, precision_curve)

        res = {"question_idxs": question_idxs, "true_y_cpus": true_y_cpus, "thresholded_ys": thresholded_ys, "pred_y_cpus": pred_y_cpus}
        with open('data.pkl', 'wb') as f:
            pickle.dump(res, f)

        # 从文件加载
        with open('data.pkl', 'rb') as f:
            loaded_data = pickle.load(f)

        print('Test accuracy: {:.3} Test precision: {:.3} Test recall: {:.3} Test f1: {:.3} Test auc: {:.3}'
              .format(accuracy, precision, recall, f1, pr_auc))



def main(intensity,                                                               # Whether to include intensity or not
         device='cuda', max_epochs=20, pos_weight=10, *,                         # training parameters
         model_name, hidden_channels, hidden_hidden_channels, num_hidden_layers,  # model parameters
         dry_run=False,
         **kwargs):                                                               # kwargs passed on to cdeint

    batch_size = 32
    loss_fn = torch.nn.BCELoss()
    static_intensity = intensity
    # these models use the intensity for their evolution. They won't explicitly use it as an input unless we include it
    # via the use_intensity parameter, though.
    time_intensity = intensity or (model_name in ('odernn', 'dt', 'decay'))

    times, train_dataloader, val_dataloader, test_dataloader = datasets.truthful_qa.get_data(static_intensity,
                                                                                        time_intensity,
                                                                                        batch_size)

    input_channels = 1 + (1 + time_intensity) * 10
    make_model = common_truthful.make_model(model_name, input_channels, 1, hidden_channels,
                                   hidden_hidden_channels, num_hidden_layers, use_intensity=intensity, initial=True)  # False

    def new_make_model():
        model, regularise = make_model()
        model.linear.weight.register_hook(lambda grad: 100 * grad)
        model.linear.bias.register_hook(lambda grad: 100 * grad)
        # return InitialValueNetwork(intensity, hidden_channels, model), regularise
        return model, regularise

    load_model_sde = torch.load('results/truthful_qa_nointensity/model_state_dict.pth')
    model, regularise = new_make_model()
    model = _SqueezeEnd(model)
    model.load_state_dict(load_model_sde)
    model.to(device)

    times = times.to(device)

    # 切换到评估模式
    model.eval()
    _evaluate_metrics(test_dataloader, model, times, loss_fn, device, kwargs)









if __name__ == "__main__":
    main(intensity=False, device='cuda', model_name='ncde', hidden_channels=256, hidden_hidden_channels=256,
                     num_hidden_layers=4)