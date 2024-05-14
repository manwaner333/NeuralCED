import torch
import os
import common_truthful_qa as common
import datasets
import matplotlib.pyplot as plt
import sklearn.metrics
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from parse import parse_args
import numpy as np
from torch.distributions import Normal

args = parse_args()
device = "cuda"
np.random.seed(0)
torch.manual_seed(0)
# import ipdb


def _evaluate_metrics_forecasting(dataloader, model_cde, model_sde, x_times, times, loss_fn, device, kwargs):
    with torch.no_grad():
        total_dataset_size = 0
        total_loss = 0
        noise_cde_std = 0.002
        noise_sde_std = 0.01
        true_y_cpus = []
        pred_sde_y_cpus = []
        thresholded_sde_ys = []
        pred_cde_y_cpus = []
        thresholded_cde_ys = []
        count = 0
        for batch in dataloader:
            batch = tuple(b.to(device) for b in batch)
            *coeffs, x, y, xy, x_lengths, y_lengths, question_ids, label = batch
            batch_size = y.size(0)

            real_y = xy
            pred_y_cde = model_cde(x_times, times, coeffs, x_lengths + y_lengths - 1, **kwargs)
            pred_y_sde = model_sde(x_times, times, coeffs, **kwargs)
            x_length = x_lengths[0]
            y_length = y_lengths[0]
            real_y_filter = real_y[0, 0:x_length + y_length, :]
            pred_y_cde_filter = pred_y_cde[0, 0:x_length + y_length, :]
            pred_y_sde_filter = pred_y_sde[0, 0:x_length + y_length, :]


            scale_sde_tor = torch.full_like(pred_y_sde_filter, noise_sde_std)
            xs_sde_dist = Normal(loc=pred_y_sde_filter, scale=scale_sde_tor)
            log_sde_pxs = xs_sde_dist.log_prob(real_y_filter).mean()
            # print(log_sde_pxs, label)

            scale_cde_tor = torch.full_like(pred_y_cde_filter, noise_cde_std)
            xs_cde_dist = Normal(loc=pred_y_cde_filter, scale=scale_cde_tor)
            log_cde_pxs = xs_cde_dist.log_prob(real_y_filter).mean()
            # print(log_cde_pxs, label)
            # ipdb.set_trace()

            true_y = label
            true_y_cpus.append(true_y.detach().cpu())

            # sde
            pred_sde_y = (log_sde_pxs >= (-50.0)).to(true_y.dtype).unsqueeze(0)
            pred_sde_y_pro = log_sde_pxs.unsqueeze(0)
            pred_sde_y_cpus.append(pred_sde_y.detach().cpu())
            thresholded_sde_ys.append(pred_sde_y_pro.detach().cpu())

            pred_cde_y = (log_cde_pxs >= (-50.0)).to(true_y.dtype).unsqueeze(0)
            pred_cde_y_pro = log_cde_pxs.unsqueeze(0)
            pred_cde_y_cpus.append(pred_cde_y.detach().cpu())
            thresholded_cde_ys.append(pred_cde_y_pro.detach().cpu())



            count += 1
            if count > 5:
                break

        true_y_cpus = torch.cat(true_y_cpus, dim=0).detach().cpu().numpy()

        # cde
        thresholded_cde_ys = torch.cat(thresholded_cde_ys, dim=0).detach().cpu().numpy()
        pred_cde_y_cpus = torch.cat(pred_cde_y_cpus, dim=0).detach().cpu().numpy()
        accuracy_cde = sklearn.metrics.accuracy_score(true_y_cpus, pred_cde_y_cpus)
        precision_cde = sklearn.metrics.precision_score(true_y_cpus, pred_cde_y_cpus)
        recall_cde = sklearn.metrics.recall_score(true_y_cpus, pred_cde_y_cpus)
        f1_cde = sklearn.metrics.f1_score(true_y_cpus, pred_cde_y_cpus)
        precision_curve_cde, recall_curve_cde, _ = sklearn.metrics.precision_recall_curve(true_y_cpus, thresholded_cde_ys)
        pr_auc_cde = sklearn.metrics.auc(recall_curve_cde, precision_curve_cde)

        # sde
        thresholded_sde_ys = torch.cat(thresholded_sde_ys, dim=0).detach().cpu().numpy()
        pred_sde_y_cpus = torch.cat(pred_sde_y_cpus, dim=0).detach().cpu().numpy()
        accuracy_sde = sklearn.metrics.accuracy_score(true_y_cpus, pred_sde_y_cpus)
        precision_sde = sklearn.metrics.precision_score(true_y_cpus, pred_sde_y_cpus)
        recall_sde = sklearn.metrics.recall_score(true_y_cpus, pred_sde_y_cpus)
        f1_sde = sklearn.metrics.f1_score(true_y_cpus, pred_sde_y_cpus)
        precision_curve_sde, recall_curve_sde, _ = sklearn.metrics.precision_recall_curve(true_y_cpus, thresholded_sde_ys)
        pr_auc_sde = sklearn.metrics.auc(recall_curve_sde, precision_curve_sde)

        print('cde accuracy: {:.3},  cde precision: {:.3}, cde recall: {:.3} cde f1: {:.3}, cde auc: {:.3}, '
        'sde accuracy: {:.3},  sde precision: {:.3}, sde recall: {:.3} sde f1: {:.3}, sde auc: {:.3}'
        ''.format(accuracy_cde, precision_cde, recall_cde, f1_cde, pr_auc_cde, accuracy_sde, precision_sde,
                  recall_sde, f1_sde, pr_auc_sde))


        # ipdb.set_trace()

        return total_loss


def main(
        manual_seed=args.seed,
        intensity=args.intensity,
        device="cuda",
        max_epochs=args.epoch,
        missing_rate=args.missing_rate,
        pos_weight=10,
        *,
        model_name=args.model,
        hidden_channels=args.h_channels,
        hidden_hidden_channels=args.hh_channels,
        num_hidden_layers=args.layers,
        ode_hidden_hidden_channels=args.ode_hidden_hidden_channels,
        dry_run=False,
        method=args.method,
        step_mode=args.step_mode,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss=args.loss,
        reg=args.reg,
        scale=args.scale,
        time_seq=args.time_seq,
        y_seq=args.y_seq,
        **kwargs
):
    # cde
    input_channels = 20
    make_model_cde = common.make_model('ncde', input_channels, input_channels, hidden_channels,
                                       hidden_hidden_channels, ode_hidden_hidden_channels, num_hidden_layers,
                                       use_intensity=intensity, initial=True)
    model_cde, regularise_cde = make_model_cde()
    # 加载之前保存的状态字典
    load_model_cde = torch.load('results/MuJoCo_0.0/model_state_dict_cde.pth')
    model_cde.load_state_dict(load_model_cde)
    model_cde.to(device)

    # 切换到评估模式
    model_cde.eval()

    # sde
    make_model_sde = common.make_model('naivesde', input_channels, input_channels, hidden_channels,
                                       hidden_hidden_channels, ode_hidden_hidden_channels, num_hidden_layers,
                                       use_intensity=intensity, initial=True)
    model_sde, regularise_sde = make_model_sde()
    # 加载之前保存的状态字典
    load_model_sde = torch.load('results/MuJoCo_0.0/model_state_dict_sde.pth')
    del load_model_sde['func.X._t']
    del load_model_sde['func.X._a']
    del load_model_sde['func.X._b']
    del load_model_sde['func.X._two_c']
    del load_model_sde['func.X._three_d']
    model_sde.load_state_dict(load_model_sde)
    model_sde.to(device)

    # 切换到评估模式
    model_sde.eval()

    batch_size = 1
    time_augment = False
    loss_fn = torch.nn.functional.mse_loss
    x_times, times, train_dataloader, val_dataloader, test_dataloader = datasets.truthful_qa.get_data(batch_size,
                                                                                                      missing_rate,
                                                                                                      time_augment,
                                                                                                      time_seq, y_seq)

    x_times = x_times.to(device)
    times = times.to(device)

    _evaluate_metrics_forecasting(test_dataloader, model_cde, model_sde, x_times, times, loss_fn, device, kwargs)


if __name__ == "__main__":
    # main(method=args.method)
    main(hidden_channels=16, hidden_hidden_channels=16, num_hidden_layers=4, lr=0.001, method="euler",
         missing_rate=0.0, time_seq=50, y_seq=10, intensity=False, max_epochs=20, step_mode='valloss',
         model_name="ncde")