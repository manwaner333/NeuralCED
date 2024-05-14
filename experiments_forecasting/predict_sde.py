import torch
import os
import common_truthful_qa as common
import datasets
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from parse import parse_args
import numpy as np
args = parse_args()
device ="cuda"
np.random.seed(0)
torch.manual_seed(0)


def _evaluate_metrics_forecasting(dataloader, model, x_times, times, loss_fn, device, kwargs):
    with torch.no_grad():
        total_dataset_size = 0
        total_loss = 0
        for batch in dataloader:
            batch = tuple(b.to(device) for b in batch)
            *coeffs, x, y, xy, x_lengths, y_lengths, question_ids, label = batch
            batch_size = y.size(0)

            real_y = xy
            pred_y = model(x_times, times, coeffs, **kwargs)
            x_length = x_lengths[0]
            y_length = y_lengths[0]
            real_y_filter = real_y[0, 0:x_length + y_length, :]
            pred_y_filter = pred_y[0, 0:x_length + y_length, :]
            real_y_filter_np = real_y_filter.tolist()
            pred_y_filter_np = pred_y_filter.tolist()

            x = range(31)
            index = 1
            real_y_1 = np.array(real_y_filter_np)[:, index]
            pred_y_1 = np.array(pred_y_filter_np)[:, index]

            plt.plot(x, real_y_1, label='real', color='blue', marker='o')  # 第一条折线
            plt.plot(x, pred_y_1, label='pred', color='red', marker='s')  # 第二条折线
            plt.legend()
            plt.show()

            qingli = 3



        #     mask_pred = torch.zeros(pred_y.shape[0], pred_y.shape[1], pred_y.shape[2], dtype=torch.bool).to(device)
        #     for i in range(len(x_lengths)):
        #         x_length = x_lengths[i]
        #         y_length = y_lengths[i]
        #         mask_pred[i, x_length:x_length + y_length, :] = True
        #
        #     mask_real = torch.zeros(real_y.shape[0], real_y.shape[1], real_y.shape[2], dtype=torch.bool).to(device)
        #     for i in range(len(y_lengths)):
        #         y_length = y_lengths[i]
        #         mask_real[i, 0:y_length, :] = True
        #
        #     total_dataset_size += batch_size
        #     mask_pred_y = torch.masked_select(pred_y, mask_pred)
        #     mask_real_y = torch.masked_select(real_y, mask_real)
        #     loss = loss_fn(mask_pred_y, mask_real_y)
        #
        #     total_loss += loss * batch_size
        #
        # total_loss /= total_dataset_size

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
    input_channels = 20
    make_model = common.make_model(model_name, input_channels, input_channels, hidden_channels,
                                   hidden_hidden_channels, ode_hidden_hidden_channels, num_hidden_layers,
                                   use_intensity=intensity, initial=True)
    model, regularise = make_model()
    # 加载之前保存的状态字典
    load_model = torch.load('results/MuJoCo_0.0/model_state_dict_sde_1.pth')
    del load_model['func.X._t']
    del load_model['func.X._a']
    del load_model['func.X._b']
    del load_model['func.X._two_c']
    del load_model['func.X._three_d']
    model.load_state_dict(load_model)
    model.to(device)

    # 切换到评估模式
    model.eval()
    batch_size = 1
    time_augment = False
    loss_fn = torch.nn.functional.mse_loss
    x_times, times, train_dataloader, val_dataloader, test_dataloader = datasets.truthful_qa.get_data(batch_size,
                                                                                                      missing_rate,
                                                                                                      time_augment,
                                                                                                      time_seq, y_seq)

    x_times = x_times.to(device)
    times = times.to(device)
    # train_dataloader.to(device)
    # val_dataloader.to(device)
    # test_dataloader.to(device)

    _evaluate_metrics_forecasting(test_dataloader, model, x_times, times, loss_fn, device, kwargs)









if __name__ == "__main__":
    # main(method=args.method)
    main(hidden_channels=16, hidden_hidden_channels=16, num_hidden_layers=2, lr=0.001, method="euler",
         missing_rate=0.0, time_seq=50, y_seq=10, intensity=False, max_epochs=20, step_mode='valloss',
         model_name="naivesde")