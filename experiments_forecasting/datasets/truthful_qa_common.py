import os
import pathlib
import sklearn.model_selection
import sys
import torch

here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(here / '..' / '..'))

import controldiffeq

def dataloader(dataset, **kwargs):
    if 'shuffle' not in kwargs:
        kwargs['shuffle'] = True
    if 'drop_last' not in kwargs:
        kwargs['drop_last'] = True
    if 'batch_size' not in kwargs:
        kwargs['batch_size'] = 32
    if 'num_workers' not in kwargs:
        kwargs['num_workers'] = 8
    kwargs['batch_size'] = min(kwargs['batch_size'], len(dataset))
    return torch.utils.data.DataLoader(dataset, **kwargs)


def split_data(tensor, stratify):
    
    (train_tensor, testval_tensor,
     train_stratify, testval_stratify) = sklearn.model_selection.train_test_split(tensor, stratify,
                                                                                  train_size=0.7,
                                                                                  random_state=0,
                                                                                  shuffle=True,
                                                                                  stratify=stratify)

    val_tensor, test_tensor = sklearn.model_selection.train_test_split(testval_tensor,
                                                                       train_size=0.5,
                                                                       random_state=1,
                                                                       shuffle=True,
                                                                       stratify=testval_stratify)
    return train_tensor, val_tensor, test_tensor


def normalise_data(X, y):
    train_X, _, _ = split_data(X, y)
    out = []
    for Xi, train_Xi in zip(X.unbind(dim=-1), train_X.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()  
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    out = torch.stack(out, dim=-1)
    return out


def preprocess_data_forecasting(times, X, y, final_index, append_times):
    
    
    X = X.cuda()
    
    full_len = X.shape[0]
    train_len = int(full_len*0.7) 
    val_len = int(full_len*0.85)
    augmented_X = []
    if append_times:
        augmented_X.append(times.unsqueeze(0).repeat(X.size(0), 1).unsqueeze(-1))

    augmented_X.append(X)
    augmented_X[0] = augmented_X[0].cuda()
    if len(augmented_X) == 1:
        X = augmented_X[0]
    else:
        X = torch.cat(augmented_X, dim=2)
    if append_times:
        print(f"time augment : {X.shape}")
    train_X, train_y = X[:train_len], y[:train_len]
    val_X, val_y = X[train_len:val_len],y[train_len:val_len]
    test_X, test_y = X[val_len:], y[val_len:]
    
    train_final_index, val_final_index, test_final_index = final_index[:train_len],final_index[train_len:val_len],final_index[val_len:]
    
    times = torch.linspace(0, X.size(1) - 1, X.size(1))
    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times.cuda(), train_X)
    val_coeffs = controldiffeq.natural_cubic_spline_coeffs(times.cuda(), val_X)
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times.cuda(), test_X)
    
    in_channels = X.size(-1)

    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index, in_channels)


def preprocess_data(x_times, times, X, y, xy, x_final_index, y_final_index, question_ids, labels, append_times, append_intensity):
    # X = normalise_data(X, labels)
    # y = normalise_data(y, labels)
    # xy = normalise_data(xy, labels)

    # Append extra channels together. Note that the order here: time, intensity, original, is important, and some models
    # depend on that order.
    augmented_X = []
    if append_times:
        augmented_X.append(x_times.unsqueeze(0).repeat(X.size(0), 1).unsqueeze(-1))
    if append_intensity:
        intensity = ~torch.isnan(X)  # of size (batch, stream, channels)
        intensity = intensity.to(X.dtype).cumsum(dim=1)
        augmented_X.append(intensity)
    augmented_X.append(X)
    if len(augmented_X) == 1:
        X = augmented_X[0]
    else:
        X = torch.cat(augmented_X, dim=2)

    # coeffs = controldiffeq.natural_cubic_spline_coeffs(x_times, X)
    coeffs = controldiffeq.natural_cubic_spline_coeffs(times, xy)
    in_channels = X.size(-1)

    return (x_times, times, coeffs, X, y, xy, x_final_index, y_final_index, question_ids, labels, in_channels)


# def preprocess_data(times, X, y, final_index, append_times, append_intensity):
#     X = normalise_data(X, y)
#
#     # Append extra channels together. Note that the order here: time, intensity, original, is important, and some models
#     # depend on that order.
#     augmented_X = []
#     if append_times:
#         augmented_X.append(times.unsqueeze(0).repeat(X.size(0), 1).unsqueeze(-1))
#     if append_intensity:
#         intensity = ~torch.isnan(X)
#         intensity = intensity.to(X.dtype).cumsum(dim=1)
#         augmented_X.append(intensity)
#     augmented_X.append(X)
#     if len(augmented_X) == 1:
#         X = augmented_X[0]
#     else:
#         X = torch.cat(augmented_X, dim=2)
#
#     train_X, val_X, test_X = split_data(X, y)
#     train_y, val_y, test_y = split_data(y, y)
#     train_final_index, val_final_index, test_final_index = split_data(final_index, y)
#
#     train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, train_X)
#     val_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, val_X)
#     test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, test_X)
#
#     in_channels = X.size(-1)
#
#     return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
#             test_final_index, in_channels)



def wrap_data(x_times, times, train_coeffs, val_coeffs, test_coeffs, train_X, val_X, test_X, train_y, val_y, test_y, train_xy, val_xy, test_xy, train_x_lengths, val_x_lengths,
              test_x_lengths, train_y_lengths, val_y_lengths, test_y_lengths, train_question_ids, val_question_ids, test_question_ids, train_labels, val_labels, test_labels,
              device, batch_size, num_workers=0):
    x_times = x_times.to(device)
    times = times.to(device)
    train_coeffs = tuple(coeff.to(device) for coeff in train_coeffs)
    val_coeffs = tuple(coeff.to(device) for coeff in val_coeffs)
    test_coeffs = tuple(coeff.to(device) for coeff in test_coeffs)

    train_y = train_y.to(device)
    val_y = val_y.to(device)
    test_y = test_y.to(device)
    train_xy = train_xy.to(device)
    val_xy = val_xy.to(device)
    test_xy = test_xy.to(device)
    train_x_lengths = train_x_lengths.to(device)
    val_x_lengths = val_x_lengths.to(device)
    test_x_lengths = test_x_lengths.to(device)
    train_y_lengths = train_y_lengths.to(device)
    val_y_lengths = val_y_lengths.to(device)
    test_y_lengths = test_y_lengths.to(device)
    train_question_ids =train_question_ids.to(device)
    val_question_ids = val_question_ids.to(device)
    test_question_ids = test_question_ids.to(device)
    train_dataset = torch.utils.data.TensorDataset(*train_coeffs, train_X, train_y, train_xy, train_x_lengths, train_y_lengths, train_question_ids, train_labels)
    val_dataset = torch.utils.data.TensorDataset(*val_coeffs, val_X, val_y, val_xy, val_x_lengths, val_y_lengths, val_question_ids, val_labels)
    test_dataset = torch.utils.data.TensorDataset(*test_coeffs, test_X, test_y, test_xy, test_x_lengths, test_y_lengths, test_question_ids, test_labels)

    train_dataloader = dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = dataloader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    
    return x_times, times, train_dataloader, val_dataloader, test_dataloader


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')


def load_data(dir):
    
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors