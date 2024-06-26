import torch

import common
import datasets


class InitialValueNetwork(torch.nn.Module):
    def __init__(self, intensity, hidden_channels, model):
        super(InitialValueNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(7 if intensity else 5, 256)
        self.linear2 = torch.nn.Linear(256, hidden_channels)

        self.model = model

    def forward(self, times, coeffs, final_index, **kwargs):
        *coeffs, static = coeffs
        z0 = self.linear1(static)
        z0 = z0.relu()
        z0 = self.linear2(z0)
        return self.model(times, coeffs, final_index, z0=z0, **kwargs)


def main(intensity, device='cuda', max_epochs=50, pos_weight=10, *,
         model_name, train_dataset_name, test_dataset_name, hidden_channels, hidden_hidden_channels, num_hidden_layers, batch_size, num_dim,
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


    times, train_dataloader, test_dataloader = datasets.main.get_data(train_dataset_name, test_dataset_name, static_intensity,
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
    name = train_dataset_name + '_' + model_name + "_7b" + intensity_str


    num_classes = 2

    if model_name == "ncde":
        return common.main(name, times, train_dataloader, test_dataloader, device,
                       new_make_model, num_classes, max_epochs, lr, kwargs, pos_weight=torch.tensor(pos_weight), step_mode=True)
    elif model_name in ["naivesde", "odernn"]:
        return common.main(name, times, train_dataloader, test_dataloader, device,
                             make_model, num_classes, max_epochs, lr, kwargs, pos_weight=torch.tensor(pos_weight), step_mode=True)




def run_all(intensity, device, model_names=('ncde', 'odernn', 'dt', 'decay', 'gruode')):
    model_kwargs = dict(ncde=dict(hidden_channels=49, hidden_hidden_channels=49, num_hidden_layers=4),
                        odernn=dict(hidden_channels=128, hidden_hidden_channels=128, num_hidden_layers=4),
                        dt=dict(hidden_channels=187, hidden_hidden_channels=None, num_hidden_layers=None),
                        decay=dict(hidden_channels=187, hidden_hidden_channels=None, num_hidden_layers=None),
                        gruode=dict(hidden_channels=187, hidden_hidden_channels=None, num_hidden_layers=None))
    for model_name in model_names:
        # Hyperparameters selected as what ODE-RNN did best with.
        for _ in range(5):
            main(intensity, device, model_name=model_name, **model_kwargs[model_name])
