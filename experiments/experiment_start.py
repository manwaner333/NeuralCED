import os
import main
import numpy as np
import torch
np.random.seed(42)
torch.manual_seed(42)


# truthful_qa.main(intensity=False, device='cuda', model_name='ncde', hidden_channels=256, hidden_hidden_channels=256, num_hidden_layers=4)

main.main(intensity=False, device='cuda', max_epochs=4, model_name='naivesde', dataset_name='company', hidden_channels=256, hidden_hidden_channels=256, num_hidden_layers=4, batch_size=32, num_dim=10)
# model: ncde  naivesde