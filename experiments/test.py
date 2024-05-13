# import uea
import sepsis
import os
import ihad
import truthful_qa
import numpy as np
import torch
np.random.seed(42)
torch.manual_seed(42)
# import speech_commands

# # group = 1 corresponds to 30% mising rate
# uea.run_all(group=1, device='cuda', dataset_name='CharacterTrajectories')
# # group = 2 corresponds to 50% mising rate
# uea.run_all(group=2, device='cuda', dataset_name='CharacterTrajectories')
# # group = 3 corresponds to 70% mising rate
# uea.run_all(group=3, device='cuda', dataset_name='CharacterTrajectories')

# sepsis.run_all(intensity=True, device='cuda')
# sepsis.run_all(intensity=False, device='cuda')

# speech_commands.run_all(device='cuda')


# sepsis.main(intensity=True, device='cuda', model_name='ncde', hidden_channels=49, hidden_hidden_channels=49, num_hidden_layers=4)
# ihad.main(intensity=True, device='cuda', model_name='ncde', hidden_channels=15, hidden_hidden_channels=15, num_hidden_layers=4)
# ihad.main(intensity=False, device='cuda', model_name='ncde', hidden_channels=15, hidden_hidden_channels=15, num_hidden_layers=4)
truthful_qa.main(intensity=False, device='cuda', model_name='ncde', hidden_channels=15, hidden_hidden_channels=15, num_hidden_layers=4)