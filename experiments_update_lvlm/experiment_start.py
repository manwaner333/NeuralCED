import os
import main
import numpy as np
import torch
import argparse
np.random.seed(42)
torch.manual_seed(42)


# truthful_qa.main(intensity=False, device='cuda', model_name='ncde', hidden_channels=256, hidden_hidden_channels=256, num_hidden_layers=4)

# main.main(intensity=False, device='cuda', max_epochs=4, model_name='naivesde', dataset_name='company', hidden_channels=256, hidden_hidden_channels=256, num_hidden_layers=4, batch_size=32, num_dim=10)
# model: ncde  naivesde


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralED4Hal")
    parser.add_argument("--intensity", type=str, default="False")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--model_name", type=str, default="ncde")  # ncde, odernn
    parser.add_argument("--train_dataset_name", type=str, default="m_hal_train")
    parser.add_argument("--test_dataset_name", type=str, default="m_hal_test")
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
    train_dataset_name = args.train_dataset_name
    test_dataset_name = args.test_dataset_name
    hidden_channels = args.hidden_channels
    hidden_hidden_channels = args.hidden_hidden_channels
    num_hidden_layers = args.num_hidden_layers
    batch_size = args.batch_size
    num_dim = args.num_dim

    main.main(intensity=intensity, device=device, max_epochs=max_epochs, model_name=model_name,
              train_dataset_name=train_dataset_name, test_dataset_name=test_dataset_name, hidden_channels=hidden_channels, hidden_hidden_channels=hidden_hidden_channels,
              num_hidden_layers=num_hidden_layers, batch_size=batch_size, num_dim=num_dim)
