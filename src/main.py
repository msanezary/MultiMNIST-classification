from train import train_model
from models import MultiTaskCNN
from dataset import get_dataloaders
from utils import set_seed, save_args
import torch.optim as optim
import torch.nn as nn

def main():
    # Settings
    config = {
        'data_path': '../data/multi_mnist.pickle',
        'epochs': 10,
        'batch_size': 256,
        'learning_rate': 0.001,
        'seed': 42,
        'output_dir': '../outputs'
    }

    # Set seed for reproducibility
    set_seed(config['seed'])

    # Get data loaders
    train_loader, test_loader = get_dataloaders(config['data_path'], config['batch_size'])

    # Initialize the model
    model = MultiTaskCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Train and evaluate the model
    train_model(model, criterion, optimizer, train_loader, test_loader, config['epochs'])

    # Save configuration and results
    save_args(config['output_dir'], config)

if __name__ == '__main__':
    main()
