import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_dataloaders(path, batch_size):
    with open(path, 'rb') as f:
        trainX, trainLabel, testX, testLabel = pickle.load(f)

    # Transform data into tensors
    trainX = torch.tensor(trainX, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    trainLabel = torch.tensor(trainLabel, dtype=torch.long)
    testX = torch.tensor(testX, dtype=torch.float32).unsqueeze(1)
    testLabel = torch.tensor(testLabel, dtype=torch.long)

    # Create datasets
    train_dataset = TensorDataset(trainX, trainLabel)
    test_dataset = TensorDataset(testX, testLabel)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
