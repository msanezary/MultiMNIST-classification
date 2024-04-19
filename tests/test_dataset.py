import unittest
import torch
from src.dataset import get_dataloaders

class TestDataset(unittest.TestCase):
    def test_dataloader(self):
        """Test if the data loaders are correctly setup and return batches."""
        train_loader, test_loader = get_dataloaders('../data/multi_mnist.pickle', batch_size=32)
        train_batch = next(iter(train_loader))
        self.assertIsInstance(train_batch, tuple)
        self.assertEqual(len(train_batch), 2)  # Images and labels
        self.assertTrue(train_batch[0].shape, (32, 1, 36, 36))  # Batch size, channels, height, width
        self.assertTrue(train_batch[1].shape, (32, 2))  # Batch size, number of tasks

if __name__ == '__main__':
    unittest.main()
