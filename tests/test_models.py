import unittest
import torch
from src.models import MultiTaskCNN

class TestMultiTaskCNN(unittest.TestCase):
    def test_initialization(self):
        """Test if the model initializes without error."""
        try:
            model = MultiTaskCNN()
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Failed to initialize MultiTaskCNN: {e}")

    def test_forward_pass(self):
        """Test the forward pass of the model."""
        model = MultiTaskCNN()
        input_tensor = torch.randn(10, 1, 36, 36)  # Batch size of 10, 1 channel, 36x36 images
        try:
            top_left_output, bottom_right_output = model(input_tensor)
            self.assertEqual(top_left_output.shape, (10, 10))  # 10 outputs for 10 classes
            self.assertEqual(bottom_right_output.shape, (10, 10))
        except Exception as e:
            self.fail(f"Forward pass of MultiTaskCNN failed: {e}")

if __name__ == '__main__':
    unittest.main()
