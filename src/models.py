import torch.nn as nn

class MultiTaskCNN(nn.Module):
    def __init__(self):
        super(MultiTaskCNN, self).__init__()
        # Shared layers
        self.conv_base = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Linear(32 * 18 * 18, 128)

        # Task-specific layers
        self.fc_top_left = nn.Linear(128, 10)
        self.fc_bottom_right = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv_base(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        top_left_output = self.fc_top_left(x)
        bottom_right_output = self.fc_bottom_right(x)
        return top_left_output, bottom_right_output
