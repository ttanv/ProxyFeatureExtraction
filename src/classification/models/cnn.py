import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """A simple 1D Convolutional Neural Network."""
    def __init__(self, in_channels: int, num_classes: int):
        """
        Initializes the CNN.

        Args:
            in_channels (int): Number of input channels (e.g., 1 for univariate time-series).
            num_classes (int): Number of output classes for classification.
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        # The input features to the linear layer depends on the output of the conv/pool layers.
        # This needs to be calculated based on your sequence length.
        # Example for a sequence length of 1500 -> 1500/2 (pool1) /2 (pool2) = 375
        # So, 32 channels * 375 sequence length = 12000
        self.fc1 = nn.Linear(32 * 375, num_classes) # Adjust 375 based on your data

    def forward(self, x):
        """The forward pass of the model."""
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        # Flatten the output for the fully connected layer
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out 