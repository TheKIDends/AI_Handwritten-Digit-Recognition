import torch.nn as nn


class RandomNet(nn.Module):

    # Constructor
    def __init__(self):
        super(RandomNet, self).__init__()

        self.flatten = nn.Flatten()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x_conv1 = self.relu(self.conv1(x))
        x_conv1 = self.max_pool(x_conv1)

        x_conv2 = self.relu(self.conv2(x_conv1))
        x_conv2 = self.max_pool(x_conv2)

        x_conv3 = self.relu(self.conv3(x_conv2))
        x_conv3 = self.max_pool(x_conv3)

        x_flat = self.flatten(x_conv3)
        x_fc1 = self.relu(self.fc1(x_flat))
        x_fc2 = self.relu(self.fc2(x_fc1))
        x_fc3 = self.fc3(x_fc2)

        return x_fc3
