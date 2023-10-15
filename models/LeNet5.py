import torch.nn as nn


class LeNet5(nn.Module):

    # Constructor
    def __init__(self):
        super(LeNet5, self).__init__()

        self.avg_pool = nn.AvgPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 4 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x_conv1 = self.relu(self.conv1(x))
        x_conv1 = self.avg_pool(x_conv1)

        x_conv2 = self.relu(self.conv2(x_conv1))
        x_conv2 = self.avg_pool(x_conv2)

        x_flat = self.flatten(x_conv2)
        x_fc1 = self.relu(self.fc1(x_flat))
        x_fc2 = self.relu(self.fc2(x_fc1))
        x_fc3 = self.fc3(x_fc2)

        return x_fc3
