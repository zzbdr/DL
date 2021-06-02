import torch.nn as nn
import torch


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(10, 16, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.PReLU(),
        )
        self.offset_layer = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.confidence_layer = nn.Conv2d(32, 1, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pre_layer(x)
        offset = self.offset_layer(x)
        confidence = torch.sigmoid(self.confidence_layer(x))
        return offset, confidence


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(28, 48, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 64, kernel_size=2, stride=1),
            nn.PReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.PReLU()
        )
        self.offset = nn.Linear(128, 4)
        self.confidence = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        offset = self.offset(x)
        confidence = torch.sigmoid(self.confidence(x))
        return offset, confidence


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.PReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(128*3*3, 256),
            nn.PReLU()
        )
        self.offset = nn.Linear(256, 4)
        self.landmark = nn.Linear(256, 10)
        self.confidence = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        offset = self.offset(x)
        landmark = self.landmark(x)
        confidence = torch.sigmoid(self.confidence(x))
        return offset, landmark, confidence
