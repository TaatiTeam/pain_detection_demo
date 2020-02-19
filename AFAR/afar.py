import sys
sys.path.append('..')
sys.path.append('.')
import torch
import torch.nn as nn


class AFAR(nn.Module):
    def __init__(self, num_classes=1, channels=1, dropout=0.5, fc_size=4608, fc2_size=400):
        super(AFAR, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout))
        # self.fc1 = nn.Linear(10368, 400)
        self.fc1 = nn.Linear(fc_size, fc2_size)
        self.fc2 = nn.Linear(fc2_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
