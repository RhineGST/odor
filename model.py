import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg


class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, cfg.k)

        self.bn_conv1 = nn.BatchNorm1d(64)
        self.bn_conv2 = nn.BatchNorm1d(64)

        self.bn0 = nn.BatchNorm1d(128)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.bn0(x)

        y = F.relu(self.bn_conv1(self.conv1(torch.unsqueeze(x, dim=1))))
        y = F.relu(self.bn_conv2(self.conv2(y)))
        y = torch.max(y, dim=2)[0]
        # print(y.shape)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))

        x = torch.cat([x, y], dim=1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)

        # x = F.sigmoid(self.bn1(self.fc1(x)))
        # x = F.sigmoid(self.bn2(self.fc2(x)))
        # x = self.fc4(x)

        return x
