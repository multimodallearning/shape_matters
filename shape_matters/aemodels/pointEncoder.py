import torch
import torch.nn as nn
import torch.nn.functional as F


#adapted from: https://github.com/antao97/dgcnn.pytorch/blob/master/model.py
class PointNet_M(nn.Module):
    def __init__(self, out_chan=2048):
        super(PointNet_M, self).__init__()

        self.out = out_chan

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.pool1 = nn.MaxPool1d(2,2)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.conv6 = nn.Conv1d(256, 512, kernel_size=1, bias=False)
        
        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(64)
        self.bn3 = nn.InstanceNorm1d(64)
        self.bn4 = nn.InstanceNorm1d(128)
        self.bn5 = nn.InstanceNorm1d(256)
        self.bn6 = nn.InstanceNorm1d(512)
        
        self.linear1 = nn.Linear(512, 1024, bias=False)
        self.bn7 = nn.InstanceNorm1d(1024)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(1024, self.out)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        #x = self.pool1(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn7(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)

        #print(x.shape)
        return x
