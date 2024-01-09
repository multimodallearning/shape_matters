import torch
import torch.nn as nn
import torch.nn.functional as F


class PointDecoder(nn.Module):
    def __init__(self, in_chan=1920*3):
        super(PointDecoder, self).__init__()

        self.in_chan = 2048
        self.out_points = 1920

        self.conv1 = nn.Conv1d(self.in_chan//8, self.out_points//8,1)
        self.conv2 = nn.Conv1d(64, 128,1, groups=64)
        self.conv3 = nn.Conv1d(32, 64, 1, groups=32)
        self.conv4 = nn.Conv1d(16, 64, 1, groups=16)
        self.conv5 = nn.Conv1d(16,32,1, groups=16)
        self.conv6 = nn.Conv1d(16,3,1)

        self.norm1 = nn.InstanceNorm1d(64)
        self.norm2 = nn.InstanceNorm1d(32)
        self.norm3 = nn.InstanceNorm1d(16)
        self.norm4 = nn.InstanceNorm1d(16)
        self.norm5 = nn.InstanceNorm1d(16)

    def forward(self, x):
        x = x.view(x.shape[0], self.in_chan//8, 8)
        x = F.relu(self.norm1(self.conv1(x).view(x.shape[0],64, self.out_points//64)))
        x = F.relu(self.norm2(self.conv2(x).view(x.shape[0],32, self.out_points//16)))
        x = F.relu(self.norm3(self.conv3(x).view(x.shape[0],16, self.out_points//4)))
        x = F.relu(self.norm4(self.conv4(x).view(x.shape[0],16, self.out_points)))
        x = F.relu(self.norm5(self.conv5(x).view(x.shape[0],16, self.out_points*2)))
        x = torch.tanh(self.conv6(x))#was tanh before
        return x
    

