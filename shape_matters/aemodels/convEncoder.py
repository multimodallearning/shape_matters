import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(nn.Conv3d(1,16,3,padding=1,stride=2),nn.InstanceNorm3d(16),nn.ReLU(),\
                    nn.Conv3d(16,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),nn.MaxPool3d(2),\
                    nn.Conv3d(32,64,3,padding=1),nn.InstanceNorm3d(64),nn.ReLU(),\
                    nn.Conv3d(64,128,3,padding=1),nn.InstanceNorm3d(128),nn.ReLU(),nn.MaxPool3d(2),\
                    nn.Conv3d(128,256,3,padding=1),nn.InstanceNorm3d(256),nn.ReLU(),\
                    nn.Conv3d(256,256,3,padding=1),nn.InstanceNorm3d(128),nn.ReLU(),nn.MaxPool3d(2),\
                    nn.Conv3d(256,256,1),nn.InstanceNorm3d(256),nn.ReLU(),nn.Conv3d(256,16,1))
        
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*6*4*5, 2048)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.proj(x)
        return x