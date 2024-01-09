import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_S(nn.Module):

    def __init__(self, out_chan = 2):
        super(MLP_S, self).__init__()
        self.out_chan = out_chan


        self.mlp = nn.Sequential(nn.Conv1d(2048,256,1),\
                                nn.ReLU(),\
                                nn.Dropout(.5),
                                nn.Conv1d(256,64,1),\
                                nn.ReLU(),\
                                nn.Dropout(.3),
                                nn.Conv1d(64,self.out_chan,1))
        
    def forward(self, x):
        return self.mlp(x)

