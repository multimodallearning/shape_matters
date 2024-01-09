import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
from torch.autograd.functional import jacobian

class Splat(Function):
    @staticmethod
    def forward(ctx, input, grid, shape):
        device = input.device
        dtype = input.dtype
        
        output = -jacobian(lambda x: (F.grid_sample(x, grid) - input).pow(2).mul(0.5).sum(), torch.zeros(shape.shape, dtype=dtype, device=device))
        
        ctx.save_for_backward(input, grid, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):        
        input, grid, output = ctx.saved_tensors
        
        B, C = input.shape[:2]
        input_dims = input.shape[2:]
        output_dims = grad_output.shape[2:]
    
        y = jacobian(lambda x: F.grid_sample(grad_output.unsqueeze(2).view(B*C, 1, *output_dims), x,  align_corners=False).mean(), grid.unsqueeze(1).repeat(1, C, *([1]*(len(input_dims)+1))).view(B*C, *input_dims, len(input_dims))).view(B, C, *input_dims, len(input_dims))
        
        grad_grid = (input.numel()*input.unsqueeze(-1)*y).sum(1)
        
        grad_input = F.grid_sample(grad_output, grid, align_corners=False)
        
        return grad_input, grad_grid, None

def get_sampler():   
    sampler = Splat().apply
    return sampler

class pbDecoder(nn.Module):
    def __init__(self, patchsize, device="cuda", smooth = True):
        super(pbDecoder, self).__init__()
        self.patchsize = patchsize
        self.smooth = smooth
        self.sampler = get_sampler()
        self.device = device

        self.q_val = torch.randn(1,1,16,120).cuda()
        self.q_val.requires_grad = True

        self.mlp = PointMLP_2(2048, 16*120)
        

    def apply_smooth(self, data):
        if self.smooth:
            smooth = nn.Sequential(nn.AvgPool3d(3,stride=1,padding=1), \
                                   nn.AvgPool3d(3,stride=1,padding=1), \
                                nn.AvgPool3d(3,stride=1,padding=1))
            
            smooth.to(self.device)
            
            output = smooth(data)
            return output
        else:
            return data
    
    def forward(self, q):
        B = q.shape[0]
        _,C,D,H,W = self.patchsize
        zz = torch.zeros(B,C,D,H,W)
        q = self.mlp(q)
        q0 = q.view(B,3,16,-1).float()
        q0_coord = torch.tanh(q0[:,:3])
        q0_val = torch.sigmoid(self.q_val.repeat(B,1,1,1,1,1))

        output = self.sampler(q0_val.view(B,1,-1,1,1),q0_coord.view(B,3,-1,1,1).permute(0,2,3,4,1),zz)
        output0 = self.sampler(torch.ones(B,1,16*q0.shape[3],1,1).cuda(),q0_coord.view(B,3,-1,1,1).permute(0,2,3,4,1),zz)
        z = self.apply_smooth(output)/(self.apply_smooth(output0)+1e-3)
        return z


class PointMLP_2(nn.Module):
    def __init__(self, in_chan, out_points):
        super(PointMLP_2, self).__init__()
        self.in_chan = in_chan
        self.out_points = out_points

        self.conv1 = nn.Conv1d(self.in_chan//8, self.out_points//8,1)# 256--240
        self.conv2 = nn.Conv1d(64, 128,1, groups=64)
        self.conv3 = nn.Conv1d(32, 64, 1, groups=32)
        self.conv4 = nn.Conv1d(16, 64, 1, groups=16)
        self.conv5 = nn.Conv1d(16,3,1)

        self.norm1 = nn.InstanceNorm1d(64)
        self.norm2 = nn.InstanceNorm1d(32)
        self.norm3 = nn.InstanceNorm1d(16)
        self.norm4 = nn.InstanceNorm1d(16)

        

    def forward(self, x):
        x = x.view(x.shape[0], self.in_chan//8, 8)
        x = F.relu(self.norm1(self.conv1(x).view(x.shape[0],64, self.out_points//64)))
        x = F.relu(self.norm2(self.conv2(x).view(x.shape[0],32, self.out_points//16)))
        x = F.relu(self.norm3(self.conv3(x).view(x.shape[0],16, self.out_points//4)))
        x = F.relu(self.norm4(self.conv4(x).view(x.shape[0],16, self.out_points)))
        x = self.conv5(x)
        return x


