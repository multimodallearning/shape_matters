import torch
import torch.nn as nn
import torch.nn.functional as F


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self,in_pc, target_pc):
        return computeChamfer(in_pc, target_pc)


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def computeChamfer(pc1, pc2):

    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)

    sqrdist12 = square_distance(pc1, pc2)

    dist1, _ = torch.topk(sqrdist12, 1, dim = -1, largest=False, sorted=False)
    dist2, _ = torch.topk(sqrdist12, 1, dim = 1, largest=False, sorted=False)
    dist1 = dist1.squeeze(2)
    dist2 = dist2.squeeze(1)

    m_dist = (dist1 + dist2) / 2.0

    return m_dist.mean()

