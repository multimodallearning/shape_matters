import torch
import torch.nn as nn
import torch.nn.functional as F

'''
reference: http://www.multisilicon.com/blog/a25332339.html
'''

class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)
    


class ConvDecoder(nn.Module):
    def __init__(self, in_chan, in_shape):
        super().__init__()
        self.in_chan = in_chan//(4*4*4)
        self.in_shape = in_shape

        self.decoder = nn.Sequential(nn.Conv3d(self.in_chan,256,1),PixelShuffle3d(2),nn.InstanceNorm3d(32),nn.ReLU(),\
                      nn.Conv3d(32,256,1),PixelShuffle3d(2),nn.InstanceNorm3d(32),nn.ReLU(),\
                      nn.Conv3d(32,128,3,padding=1),PixelShuffle3d(2),nn.InstanceNorm3d(16),nn.ReLU(),\
                      nn.Conv3d(16,64,3,padding=1),PixelShuffle3d(2),nn.InstanceNorm3d(8),nn.ReLU(),\
                      nn.Conv3d(8,1,1))
        
    def forward(self, x):
        B,C = x.shape
        x = self.decoder(x.view(B,-1,4,4,4))
        x = F.interpolate(x, self.in_shape, mode = 'nearest')
        return x

