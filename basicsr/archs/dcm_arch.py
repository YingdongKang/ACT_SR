
# from . import common

# from argparse import Namespace
import math
import torch
import torch.nn as nn
# from models import register
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

# def make_model(args, parent=False):
#     return DIM(args)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None: 
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


# @register('DCM')
# def DCM(scale_ratio, rgb_range=1):
#     args = Namespace()
#     args.scale = [scale_ratio]
#     args.n_colors = 3
#     args.rgb_range = rgb_range
#     return DIM(args)

@ARCH_REGISTRY.register()
class DCM(nn.Module):
    def __init__(self, num_feat, scale):
        super(DCM, self).__init__()

        self.scale = scale

        # feature extractor part
        self.fe_conv1 = BasicBlock(default_conv, num_feat, 196, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv2 = BasicBlock(default_conv, 196, 166, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv3 = BasicBlock(default_conv, 166, 148, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv4 = BasicBlock(default_conv, 148, 133, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv5 = BasicBlock(default_conv, 133, 120, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv6 = BasicBlock(default_conv, 120, 108, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv7 = BasicBlock(default_conv, 108, 97, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv8 = BasicBlock(default_conv, 97, 86, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv9 = BasicBlock(default_conv, 86, 76, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv10 = BasicBlock(default_conv, 76, 66, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv11 = BasicBlock(default_conv, 66, 57, kernel_size=3, bias=True, act=nn.PReLU())
        self.fe_conv12 = BasicBlock(default_conv, 57, 48, kernel_size=3, bias=True, act=nn.PReLU())

        # reconstruction part
        self.re_a = BasicBlock(default_conv, 196 + 48, 64, kernel_size=3, bias=True, act=nn.PReLU())
        self.re_b1 = BasicBlock(default_conv, 196 + 48, 32, kernel_size=3, bias=True, act=nn.PReLU())
        self.re_b2 = BasicBlock(default_conv, 32, 32, kernel_size=3, bias=True, act=nn.PReLU())
        self.re_u = Upsampler(default_conv, self.scale, 96, act=False)
        self.re_r = default_conv(96, num_feat, kernel_size=1)


    def forward(self, x, out_size=None):

        residual = F.interpolate(x, scale_factor=self.scale, mode='bicubic')

        # feature extractor part
        fe_conv1 = self.fe_conv1(x)
        fe_conv2 = self.fe_conv2(fe_conv1)
        fe_conv3 = self.fe_conv3(fe_conv2)
        fe_conv4 = self.fe_conv4(fe_conv3)
        fe_conv5 = self.fe_conv5(fe_conv4)
        fe_conv6 = self.fe_conv6(fe_conv5)
        fe_conv7 = self.fe_conv7(fe_conv6)
        fe_conv8 = self.fe_conv8(fe_conv7)
        fe_conv9 = self.fe_conv9(fe_conv8)
        fe_conv10 = self.fe_conv10(fe_conv9)
        fe_conv11 = self.fe_conv11(fe_conv10)
        fe_conv12 = self.fe_conv12(fe_conv11)

        # reconstruction part
        feat = torch.cat((fe_conv1, fe_conv12), dim=1)
        re_a = self.re_a(feat)
        re_b1 = self.re_b1(feat)
        re_b2 = self.re_b2(re_b1)
        feat = torch.cat((re_a, re_b2), dim=1)
        re_u = self.re_u(feat)
        re_r = self.re_r(re_u)
        out = re_r + residual

        return out

    # def load_state_dict(self, state_dict, strict=False):
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name in own_state:
    #             if isinstance(param, nn.Parameter):
    #                 param = param.data
    #             try:
    #                 own_state[name].copy_(param)
    #             except Exception:
    #                 if name.find('tail') >= 0:
    #                     print('Replace pre-trained upsampler to new one...')
    #                 else:
    #                     raise RuntimeError('While copying the parameter named {}, '
    #                                        'whose dimensions in the model are {} and '
    #                                        'whose dimensions in the checkpoint are {}.'
    #                                        .format(name, own_state[name].size(), param.size()))
    #         elif strict:
    #             if name.find('tail') == -1:
    #                 raise KeyError('unexpected key "{}" in state_dict'
    #                                .format(name))

    #     if strict:
    #         missing = set(own_state.keys()) - set(state_dict.keys())
    #         if len(missing) > 0:
    #             raise KeyError('missing keys in state_dict: "{}"'.format(missing))



if __name__ == '__main__':
    
    model = DCM(3,4).cuda()
    

    from torchinfo import summary
    import time

    summary(model,input_size=(1, 3, 64, 64))

    class Timer(object):
        """A simple timer."""
        def __init__(self):
            self.total_time = 0.
            self.calls = 0
            self.start_time = 0.
            self.diff = 0.
            self.average_time = 0.

        def tic(self):
            # using time.time instead of time.clock because time time.clock
            # does not normalize for multithreading
            self.start_time = time.time()

        def toc(self, average=True):
            self.diff = time.time() - self.start_time
            self.total_time += self.diff
            self.calls += 1
            self.average_time = self.total_time / self.calls
            if average:
                return self.average_time
            else:
                return self.diff


    # model = nn.Upsample(scale_factor=4,mode='bicubic')
    # summary(model,input_size=(1, 3, 64, 64))
    
    from thop import profile
    
    torch.cuda.reset_max_memory_allocated()
    x = torch.rand(1, 3, 64, 64).cuda()
    y = model(x)
    # 获取模型最大内存消耗
    max_memory_reserved = torch.cuda.max_memory_reserved(device='cuda') / (1024 ** 2)
    print(f"模型最大内存消耗: {max_memory_reserved:.2f} MB")

    flops, params = profile(model, (x,))
    print('flops: %.4f G, params: %.4f M' % (flops / 1e9, params / 1000000.0))

    model = model.cuda()
    x = x.cuda()
    timer = Timer()
    timer.tic()
    for i in range(100):
        timer.tic()
        y = model(x)
        timer.toc()
    print('Do once forward need {:.3f}ms '.format(timer.total_time * 1000 / 100.0))
