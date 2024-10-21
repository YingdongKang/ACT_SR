
# code ref: https://github.com/yjn870/FSRCNN-pytorch/blob/master/models.py

# from . import common
import math
from basicsr.utils.registry import ARCH_REGISTRY
# from argparse import Namespace
import torch.nn as nn
import torch
# from models import register
# import torch.nn.functional as F


# def make_model(args, parent=False):
#     return FSRCNN(args)


# @register('FSRCNN')
# def FSRCNN(scale_ratio, rgb_range=1):
#     args = Namespace()
#     args.scale = [scale_ratio]
#     args.n_colors = 3
#     args.rgb_range = rgb_range
#     return FSRCNN(args)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

# @ARCH_REGISTRY.register()
class FSRCNN(nn.Module):
    def __init__(self, scale, num_feat,   d=56, s=12 * 3, m=8):
        super(FSRCNN, self).__init__()

        self.scale = scale
        act = nn.PReLU()
        # act = nn.ReLU()

        m_first_part = []
        m_first_part.append(default_conv(num_feat, d, kernel_size=5))
        m_first_part.append(act)
        self.first_part = nn.Sequential(*m_first_part)

        m_mid_part = []
        m_mid_part.append(default_conv(d, s, kernel_size=1))
        m_mid_part.append(act)
        for _ in range(m):
            m_mid_part.append(default_conv(s, s, kernel_size=3))
            m_mid_part.append(act)
        m_mid_part.append(default_conv(s, d, kernel_size=1))
        m_mid_part.append(act)
        self.mid_part = nn.Sequential(*m_mid_part)

        self.last_part = nn.ConvTranspose2d(d, num_feat, kernel_size=9, stride=scale, padding=9//2,
                                            output_padding=scale-1)

        # self._initialize_weights()


    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x, out_size=None):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x

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
    
    model = FSRCNN(4,3).cuda()
    

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
    # y = model(x)
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
