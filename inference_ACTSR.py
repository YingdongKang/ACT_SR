import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torch.nn import functional as F

from basicsr.archs.actsr_arch import ACTSR



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/data/data/sentinel/lr_x2_split/test', help='input test image folder')
    parser.add_argument('--output', type=str, default='/data/data/sentinel/lr_x2_split/test_x4', help='output folder')
    # dn: denoising; car: compression artifact removal
    # TODO: it now only supports sr, need to adapt to dn and jpeg_car
    parser.add_argument('--patch_size', type=int, default=500, help='training patch size')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument(
        '--model_path',
        type=str,
        default='./experiments/pre_train/net_g.pth')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = define_model(args)
    model.eval()
    model = model.to(device)

    
    window_size = 16

    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        # read image
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        # img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = img.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

            output = model(img)
            _, _, h, w = output.size()
            output = output[:, :, 0:h - mod_pad_h * args.scale, 0:w - mod_pad_w * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(args.output, f'{imgname}_ACT.png'), output)


def define_model(args):
    # 001 classical image sr
    
    model = ACTSR(
        upscale=args.scale,
        in_chans=3,
        img_size=args.patch_size,
        window_size=16,
        compress_ratio=10,
        squeeze_factor=30,
        img_range=1.,
        depths=[8, 8, 8, 8],
        embed_dim=96,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv')


    loadnet = torch.load(args.model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)

    return model


if __name__ == '__main__':
    main()
