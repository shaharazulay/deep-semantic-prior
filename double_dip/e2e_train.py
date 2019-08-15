import glob

from segmentation import *
from utils.image_io import prepare_image, save_image, median
from net import skip
import torch
from torch import optim
from tqdm import tqdm

import numpy as np

import argparse

import os

import matplotlib.pyplot as plt

from cv2.ximgproc import guidedFilter


def apply_guided_filter(image, mask):
    return np.expand_dims(guidedFilter(image.transpose(1, 2, 0).astype(np.float32),
                 mask[0].astype(np.float32), 50, 1e-4), 0)


def to_bin(x):
    v = np.zeros_like(x)
    v[x > 0.5] = 1
    return v


parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str, required=True)
parser.add_argument("--fg_hint", type=str, required=True)
parser.add_argument("--bg_hint", type=str, required=True)
parser.add_argument("--prior_samples", type=int, default=20)
parser.add_argument("--prior_iters", type=int, default=100)
parser.add_argument("--dip_iters", type=int, default=1000)
parser.add_argument("--output_path", type=str, default="./output/")

args = parser.parse_args()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

device = "cuda" if torch.cuda.is_available() else "cpu"


# train prior
img = prepare_image(args.img)

prior_input = torch.from_numpy(img).unsqueeze(0).to(device)
rec_samples = []
for ae_sample in range(1, args.prior_samples + 1):
    ae = skip(prior_input.size(1), prior_input.size(1),
              num_channels_down=[8, 16, 32],
              num_channels_up=[8, 16, 32],
              num_channels_skip=[0, 0, 0],
              upsample_mode='bilinear',
              filter_size_down=3,
              filter_size_up=3,
              need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').to(device)

    # TODO: tune lr?
    optimizer = optim.Adam(ae.parameters(), lr=0.0001)
    loss_fn = torch.nn.L1Loss()

    print("--- Train AE sample % d-----" % ae_sample)
    for _ in tqdm(range(args.prior_iters)):
        optimizer.zero_grad()

        loss = loss_fn(ae(prior_input), prior_input)

        loss.backward()

        optimizer.step()

    rec_samples.append(ae(prior_input).detach())

img_name = os.path.splitext(os.path.basename(args.img))[0]


img_prior = torch.mean(torch.cat(rec_samples), dim=0).cpu().numpy()

# save prior to output
save_image(img_name, img_prior, args.output_path)

fg = prepare_image(args.fg_hint)
bg = prepare_image(args.bg_hint)

masks = []
for _ in range(5):
    s = Segmentation(
        img_name,
        img_prior,
        bg_hint=bg,
        fg_hint=fg,
        plot_during_training=True,
        show_every=200,
        first_step_iter_num=args.dip_iters // 2,
        second_step_iter_num=args.dip_iters,
        output_path=args.output_path)

    s.optimize()
    #plot(s.learning_curve)
    #s.finalize()
    masks.append(apply_guided_filter(img, to_bin(s.best_result.mask)))



# tODO: guided filter with original image
median_mask = to_bin(median(masks))
save_image(img_name + "_final_mask", median_mask, args.output_path)
