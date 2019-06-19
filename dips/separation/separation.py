from collections import namedtuple

import numpy as np
import torch.nn as nn
from scipy import signal

from net import *
from net.downsampler import get_imresize_downsampled, get_downsampled
from net.layers import FixedBlurLayer, Ratio
from net.losses import MultiScaleLoss, GrayscaleLoss, StdLoss, YIQGNGCLoss, GNGCLoss, GradientLoss, ExclusionLoss
from net.noise import get_noise, NoiseNet
from skimage.measure import compare_psnr
from net.ssim import SSIM
from utils.image_io import *
from utils.imresize import imresize

SeparationResult = namedtuple("SeparationResult", ['reflection', 'transmission', 'psnr'])


class Separation(object):
    def __init__(self, image_name, image, plot_during_training=True, show_every=500, num_iter=4000,
                 downsampling_factor=0.1, downsampling_number=0,
                 original_reflection=None, original_transmission=None,
                 kernel=None):
        self.downsample_factors = []
        self.image = image
        self.kernel = kernel
        self.downsampling_number = downsampling_number
        self.downsampling_factor = downsampling_factor
        self.plot_during_training = plot_during_training
        # self.ratio = ratio
        self.psnrs = []
        self.show_every = show_every
        self.image_name = image_name
        self.num_iter = num_iter
        self.loss_function = None
        self.gngc_loss = None
        # self.ratio_net = None
        self.parameters = None
        self.learning_rate = 0.001
        self.input_depth = 3
        self.reflection_net_inputs = None
        self.transmission_net_inputs = None
        self.original_transmission = original_transmission
        self.original_reflection = original_reflection
        self.reflection_net = None
        self.transmission_net = None
        self.gngc = None
        self.total_loss = None
        self.multiscale_loss = None
        self.multiscale = None
        self.reflection_outs = None
        self.transmission_outs = None
        self.current_result = None
        self.best_result = None
        self._init_all()

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_losses()

    def _init_images(self):
        self.images = get_imresize_downsampled(self.image, downsampling_factor=self.downsampling_factor,
                                               downsampling_number=self.downsampling_number)
        self.images_torch = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.images]
        if self.original_reflection:
            self.left_original_torch = np_to_torch(self.original_reflection).type(torch.cuda.FloatTensor)
        if self.original_transmission:
            self.right_original_torch = np_to_torch(self.original_transmission).type(torch.cuda.FloatTensor)

    def _init_inputs(self):
        input_type = 'noise'
        # input_type = 'meshgrid'
        data_type = torch.cuda.FloatTensor
        self.reflection_net_inputs = [get_noise(self.input_depth,
                                                input_type,
                                                (image.shape[2], image.shape[3])).type(data_type).detach()
                                      for image in self.images_torch]
        self.transmission_net_inputs = [get_noise(self.input_depth,
                                                  input_type,
                                                  (image.shape[2], image.shape[3])).type(data_type).detach()
                                        for image in self.images_torch]
        if isinstance(self.kernel, np.ndarray):
            self.blur_kernel = FixedBlurLayer(self.kernel)

    def _init_parameters(self):
        self.parameters = [p for p in self.reflection_net.parameters()] + \
                          [p for p in self.transmission_net.parameters()]

    def _init_nets(self):
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'
        reflection_net = skip(
            self.input_depth, self.images[0].shape[0],
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.reflection_net = reflection_net.type(data_type)

        transmission_net = skip(
            self.input_depth, self.images[0].shape[0],
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.transmission_net = transmission_net.type(data_type)

    def _init_losses(self):
        data_type = torch.cuda.FloatTensor
        self.l1_loss = nn.L1Loss().type(data_type)
        self.exclusion_loss = ExclusionLoss().type(data_type)
        self.gngc_loss = YIQGNGCLoss().type(data_type)

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(j)
            self._obtain_current_result()
            if self.plot_during_training:
                self._plot_closure(j)
            optimizer.step()

    def _optimization_closure(self, step):
        if step == self.num_iter:
            reg_noise_std = 0
        elif step < 1000:
            reg_noise_std = (1 / 1000.) * (step // 100)
        else:
            reg_noise_std = 1 / 1000.
        reflection_net_inputs = [reflection_net_input + (reflection_net_input.clone().normal_() * reg_noise_std)
                                 for reflection_net_input in self.reflection_net_inputs]
        transmission_net_inputs = [transmission_net_input + (transmission_net_input.clone().normal_() * reg_noise_std)
                                   for transmission_net_input in self.transmission_net_inputs]

        self.reflection_outs = [self.reflection_net(reflection_net_input)
                                for reflection_net_input in reflection_net_inputs]
        self.transmission_outs = [self.transmission_net(transmission_net_input)
                                  for transmission_net_input in transmission_net_inputs]

        self.total_loss = sum([self.l1_loss(reflection + transmission, image)
                               for reflection, transmission, image in
                               zip(self.reflection_outs, self.transmission_outs, self.images_torch)])
        self.exclusion = sum([self.gngc_loss(transmission, reflection) for reflection, transmission, image in
                              zip(self.reflection_outs, self.transmission_outs, self.images_torch)])
        # self.total_loss += 0.1 * self.exclusion
        self.total_loss.backward()

    def _obtain_current_result(self):
        """
        puts in self.current result the current result.
        also updates the best result
        :return:
        """
        reflection_out_nps = [np.clip(torch_to_np(l), 0, 1) for l in self.reflection_outs]
        transmission_out_nps = [np.clip(torch_to_np(r), 0, 1) for r in self.transmission_outs]
        psnr = compare_psnr(self.images[0],  reflection_out_nps[0] + transmission_out_nps[0])
        self.psnrs.append(psnr)
        self.current_result = SeparationResult(reflection=reflection_out_nps, transmission=transmission_out_nps,
                                               psnr=psnr)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result

    def _plot_closure(self, step):
        print('Iteration {:5d}    Loss {:5f}  PSRN_gt: {:f}'.format(step,
                                                                               self.total_loss.item(),
                                                                              # self.exclusion.item(),
                                                                               self.current_result.psnr),
              '\r', end='')
        if step % self.show_every == self.show_every - 1:
            for i, (reflection, transmission) in enumerate(zip(self.current_result.reflection,
                                                               self.current_result.transmission)):
                plot_image_grid("left_right_{}_{}".format(i, step), [reflection, transmission])
                save_image("sum_{}_{}".format(i, step),reflection +transmission)

    def _plot_distance_map(self):
        calculated_left = self.best_result.reflection[0]
        calculated_right = self.best_result.transmission[0]
        # this is to left for reason
        # print(distance_to_left.shape)
        pass

    def finalize(self):
        save_graph(self.image_name + "_psnr", self.psnrs)
        save_image(self.image_name + "_reflection", self.best_result.reflection[0])
        save_image(self.image_name + "_transmission", self.best_result.transmission[0])
        save_image(self.image_name + "_reflection2", 2 * self.best_result.reflection[0])
        save_image(self.image_name + "_transmission2", 2 * self.best_result.transmission[0])
        save_image(self.image_name + "_original", self.images[0])
        if isinstance(self.original_reflection, np.ndarray) and isinstance(self.original_transmission, np.ndarray):
            self._plot_distance_map()


def mix_images(image1, image2, ratio=0.5):
    """

    :param np.array image1:
    :param np.array image2:
    :param float ratio:
    :return:
    """
    return ratio * image1 + (1 - ratio) * image2


def mul_mix_images(image1, image2):
    """

    :param np.array image1:
    :param np.array image2:
    :param float ratio:
    :return:
    """
    return np.sqrt(image1 * image2)


def realistic_mix_images(image1, image2):
    """

    :param np.array image1:
    :param np.array image2:
    :param float ratio:
    :return:
    """
    ratio = 0.3
    # First a 1-D  Gaussian
    t = np.linspace(-10, 10, 35)
    bump = np.exp(-0.1 * t ** 2)
    bump /= np.trapz(bump)  # normalize the integral to 1

    # make a 2-D kernel out of it
    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
    i1 = ratio * signal.convolve2d(image1[0], kernel, boundary='symm', mode='same') + (1 - ratio) * image2[0]
    i2 = ratio * signal.convolve2d(image1[1], kernel, boundary='symm', mode='same') + (1 - ratio) * image2[1]
    i3 = ratio * signal.convolve2d(image1[2], kernel, boundary='symm', mode='same') + (1 - ratio) * image2[2]

    return ratio * np.array([i1, i2, i3]) + (1-ratio) * image2, kernel, ratio


def two_mixed_images(image1, image2):
    """

    :param np.array image1:
    :param np.array image2:
    :return:

    """
    a = np.random.rand()
    b = np.random.rand()
    return a * image1 + (1 - a) * image2, b * image1 + (1 - b) * image2


def two_masked_mixed_images(image1, image2):
    """

    :param np.array image1:
    :param np.array image2:
    :return:
    """
    a = np.random.rand(1, image1.shape[1] // 32, image1.shape[2] // 32) / 2
    b = np.random.rand(1, image1.shape[1] // 32, image1.shape[2] // 32) / 2 + 0.5
    a = np.clip(imresize(a.transpose(1, 2, 0), output_shape=(image1.shape[1], image1.shape[2]),
                         kernel='linear').transpose(2, 0, 1), 0, 1)
    b = np.clip(imresize(b.transpose(1, 2, 0), output_shape=(image1.shape[1], image1.shape[2]),
                         kernel='linear').transpose(2, 0, 1), 0, 1)
    return a * image1 + (1 - a) * image2, b * image1 + (1 - b) * image2