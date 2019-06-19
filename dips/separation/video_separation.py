from collections import namedtuple

import numpy as np
import torch.nn as nn
from scipy import signal

from net import *
from net.downsampler import get_imresize_downsampled, get_downsampled
from net.layers import FixedBlurLayer, Ratio, VectorRatio
from net.losses import MultiScaleLoss, GrayscaleLoss, StdLoss, YIQGNGCLoss, GNGCLoss, GradientLoss, ExclusionLoss
from net.noise import get_noise, NoiseNet, get_video_noise
from skimage.measure import compare_psnr
from net.ssim import SSIM
from utils.image_io import *
from utils.imresize import imresize
import cv2

SeparationResult = namedtuple("SeparationResult", ['reflection', 'transmission', 'psnr', 'alpha'])


def image_histogram_equalization(image):
    image = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / 255.
    return np.clip(image, 0, 1)


def video_histogram_equalization(video):
    v = np.zeros_like(video)
    for image_idx in range(video.shape[0]):
        v[image_idx] = image_histogram_equalization(video[image_idx])
    return v


class ImageVideoSeparation(object):
    def __init__(self, video_name, video, plot_during_training=True, show_every=500, num_iter=4000,
                 original_reflection=None, original_transmission=None, use_alpha=False):
        # we assume the reflection is static
        self.video = video
        self.plot_during_training = plot_during_training
        self.use_alpha = use_alpha
        self.psnrs = []
        self.show_every = show_every
        self.image_name = video_name
        self.num_iter = num_iter
        self.loss_function = None
        self.gngc_loss = None
        self.parameters = None
        self.learning_rate = 0.001
        self.input_depth = 2
        self.reflection_net_input = None
        self.transmission_net_input = None
        self.original_transmission = original_transmission
        self.original_reflection = original_reflection
        self.reflection_net = None
        self.transmission_net = None
        self.gngc = None
        self.total_loss = None
        self.multiscale_loss = None
        self.multiscale = None
        self.reflection_out = None
        self.transmission_out = None
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
        self.video_torch = np_to_torch(self.video).type(torch.cuda.FloatTensor)[0, :, :, :, :]

    def _init_inputs(self):
        input_type = 'noise'
        # input_type = 'meshgrid'
        data_type = torch.cuda.FloatTensor
        self.reflection_net_input = get_noise(self.input_depth, input_type,
                                               (self.video.shape[2], self.video.shape[3])).type(data_type).detach()
        self.transmission_net_input = get_video_noise(self.input_depth, input_type, self.video.shape[0],
                                                      (self.video_torch.shape[2],
                                                      self.video_torch.shape[3])).type(data_type).detach()

    def _init_parameters(self):
        self.parameters = [p for p in self.reflection_net.parameters()] + \
                          [p for p in self.transmission_net.parameters()]
        if self.use_alpha:
            self.parameters += [p for p in self.alpha.parameters()]

    def _init_nets(self):
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'
        reflection_net = skip(
            self.input_depth, self.video.shape[1],
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.reflection_net = reflection_net.type(data_type)

        transmission_net = skip(
            self.input_depth, self.video.shape[1],
            num_channels_down=[8, 16, 32, 64],
            num_channels_up=[8, 16, 32, 64],
            num_channels_skip=[0, 0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.transmission_net = transmission_net.type(data_type)
        if self.use_alpha:
            self.alpha = VectorRatio(self.video.shape[0]).type(data_type)

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
        reg_noise_std = 0
        reflection_net_input = self.reflection_net_input + (self.reflection_net_input.clone().normal_() * reg_noise_std)
        transmission_net_input = self.transmission_net_input + \
                                 (self.transmission_net_input.clone().normal_() * reg_noise_std)

        self.reflection_out = self.reflection_net(reflection_net_input)
        self.transmission_out = self.transmission_net(transmission_net_input)
        if self.use_alpha:
            self.current_alpha = self.alpha()
            self.total_loss = self.l1_loss(self.current_alpha * self.reflection_out +
                                           (1 - self.current_alpha) * self.transmission_out,
                                           self.video_torch)
        else:
            self.total_loss = self.l1_loss(self.reflection_out + self.transmission_out, self.video_torch)
        self.exclusion = self.exclusion_loss(self.reflection_out, self.transmission_out)
        self.total_loss += 0.5 * self.exclusion
        self.total_loss.backward()

    def _obtain_current_result(self):
        """
        puts in self.current result the current result.
        also updates the best result
        :return:
        """
        reflection_out_np = np.clip(self.reflection_out.detach().cpu().numpy(), 0, 1)
        transmission_out_np = np.clip(self.transmission_out.detach().cpu().numpy(), 0, 1)
        if self.use_alpha:
            alpha = self.current_alpha.detach().cpu().numpy()
            psnr = compare_psnr(self.video, alpha * reflection_out_np + (1 - alpha) * transmission_out_np)
        else:
            alpha = None
            psnr = compare_psnr(self.video, reflection_out_np + transmission_out_np)
        self.psnrs.append(psnr)
        self.current_result = SeparationResult(reflection=reflection_out_np, transmission=transmission_out_np,
                                               psnr=psnr, alpha=alpha)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result

    def _plot_closure(self, step):
        print('Iteration {:5d}    Loss {:5f} Exclusion {:5f}  PSRN_gt: {:f}'.format(step,
                                                                                    self.total_loss.item(),
                                                                                    self.exclusion.item(),
                                                                                    self.current_result.psnr),
              '\r', end='')
        if self.plot_during_training and step % self.show_every == self.show_every - 1:
            plot_image_grid("reflection_transmission_{}".format(step),
                            [self.current_result.reflection[0], self.current_result.transmission[20]])
            save_video("sum_{}".format(step), self.current_result.reflection + self.current_result.transmission)

    def finalize(self):
        save_graph(self.image_name + "_psnr", self.psnrs)
        save_image(self.image_name + "_reflection", self.best_result.reflection[0])
        save_video(self.image_name + "_transmission", self.best_result.transmission)
        save_image(self.image_name + "_transmission_img", self.best_result.transmission[20])
        save_image(self.image_name + "_reflection_x2", np.clip(2 * self.best_result.reflection[0], 0, 1))
        save_video(self.image_name + "_transmission_x2", np.clip(2 * self.best_result.transmission, 0, 1))
        save_image(self.image_name + "_transmission_img_x2", np.clip(2 * self.best_result.transmission[20], 0, 1))
        save_image(self.image_name + "_reflection_eq", image_histogram_equalization(self.best_result.reflection[0]))
        save_video(self.image_name + "_transmission_eq", video_histogram_equalization(self.best_result.transmission))
        save_video(self.image_name + "_original", self.video)
        if self.use_alpha:
            print(self.current_alpha.detach().cpu().numpy())


def image_video_separation(name, video):
    s = ImageVideoSeparation(name + "_0", video, use_alpha=False)
    s.optimize()
    s.finalize()
    reflection = s.best_result.reflection[0]
    transmission = s.best_result.transmission

    # for i in range(3):
    #     s = ImageVideoSeparation(name + "_{}".format(i + 1), video - s.best_result.reflection, use_alpha=False)
    #     s.optimize()
    #     # s.finalize()
    #     reflection += s.best_result.reflection[0]
    #     transmission += s.best_result.transmission
    #     save_image(name + "_{}_final_reflection".format(i+1), np.clip(reflection, 0, 1))
    #     save_video(name + "_{}_final_transmission".format(i+1), np.clip(video - reflection, 0, 1))
    #     save_image(name + "_{}_final_reflection_x2".format(i+1), np.clip(2 * reflection, 0, 1))
    #     save_video(name + "_{}_final_transmission_x2".format(i+1), np.clip(2 * (video - reflection), 0, 1))
    #     save_image(name + "_{}_final_reflection_eq".format(i+1),
    #                image_histogram_equalization(np.clip(reflection, 0, 1)))
    #     save_video(name + "_{}_final_transmission_eq".format(i+1),
    #                video_histogram_equalization(np.clip(video - reflection, 0, 1)))


def image_video_separation_with_alpha(name, video):
    s = ImageVideoSeparation(name + "_0", video,
                             use_alpha=True)
    s.optimize()
    s.finalize()
    reflection = (s.best_result.alpha * s.best_result.reflection)[0]
    transmission = (1 - s.best_result.alpha) * s.best_result.transmission
    for i in range(3):
        s = ImageVideoSeparation(name + "_{}".format(i + 1), video - s.best_result.alpha * s.best_result.reflection,
                                 use_alpha=True)
        s.optimize()
        # s.finalize()
        reflection += (s.best_result.alpha * s.best_result.reflection)[0]
        transmission += (1 - s.best_result.alpha) * s.best_result.transmission
        save_image(name + "_{}_final_reflection".format(i+1), np.clip(reflection, 0, 1))
        save_video(name + "_{}_final_transmission".format(i+1), np.clip(video - reflection, 0, 1))
        save_image(name + "_{}_final_reflection_x2".format(i+1), np.clip(2 * reflection, 0, 1))
        save_video(name + "_{}_final_transmission_x2".format(i+1), np.clip(2 * (video - reflection), 0, 1))
        save_image(name + "_{}_final_reflection_eq".format(i+1),
                   image_histogram_equalization(np.clip(reflection, 0, 1)))
        save_video(name + "_{}_final_transmission_eq".format(i+1),
                   video_histogram_equalization(np.clip(video - reflection, 0, 1)))


class AlphaVideoVideoSeparation(object):
    def __init__(self, video_name, video, plot_during_training=True, show_every=20, num_iter=2000,
                 original_reflection=None, original_transmission=None, use_alpha=False):
        # we assume the reflection is static
        self.video = video
        self.plot_during_training = plot_during_training
        self.use_alpha = use_alpha
        self.psnrs = []
        self.show_every = show_every
        self.image_name = video_name
        self.num_iter = num_iter
        self.loss_function = None
        self.gngc_loss = None
        self.parameters = None
        self.learning_rate = 0.001
        self.input_depth = 2
        self.reflection_net_input = None
        self.transmission_net_input = None
        self.original_transmission = original_transmission
        self.original_reflection = original_reflection
        self.reflection_net = None
        self.transmission_net = None
        self.gngc = None
        self.total_loss = None
        self.multiscale_loss = None
        self.multiscale = None
        self.reflection_out_frame = None
        self.transmission_out_frame = None
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
        self.video_torch = np_to_torch(self.video).type(torch.cuda.FloatTensor)[0, :, :, :, :]

    def _init_inputs(self):
        input_type = 'noise'
        data_type = torch.cuda.FloatTensor
        self.reflection_net_input = [get_noise(self.input_depth, input_type,
                                              (self.video.shape[2], self.video.shape[3])).type(data_type).detach()]
        self.reflection_net_input = self.reflection_net_input * self.video.shape[0]
        self.transmission_net_input = get_video_noise(self.input_depth, input_type, self.video.shape[0],
                                                      (self.video_torch.shape[2],
                                                       self.video_torch.shape[3])).type(data_type).detach()

    def _init_parameters(self):
        self.parameters = [p for p in self.reflection_net.parameters()] + \
                          [p for p in self.transmission_net.parameters()]
        if self.use_alpha:
            self.parameters += sum([[p for p in self.alpha[i].parameters()] for i in range(self.video.shape[0])], [])

    def _init_nets(self):
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'
        reflection_net = skip(
            self.input_depth, self.video.shape[1],
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.reflection_net = reflection_net.type(data_type)

        transmission_net = skip(
            self.input_depth, self.video.shape[1],
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.transmission_net = transmission_net.type(data_type)
        if self.use_alpha:
            self.alpha = [Ratio() for r in range(self.video.shape[0])]

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
        for i in range(self.video.shape[0]):
            reg_noise_std = 0
            reflection_net_input = self.reflection_net_input[i] + \
                                   (self.reflection_net_input[i].clone().normal_() * reg_noise_std)
            transmission_net_input = self.transmission_net_input[i:i+1] + \
                                     (self.transmission_net_input[i:i+1].clone().normal_() * reg_noise_std)

            self.reflection_out_frame = self.reflection_net(reflection_net_input)
            self.transmission_out_frame = self.transmission_net(transmission_net_input)
            if self.use_alpha:
                self.current_alpha = self.alpha[i]()
                self.total_loss = self.l1_loss(self.current_alpha * self.reflection_out_frame +
                                               (1 - self.current_alpha) * self.transmission_out_frame,
                                               self.video_torch[i:i+1])
            else:
                self.total_loss = self.l1_loss(0.5 * self.reflection_out_frame + 0.5 * self.transmission_out_frame,
                                               self.video_torch[i:i+1])
            self.exclusion = self.exclusion_loss(self.reflection_out_frame, self.transmission_out_frame)
            self.total_loss += 0.5 * self.exclusion
            self.total_loss.backward()

    def _get_videos(self):
        """
        :return: numpy representation of the video
        """
        transmission = []
        reflection = []
        alpha = []
        for i in range(self.video.shape[0]):
            reflection_net_input = self.reflection_net_input[i]
            transmission_net_input = self.transmission_net_input[i:i+1]
            reflection.append(np.clip(self.reflection_net(reflection_net_input)[0].detach().cpu().numpy(), 0, 1))
            transmission.append(np.clip(self.transmission_net(transmission_net_input)[0].detach().cpu().numpy(), 0, 1))
            if self.use_alpha:
                alpha.append(np.clip(self.alpha[i]().detach().cpu().numpy(), 0, 1))
        self.transmission_out = np.array(transmission)
        self.reflection_out = np.array(reflection)
        if self.use_alpha:
            self.alpha_out = np.array(alpha).reshape(self.video.shape[0], 1, 1, 1)
        else:
            self.alpha_out = None

    def _obtain_current_result(self):
        """
        puts in self.current result the current result.
        also updates the best result
        :return:
        """
        # if np.random.rand() > 0.5:
        #     return
        self._get_videos()
        # print(self.video.shape, self.transmission_out.shape, self.reflection_out.shape)
        if self.use_alpha:
            psnr = compare_psnr(self.video, self.alpha_out * self.reflection_out +
                                (1 - self.alpha_out) * self.transmission_out)
        else:
            alpha = None
            psnr = compare_psnr(self.video, self.reflection_out + self.transmission_out)
        self.psnrs.append(psnr)
        self.current_result = SeparationResult(reflection=self.reflection_out, transmission=self.transmission_out,
                                               psnr=psnr, alpha=self.alpha_out)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result

    def _plot_closure(self, step):
        print('Iteration {:5d}    Loss {:5f} Exclusion {:5f}  PSRN_gt: {:f}'.format(step,
                                                                                    self.total_loss.item(),
                                                                                    self.exclusion.item(),
                                                                                    self.current_result.psnr),
              '\r', end='')
        if self.plot_during_training and step % self.show_every == self.show_every - 1:
            plot_image_grid("reflection_transmission_{}".format(step),
                            [self.current_result.reflection[20], self.current_result.transmission[20]])
            # if self.use_alpha:
            #     print(self.alpha_out.reshape(self.video.shape[0]))
            # save_video("sum_{}".format(step), self.current_result.reflection + self.current_result.transmission)

    def finalize(self):
        save_graph(self.image_name + "_psnr", self.psnrs)
        save_video(self.image_name + "_reflection", self.best_result.reflection)
        save_video(self.image_name + "_transmission", self.best_result.transmission)
        save_video(self.image_name + "_reflection_x2", np.clip(2 * self.best_result.reflection, 0, 1))
        save_video(self.image_name + "_transmission_x2", np.clip(2 * self.best_result.transmission, 0, 1))
        save_video(self.image_name + "_reflection_eq", video_histogram_equalization(self.best_result.reflection))
        save_video(self.image_name + "_transmission_eq", video_histogram_equalization(self.best_result.transmission))
        save_video(self.image_name + "_original", self.video)
        if self.use_alpha:
            print(self.alpha_out.reshape(self.video.shape[0]))



class VideoVideoSeparation(object):
    def __init__(self, video_name, video, plot_during_training=True, show_every=500, num_iter=4000,
                 original_reflection=None, original_transmission=None):
        # we assume the reflection is static
        self.video = np.mean(video, axis=1 ,keepdims=True)
        self.plot_during_training = plot_during_training
        self.psnrs = []
        self.show_every = show_every
        self.image_name = video_name
        self.num_iter = num_iter
        self.loss_function = None
        self.gngc_loss = None
        self.parameters = None
        self.learning_rate = 0.001
        self.input_depth = 2
        self.reflection_net_input = None
        self.transmission_net_input = None
        self.original_transmission = original_transmission
        self.original_reflection = original_reflection
        self.reflection_net = None
        self.transmission_net = None
        self.gngc = None
        self.total_loss = None
        self.multiscale_loss = None
        self.multiscale = None
        self.reflection_out = None
        self.transmission_out = None
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
        self.video_torch = np_to_torch(self.video).type(torch.cuda.FloatTensor)[0, :, :, :, :]

    def _init_inputs(self):
        input_type = 'noise'
        # input_type = 'meshgrid'
        data_type = torch.cuda.FloatTensor
        self.reflection_net_input = get_video_noise(self.input_depth, input_type, self.video.shape[0],
                                                      (self.video_torch.shape[2],
                                                      self.video_torch.shape[3]), var=.1).type(data_type).detach()
        self.transmission_net_input = get_video_noise(self.input_depth, input_type, self.video.shape[0],
                                                      (self.video_torch.shape[2],
                                                      self.video_torch.shape[3]), var=.1).type(data_type).detach()

    def _init_parameters(self):
        self.parameters = [p for p in self.reflection_net.parameters()] + \
                          [p for p in self.transmission_net.parameters()]

    def _init_nets(self):
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'
        reflection_net = skip(
            self.input_depth, self.video.shape[1],
            num_channels_down=[8, 16, 32, 64],
            num_channels_up=[8, 16, 32, 64],
            num_channels_skip=[0, 0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.reflection_net = reflection_net.type(data_type)

        transmission_net = skip(
            self.input_depth, self.video.shape[1],
            num_channels_down=[8, 16, 32, 64],
            num_channels_up=[8, 16, 32, 64],
            num_channels_skip=[0, 0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.transmission_net = transmission_net.type(data_type)

    def _init_losses(self):
        data_type = torch.cuda.FloatTensor
        self.l1_loss = nn.L1Loss().type(data_type)
        # self.exclusion_loss = ExclusionLoss().type(data_type)
        self.gngc_loss = YIQGNGCLoss().type(data_type)

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(j)
            self._obtain_current_result(j)
            if self.plot_during_training:
                self._plot_closure(j)
            optimizer.step()

    def _optimization_closure(self, step):
        if step == self.num_iter - 1:
            reg_noise_std = 0
        elif step < 1000:
             reg_noise_std = (1 / 1000.) * (step // 100)
        else:
             reg_noise_std = 1 / 1000.
        reflection_net_input = self.reflection_net_input + (self.reflection_net_input.clone().normal_() * reg_noise_std)
        transmission_net_input = self.transmission_net_input + \
                                 (self.transmission_net_input.clone().normal_() * reg_noise_std)

        self.reflection_out = self.reflection_net(reflection_net_input)
        self.transmission_out = self.transmission_net(transmission_net_input)
        self.total_loss = self.l1_loss(0.5 * self.reflection_out + 0.5 * self.transmission_out, self.video_torch)
        self.exclusion = self.gngc_loss(self.reflection_out, self.transmission_out)
        self.total_loss += 0.5 * self.exclusion
        self.total_loss.backward()

    def _obtain_current_result(self, j):
        """
        puts in self.current result the current result.
        also updates the best result
        :return:
        """
        if j > 0 and (j < 1000 or j % 10 != 9):
            return
        reflection_out_np = np.clip(self.reflection_out.detach().cpu().numpy(), 0, 1)
        transmission_out_np = np.clip(self.transmission_out.detach().cpu().numpy(), 0, 1)
        psnr = compare_psnr(self.video, 0.5 * reflection_out_np + 0.5 * transmission_out_np)
        self.psnrs.append(psnr)
        self.current_result = SeparationResult(reflection=reflection_out_np, transmission=transmission_out_np,
                                               psnr=psnr, alpha=None)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result

    def _plot_closure(self, step):
        print('Iteration {:5d}    Loss {:5f} PSRN_gt: {:f}'.format(step,
                                                                                    self.total_loss.item(),
                                                                     #               self.exclusion.item(),
                                                                                    self.current_result.psnr),
              '\r', end='')
        if self.plot_during_training and step % self.show_every == self.show_every - 1:
            plot_image_grid("reflection_transmission_{}".format(step),
                            [self.current_result.reflection[20], self.current_result.transmission[20]])
            save_video("sum_{}".format(step), self.current_result.reflection + self.current_result.transmission)

    def finalize(self):
        save_graph(self.image_name + "_psnr", self.psnrs)
        save_video(self.image_name + "_reflection", self.best_result.reflection)
        save_video(self.image_name + "_transmission", self.best_result.transmission)
        # save_video(self.image_name + "_reflection_x2", np.clip(2 * self.best_result.reflection, 0, 1))
        # save_video(self.image_name + "_transmission_x2", np.clip(2 * self.best_result.transmission, 0, 1))
        save_video(self.image_name + "_reflection_eq", video_histogram_equalization(self.best_result.reflection))
        save_video(self.image_name + "_transmission_eq", video_histogram_equalization(self.best_result.transmission))
        save_video(self.image_name + "_original", self.video)


def alpha_video_video_separation(name, video):
    s = AlphaVideoVideoSeparation(name + "_bigger_net", video)
    s.optimize()
    s.finalize()
    # reflection = s.best_result.reflection[0]
    # transmission = s.best_result.transmission
    # for i in range(3):
    #     s = VideoVideoSeparation(name + "_{}".format(i + 1), video - s.best_result.reflection)
    #     s.optimize()
    #     # s.finalize()
    #     reflection += s.best_result.reflection[0]
    #     transmission += s.best_result.transmission
    #     save_image(name + "_{}_final_reflection".format(i + 1), np.clip(reflection, 0, 1))
    #     save_video(name + "_{}_final_transmission".format(i + 1), np.clip(video - reflection, 0, 1))
    #     save_image(name + "_{}_final_reflection_x2".format(i + 1), np.clip(2 * reflection, 0, 1))
    #     save_video(name + "_{}_final_transmission_x2".format(i + 1), np.clip(2 * (video - reflection), 0, 1))
    #     save_image(name + "_{}_final_reflection_eq".format(i + 1),
    #                image_histogram_equalization(np.clip(reflection, 0, 1)))
    #     save_video(name + "_{}_final_transmission_eq".format(i + 1),
    #                video_histogram_equalization(np.clip(video - reflection, 0, 1)))

