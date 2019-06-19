from collections import namedtuple

import numpy as np
import torch
from skimage.measure import compare_psnr
from torch import nn

from net import skip
from net.losses import YIQGNGCLoss, ExtendedL1Loss, StdLoss, GradientLoss, GrayLoss
from net.noise import get_noise
from utils.image_io import np_to_torch, save_image, torch_to_np, plot_image_grid

ManyImageWatermarkResult = namedtuple("ManyImageWatermarkResult", ['cleans', 'mask', 'watermark', 'psnr'])


class TwoImagesWatermark(object):
    # DEPRECATED
    def __init__(self, image_name, image1, image2, plot_during_training=True, num_iter_per_step=2000, step_num=2,
                 watermark_hint=None):
        self.image1 = image1
        self.image2 = image2
        self.image_name = image_name
        self.plot_during_training = plot_during_training
        self.watermark_hint_torch = None
        self.watermark_hint = watermark_hint
        self.clean1_net = None
        self.clean2_net = None
        self.watermark_net = None
        self.image1_torch = None
        self.image2_torch = None
        self.clean_net_input1 = None
        self.clean_net_output1 = None
        self.clean_net_input2 = None
        self.clean_net_output2 = None
        self.watermark_net_input = None
        self.watermark_net_output = None
        self.parameters = None
        self.gngc_loss = None
        self.blur_function = None
        self.num_iter_per_step = num_iter_per_step  # per step
        self.steps = step_num
        self.input_depth = 2
        self.multiscale_loss = None
        self.total_loss = None
        self.gngc = None
        self.blur = None
        self.current_gradient = None
        self.current_result = None
        self.best_result = None
        self.learning_rate = 0.001
        self._init_all()

    def _init_nets(self):
        pad = 'reflection'
        clean1 = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.clean1_net = clean1.type(torch.cuda.FloatTensor)

        clean2 = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.clean2_net = clean2.type(torch.cuda.FloatTensor)

        watermark = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32],
            num_channels_up=[8, 16, 32],
            num_channels_skip=[0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.watermark_net = watermark.type(torch.cuda.FloatTensor)

    def _init_images(self):
        self.image1_torch = np_to_torch(self.image1).type(torch.cuda.FloatTensor)
        self.image2_torch = np_to_torch(self.image2).type(torch.cuda.FloatTensor)
        if self.watermark_hint is not None:
            self.watermark_hint_torch = np_to_torch(self.watermark_hint).type(torch.cuda.FloatTensor)

    def _init_noise(self):
        input_type = 'noise'
        # self.left_net_inputs = self.images_torch
        self.clean_net_input1 = get_noise(self.input_depth, input_type,
                                          (self.image1_torch.shape[2],
                                           self.image1_torch.shape[3])).type(torch.cuda.FloatTensor).detach()
        self.clean_net_input2 = get_noise(self.input_depth, input_type,
                                          (self.image2_torch.shape[2],
                                           self.image2_torch.shape[3])).type(torch.cuda.FloatTensor).detach()
        self.watermark_net_input = get_noise(self.input_depth, input_type,
                                          (self.image1_torch.shape[2],
                                           self.image1_torch.shape[3]),
                                             var=1/100.).type(torch.cuda.FloatTensor).detach()

    def _init_parameters(self):
        self.parameters = [p for p in self.clean1_net.parameters()] + \
                          [p for p in self.clean2_net.parameters()] + \
                          [p for p in self.watermark_net.parameters()]

    def _init_losses(self):
        data_type = torch.cuda.FloatTensor
        self.gngc_loss = YIQGNGCLoss().type(data_type)
        self.l1_loss = nn.L1Loss().type(data_type)
        self.extended_l1_loss = ExtendedL1Loss().type(data_type)
        self.blur_function = StdLoss().type(data_type)
        self.gradient_loss = GradientLoss().type(data_type)
        self.gray_loss = GrayLoss().type(data_type)

    def _init_all(self):
        self._init_images()
        self._init_losses()
        self._init_nets()
        self._init_parameters()
        self._init_noise()

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        for step in range(self.steps):
            self._step_initialization_closure(step)
            optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
            for j in range(self.num_iter_per_step):
                optimizer.zero_grad()
                self._optimization_closure(j, step)
                if self.plot_during_training:
                    self._iteration_plot_closure(step, j)
                optimizer.step()
            self._update_result_closure(step)
            self._step_plot_closure(step)

    def finalize(self):
        save_image(self.image_name + "_watermark", self.best_result.watermark)
        save_image(self.image_name + "_clean1", self.best_result.clean1)
        save_image(self.image_name + "_clean2", self.best_result.clean2)
        save_image(self.image_name + "_original1", self.image1)
        save_image(self.image_name + "_original2", self.image2)
        save_image(self.image_name + "_final1", self.image1 - self.best_result.watermark * self.watermark_hint)
        save_image(self.image_name + "_final2", self.image2 - self.best_result.watermark * self.watermark_hint)

    def _update_result_closure(self, step):
        self.current_result = TwoImageWatermarkResult(clean1=torch_to_np(self.clean_net_output1),
                                                      clean2=torch_to_np(self.clean_net_output2),
                                                      watermark=torch_to_np(self.watermark_net_output),
                                                      psnr=self.current_psnr)
        if self.best_result is None or self.best_result.psnr <= self.current_result.psnr:
            self.best_result = self.current_result

    def _step_initialization_closure(self, step):
        """
        at each start of step, we apply this
        :param step:
        :return:
        """
        # we updating the inputs to new noises
        # self._init_nets()
        # self._init_parameters()
        # self._init_noise()
        pass

    def _optimization_closure(self, iteration, step):
        """
        the real iteration is step * self.num_iter_per_step + iteration
        :param iteration:
        :param step:
        :return:
        """
        if iteration == self.num_iter_per_step - 1:
            reg_noise_std = 0
        else:
            reg_noise_std = (1 / 1000.) * (iteration // 200)
        # creates left_net_inputs and right_net_inputs by adding small noise
        clean_net_input1 = self.clean_net_input1 + (self.clean_net_input1.clone().normal_() * reg_noise_std)
        clean_net_input2 = self.clean_net_input2 + (self.clean_net_input2.clone().normal_() * reg_noise_std)
        watermark_net_input = self.watermark_net_input + (self.watermark_net_input.clone().normal_() * reg_noise_std)
        # applies the nets
        self.clean_net_output1 = self.clean1_net(clean_net_input1)
        self.clean_net_output2 = self.clean2_net(clean_net_input2)
        self.watermark_net_output = self.watermark_net(watermark_net_input)
        self.total_loss = 0
        self.gngc = 0
        self.blur = 0
        self.total_loss += self.extended_l1_loss(self.clean_net_output1,
                                                 self.image1_torch,
                                                 (1 - self.watermark_hint_torch))
        self.total_loss += self.extended_l1_loss(self.clean_net_output2,
                                                 self.image2_torch,
                                                 (1 - self.watermark_hint_torch))
        if step == 0:
            # self.total_loss += self.l1_loss(torch.zeros_like(self.watermark_net_output) +
            #                                 self.watermark_hint_torch / 2, self.watermark_net_output)
            self.total_loss.backward(retain_graph=True)
        else:
            self.total_loss += 0.5 * self.l1_loss(self.watermark_hint_torch *
                                                  self.watermark_net_output  # this part learns the watermark
                                                  +
                                                  self.clean_net_output2,
                                                  self.image2_torch)
            self.total_loss += 0.5 * self.l1_loss(self.watermark_hint_torch *
                                                  self.watermark_net_output  # this part learns the watermark
                                                  +
                                                  self.clean_net_output1,
                                                  self.image1_torch)
            self.total_loss.backward(retain_graph=True)

    def _iteration_plot_closure(self, step_number, iter_number):
        clean_out_np1 = torch_to_np(self.clean_net_output1)
        watermark_out_np = torch_to_np(self.watermark_net_output)
        self.current_psnr = compare_psnr(self.image1, self.watermark_hint * watermark_out_np + clean_out_np1)
        if self.current_gradient is not None:
            print('Iteration {:5d} total_loss {:5f} grad {:5f} PSNR {:5f} '.format(iter_number, self.total_loss.item(),
                                                                                   self.current_gradient.item(),
                                                                                   self.current_psnr),
                  '\r', end='')
        else:
            print('Iteration {:5d} total_loss {:5f} PSNR {:5f} '.format(iter_number, self.total_loss.item(),
                                                                        self.current_psnr),
                  '\r', end='')

    def _step_plot_closure(self, step_number):
        """
        runs at the end of each step
        :param step_number:
        :return:
        """
        if self.watermark_hint is not None:
            plot_image_grid("watermark_hint_{}".format(step_number),
                            [np.clip(self.watermark_hint, 0, 1),
                             np.clip(1 - self.watermark_hint, 0, 1)])

        plot_image_grid("watermark_clean_{}".format(step_number),
                        [np.clip(torch_to_np(self.watermark_net_output), 0, 1),
                         np.clip(torch_to_np(self.clean_net_output1), 0, 1)])

        plot_image_grid("learned_image1_{}".format(step_number),
                        [np.clip(self.watermark_hint * torch_to_np(self.watermark_net_output) +
                                 torch_to_np(self.clean_net_output1),
                                 0, 1), self.image1])
        plot_image_grid("learned_image2_{}".format(step_number),
                        [np.clip(self.watermark_hint * torch_to_np(self.watermark_net_output) +
                                 torch_to_np(self.clean_net_output2),
                                 0, 1), self.image2])


class ManyImagesWatermarkNoHint(object):
    def __init__(self, images_names, images, plot_during_training=True, num_iter_per_step=4000, num_step=2):
        self.images = images
        self.images_names = images_names
        self.plot_during_training = plot_during_training
        self.clean_nets = []
        self.watermark_net = None
        self.steps = num_step
        self.images_torch = None
        self.clean_nets_inputs = None
        self.clean_nets_outputs = None
        self.watermark_net_input = None
        self.watermark_net_output = None
        self.mask_net_input = None
        self.mask_net_output = None
        self.parameters = None
        self.gngc_loss = None
        self.blur_function = None
        self.num_iter_per_step = num_iter_per_step  # per step
        self.input_depth = 2
        self.multiscale_loss = None
        self.total_loss = None
        self.gngc = None
        self.blur = None
        self.current_gradient = None
        self.current_result = None
        self.best_result = None
        self.learning_rate = 0.001
        self._init_all()

    def _init_nets(self):
        pad = 'reflection'
        cleans = [skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64],
            num_channels_up=[8, 16, 32, 64],
            num_channels_skip=[0, 0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU') for _ in self.images]

        self.clean_nets = [clean.type(torch.cuda.FloatTensor) for clean in cleans]

        mask_net = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32],
            num_channels_up=[8, 16, 32],
            num_channels_skip=[0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask_net = mask_net.type(torch.cuda.FloatTensor)

        watermark = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64],
            num_channels_up=[8, 16, 32, 64],
            num_channels_skip=[0, 0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.watermark_net = watermark.type(torch.cuda.FloatTensor)

    def _init_images(self):
        self.images_torch = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.images]

    def _init_noise(self):
        input_type = 'noise'
        # self.left_net_inputs = self.images_torch
        self.clean_nets_inputs = [get_noise(self.input_depth, input_type,
                                            (image.shape[2],
                                             image.shape[3])).type(torch.cuda.FloatTensor).detach()
                                  for image in self.images_torch]
        self.mask_net_input = get_noise(self.input_depth, input_type,
                                          (self.images_torch[0].shape[2],
                                           self.images_torch[0].shape[3])).type(torch.cuda.FloatTensor).detach()
        self.watermark_net_input = get_noise(self.input_depth, input_type,
                                          (self.images_torch[0].shape[2],
                                           self.images_torch[0].shape[3]),
                                             var=1/100.).type(torch.cuda.FloatTensor).detach()

    def _init_parameters(self):
        self.parameters = sum([[p for p in clean_net.parameters()] for clean_net in self.clean_nets], []) + \
                          [p for p in self.mask_net.parameters()] + \
                          [p for p in self.watermark_net.parameters()]

    def _init_losses(self):
        data_type = torch.cuda.FloatTensor
        self.gngc_loss = YIQGNGCLoss().type(data_type)
        self.l1_loss = nn.L1Loss().type(data_type)
        self.extended_l1_loss = ExtendedL1Loss().type(data_type)
        self.blur_function = StdLoss().type(data_type)
        self.gradient_loss = GradientLoss().type(data_type)
        self.gray_loss = GrayLoss().type(data_type)

    def _init_all(self):
        self._init_images()
        self._init_losses()
        self._init_nets()
        self._init_parameters()
        self._init_noise()

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        for step in range(self.steps):
            self._step_initialization_closure(step)
            optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
            for j in range(self.num_iter_per_step):
                optimizer.zero_grad()
                self._optimization_closure(j, step)
                if self.plot_during_training:
                    self._iteration_plot_closure(step, j)
                optimizer.step()
            self._update_result_closure(step)
            self._step_plot_closure(step)

    def finalize(self):
        for image_name, clean, image in zip(self.images_names, self.best_result.cleans, self.images):
            save_image(image_name + "_watermark", self.best_result.watermark)
            save_image(image_name + "_mask", self.best_result.mask)
            save_image(image_name + "_obtained_mask", self.best_result.mask * self.best_result.watermark)
            save_image(image_name + "_clean", clean)
            save_image(image_name + "_original", image)

    def _update_result_closure(self, step):
        self.current_result = ManyImageWatermarkResult(cleans=[torch_to_np(c) for c in self.clean_nets_outputs],
                                                       watermark=torch_to_np(self.watermark_net_output),
                                                       mask=torch_to_np(self.mask_net_output),
                                                       psnr=self.current_psnr)
        if self.best_result is None or self.best_result.psnr <= self.current_result.psnr:
            self.best_result = self.current_result

    def _step_initialization_closure(self, step):
        """
        at each start of step, we apply this
        :param step:
        :return:
        """
        # we updating the inputs to new noises
        # self._init_nets()
        # self._init_parameters()
        # self._init_noise()
        pass

    def _optimization_closure(self, iteration, step):
        """
        the real iteration is step * self.num_iter_per_step + iteration
        :param iteration:
        :param step:
        :return:
        """
        if iteration == self.num_iter_per_step - 1:
            reg_noise_std = 0
        else:
            reg_noise_std = (1 / 1000.) * (iteration // 200)
        # creates left_net_inputs and right_net_inputs by adding small noise
        clean_nets_inputs = [clean_net_input + (clean_net_input.clone().normal_() * reg_noise_std)
                             for clean_net_input in self.clean_nets_inputs]
        watermark_net_input = self.watermark_net_input + (self.watermark_net_input.clone().normal_() * reg_noise_std)
        mask_net_input = self.mask_net_input
        # applies the nets
        self.clean_nets_outputs = [clean_net(clean_net_input) for clean_net, clean_net_input
                                   in zip(self.clean_nets, clean_nets_inputs)]
        self.watermark_net_output = self.watermark_net(watermark_net_input)
        self.mask_net_output = 0.8 * self.mask_net(mask_net_input)
        self.total_loss = 0
        self.gngc = 0
        self.blur = 0

        self.total_loss += sum(self.l1_loss(self.watermark_net_output * self.mask_net_output +
                                            clean_net_output * (1 - self.mask_net_output), image_torch)
                               for clean_net_output, image_torch in zip(self.clean_nets_outputs, self.images_torch))
        self.total_loss.backward(retain_graph=True)

    def _iteration_plot_closure(self, step_number, iter_number):
        clean_out_nps = [torch_to_np(clean_net_output) for clean_net_output in self.clean_nets_outputs]
        watermark_out_np = torch_to_np(self.watermark_net_output)
        mask_out_np = torch_to_np(self.mask_net_output)
        self.current_psnr = compare_psnr(self.images[0], clean_out_nps[0] * (1 - mask_out_np) +
                                         mask_out_np * watermark_out_np)
        if self.current_gradient is not None:
            print('Iteration {:5d} total_loss {:5f} grad {:5f} PSNR {:5f} '.format(iter_number, self.total_loss.item(),
                                                                                   self.current_gradient.item(),
                                                                                   self.current_psnr),
                  '\r', end='')
        else:
            print('Iteration {:5d} total_loss {:5f} PSNR {:5f} '.format(iter_number, self.total_loss.item(),
                                                                        self.current_psnr),
                  '\r', end='')

    def _step_plot_closure(self, step_number):
        """
        runs at the end of each step
        :param step_number:
        :return:
        """
        for image_name, image, clean_net_output in zip(self.images_names, self.images, self.clean_nets_outputs):

            # plot_image_grid(image_name + "_watermark_clean_{}".format(step_number),
            #                 [np.clip(torch_to_np(self.watermark_net_output), 0, 1),
            #                  np.clip(torch_to_np(clean_net_output), 0, 1)])

            plot_image_grid(image_name + "_learned_image_{}".format(step_number),
                            [np.clip(torch_to_np(self.watermark_net_output) * torch_to_np(self.mask_net_output) +
                                     (1 - torch_to_np(self.mask_net_output)) * torch_to_np(clean_net_output),
                                 0, 1), image])