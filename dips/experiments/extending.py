import numpy as np
import torch

from net import skip
from net.noise import get_noise
from utils.image_io import np_to_torch, torch_to_np, save_image, save_heatmap
from utils.imresize import imresize


class SeparationExtendingExperiment(object):
    def __init__(self, image_name, image1, image2, iterations, plot_during_training):
        self.input_depth = 3
        self.iterations = iterations
        self.image1 = image1
        self.image2 = image2
        self.image_name = image_name
        self.learning_rate = 0.001
        self.plot_during_training = plot_during_training
        self._init_all()

    def _init_nets(self):
        pad = 'reflection'
        net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32],
            num_channels_up=[8, 16, 32],
            num_channels_skip=[0, 0, 0],
            filter_size_down=3,
            filter_size_up=3,
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.net1 = net.type(torch.cuda.FloatTensor)

        net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32],
            num_channels_up=[8, 16, 32],
            num_channels_skip=[0, 0, 0],
            filter_size_down=3,
            filter_size_up=3,
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.net2 = net.type(torch.cuda.FloatTensor)

    def _init_image(self):
        self.image_torch1 = np_to_torch(self.image1).type(torch.cuda.FloatTensor)
        self.image_torch2 = np_to_torch(self.image2).type(torch.cuda.FloatTensor)
        self.first_half = np.zeros_like(self.image1)
        self.first_half[:, :, :self.first_half.shape[2] // 2] = 1.
        self.first_half_torch = np_to_torch(self.first_half).type(torch.cuda.FloatTensor)
        self.second_half = np.zeros_like(self.image1)
        self.second_half[:, :, self.second_half.shape[2] // 2:] = 1.
        self.second_half_torch = np_to_torch(self.second_half).type(torch.cuda.FloatTensor)

    def _init_noise(self):
        input_type = 'noise'

        self.net_input1 = get_noise(self.input_depth, input_type,
                                   (self.image_torch1.shape[2],
                                    self.image_torch1.shape[3])).type(torch.cuda.FloatTensor).detach()
        self.net_input2 = get_noise(self.input_depth, input_type,
                                    (self.image_torch2.shape[2],
                                     self.image_torch2.shape[3])).type(torch.cuda.FloatTensor).detach()

    def _init_parameters(self):
        self.parameters = [p for p in self.net2.parameters()] + [p for p in self.net1.parameters()]

    def _init_losses(self):
        data_type = torch.cuda.FloatTensor
        self.l1_loss = torch.nn.MSELoss().type(data_type)

    def _init_all(self):
        self._init_image()
        self._init_losses()
        self._init_noise()

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        self._init_nets()
        self._init_parameters()
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.iterations):
            optimizer.zero_grad()
            self._first_optimization_closure(j)
            if self.plot_during_training:
                self._first_plot_closure(j)
            optimizer.step()
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.iterations):
            optimizer.zero_grad()
            self._second_optimization_closure(j)
            if self.plot_during_training:
                self._second_plot_closure(j)
            optimizer.step()

    def _first_optimization_closure(self, iteration):
        """
        :param iteration:
        :return:
        """
        # creates left_net_inputs and right_net_inputs by adding small noise
        # applies the nets
        if iteration < 1000:
            reg_noise_std = (1 / 1000.) * (iteration // 100)
        else:
            reg_noise_std = 1 / 1000.

        self.net_output1 = self.net1(self.net_input1 + (self.net_input1.clone().normal_() * reg_noise_std))
        self.net_output2 = self.net2(self.net_input2 + (self.net_input2.clone().normal_() * reg_noise_std))
        self.total_loss = self.l1_loss(self.first_half_torch * self.net_output1, self.first_half_torch * self.image_torch1)
        self.total_loss += self.l1_loss(self.first_half_torch * self.net_output2, self.first_half_torch * self.image_torch2)
        self.total_loss.backward(retain_graph=True)

    def _first_plot_closure(self, iter_number):
        print('Iteration {:5d} total_loss {:5f} PSNR {:5f} '.format(iter_number,
                                                                    self.total_loss.item(),
                                                                    0),
              '\r', end='')

    def _second_optimization_closure(self, iteration):
        if iteration < 1000:
            reg_noise_std = (1 / 1000.) * (iteration // 100)
        else:
            reg_noise_std = 1 / 1000.

        self.net_output1 = self.net1(self.net_input1 + (self.net_input1.clone().normal_() * reg_noise_std))
        self.net_output2 = self.net2(self.net_input2 + (self.net_input2.clone().normal_() * reg_noise_std))
        self.total_loss = self.l1_loss(0.5 * self.second_half_torch * (self.net_output1 + self.net_output2),
                                       0.5 * self.second_half_torch * (self.image_torch1 + self.image_torch2))
        self.total_loss += self.l1_loss(self.first_half_torch * self.net_output1, self.first_half_torch * self.image_torch1)
        self.total_loss += self.l1_loss(self.first_half_torch * self.net_output2, self.first_half_torch * self.image_torch2)
        self.total_loss.backward(retain_graph=True)

    def _second_plot_closure(self, iteration):
        pass

    def finalize(self):
        self.net_output1 = self.net1(self.net_input1)
        self.net_output2 = self.net2(self.net_input2)
        save_image("original_image1", self.image1)
        save_image("original_image2", self.image2)
        save_image("learn_on", self.first_half)
        save_image("apply_on", self.second_half)
        save_image("learned_image1", torch_to_np(self.net_output1))
        save_image("learned_image2", torch_to_np(self.net_output2))
        # save_heatmap("heatmap", self._get_heatmap())