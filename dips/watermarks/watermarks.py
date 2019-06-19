from collections import namedtuple

from skimage.measure import compare_psnr

from net import *
from net.downsampler import *
from net.losses import StdLoss, YIQGNGCLoss, GradientLoss, ExtendedL1Loss, GrayLoss
from net.noise import get_noise

WatermarkResult = namedtuple("WatermarkResult", ['clean', 'watermark', 'mask', 'psnr'])


class Watermark(object):
    def __init__(self, image_name, image, plot_during_training=True, num_iter_per_step=2000,
                 watermark_hint=None):
        self.image = image
        self.image_name = image_name
        self.plot_during_training = plot_during_training
        self.watermark_hint_torch = None
        self.watermark_hint = watermark_hint
        self.clean_net = None
        self.watermark_net = None
        self.image_torch = None
        self.clean_net_input = None
        self.watermark_net_input = None
        self.clean_net_output = None
        self.watermark_net_output = None
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
        clean = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.clean_net = clean.type(torch.cuda.FloatTensor)

        watermark = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32],
            num_channels_up=[8, 16, 32],
            num_channels_skip=[0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.watermark_net = watermark.type(torch.cuda.FloatTensor)

        mask = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64],
            num_channels_up=[8, 16, 32, 64],
            num_channels_skip=[0, 0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
        self.mask_net = mask.type(torch.cuda.FloatTensor)

    def _init_images(self):
        self.image_torch = np_to_torch(self.image).type(torch.cuda.FloatTensor)
        if self.watermark_hint is not None:
            self.watermark_hint_torch = np_to_torch(self.watermark_hint).type(torch.cuda.FloatTensor)

    def _init_noise(self):
        input_type = 'noise'
        # self.left_net_inputs = self.images_torch
        self.clean_net_input = get_noise(self.input_depth, input_type,
                                         (self.image_torch.shape[2],
                                          self.image_torch.shape[3])).type(torch.cuda.FloatTensor).detach()
        self.watermark_net_input = get_noise(self.input_depth, input_type,
                                         (self.image_torch.shape[2],
                                          self.image_torch.shape[3])).type(torch.cuda.FloatTensor).detach()
        self.mask_net_input = get_noise(self.input_depth, input_type,
                                         (self.image_torch.shape[2],
                                          self.image_torch.shape[3])).type(torch.cuda.FloatTensor).detach()

    def _init_parameters(self):
        self.parameters = [p for p in self.clean_net.parameters()] + \
                          [p for p in self.watermark_net.parameters()] + \
                          [p for p in self.mask_net.parameters()]

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
        # step 1
        self._step_initialization_closure(0)
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter_per_step):
            optimizer.zero_grad()
            self._step1_optimization_closure(j, 0)
            if self.plot_during_training:
                self._iteration_plot_closure(0, j)
            optimizer.step()
        self._update_result_closure(0)
        self._step_plot_closure(0)
        # step 2
        self._step_initialization_closure(1)
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter_per_step):
            optimizer.zero_grad()
            self._step2_optimization_closure(j, 1)
            if self.plot_during_training:
                self._iteration_plot_closure(1, j)
            optimizer.step()
        self._update_result_closure(1)
        self._step_plot_closure(1)

    def finalize(self):
        save_image(self.image_name + "_watermark", self.best_result.watermark)
        save_image(self.image_name + "_clean", self.best_result.clean)
        save_image(self.image_name + "_original", self.image)
        save_image(self.image_name + "_mask", self.best_result.mask)
        save_image(self.image_name + "_final", (1 - self.watermark_hint) * self.image +
                   self.best_result.clean * self.watermark_hint)

    def _update_result_closure(self, step):
        self.current_result = WatermarkResult(clean=torch_to_np(self.clean_net_output),
                                              watermark=torch_to_np(self.watermark_net_output),
                                              mask=torch_to_np(self.mask_net_output),
                                              psnr=self.current_psnr)
        # if self.best_result is None or self.best_result.psnr <= self.current_result.psnr:
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

    def _step2_optimization_closure(self, iteration, step):
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
        clean_net_input = self.clean_net_input + (self.clean_net_input.clone().normal_() * reg_noise_std)
        watermark_net_input = self.watermark_net_input + (self.watermark_net_input.clone().normal_() * reg_noise_std)
        mask_net_input = self.mask_net_input  # + (self.mask_net_input.clone().normal_() * reg_noise_std)
        # applies the nets
        self.clean_net_output = self.clean_net(clean_net_input)
        self.watermark_net_output = self.watermark_net(watermark_net_input)
        self.mask_net_output = 0.8 * self.mask_net(mask_net_input)
        self.total_loss = 0
        self.gngc = 0
        self.blur = 0
        # loss on clean region
        self.total_loss += self.extended_l1_loss(self.clean_net_output,
                                                 self.image_torch,
                                                 (1 - self.watermark_hint_torch))
        # loss in second region
        self.total_loss += 0.5 * self.l1_loss(self.watermark_hint_torch *
                                              self.mask_net_output * self.watermark_net_output
                                              +
                                              (1 - self.mask_net_output) * self.clean_net_output,
                                              self.image_torch)  # this part learns the watermark
        self.total_loss.backward(retain_graph=True)

    def _step1_optimization_closure(self, iteration, step):
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
        clean_net_input = self.clean_net_input + (self.clean_net_input.clone().normal_() * reg_noise_std)
        watermark_net_input = self.watermark_net_input#  + (self.watermark_net_input.clone().normal_())
        mask_net_input = self.mask_net_input
        # applies the nets
        self.clean_net_output = self.clean_net(clean_net_input)
        self.watermark_net_output = self.watermark_net(watermark_net_input)
        self.mask_net_output = self.mask_net(mask_net_input)
        self.total_loss = 0
        self.gngc = 0
        self.blur = 0
        self.total_loss += self.extended_l1_loss(self.clean_net_output,
                                                 self.image_torch,
                                                 (1 - self.watermark_hint_torch))
        self.total_loss.backward(retain_graph=True)

    def _iteration_plot_closure(self, step_number, iter_number):
        clean_out_np = torch_to_np(self.clean_net_output)
        watermark_out_np = torch_to_np(self.watermark_net_output)
        mask_out_np = torch_to_np(self.watermark_net_output)
        if step_number == 0:
            self.current_psnr = 0
        self.current_psnr = compare_psnr(self.image, mask_out_np * self.watermark_hint * watermark_out_np +
                                         (1 - mask_out_np) * clean_out_np)
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
        save_image(self.image_name + "_completion", torch_to_np(self.clean_net_output))
        if self.watermark_hint is not None:
            plot_image_grid("watermark_hint_and_mask_{}".format(step_number),
                            [np.clip(self.watermark_hint, 0, 1),
                             np.clip(torch_to_np(self.mask_net_output), 0, 1)])

        plot_image_grid("watermark_clean_{}".format(step_number),
                        [np.clip(torch_to_np(self.watermark_net_output), 0, 1),
                         np.clip(torch_to_np(self.clean_net_output), 0, 1)])

        plot_image_grid("learned_image_{}".format(step_number),
                        [np.clip(self.watermark_hint * torch_to_np(self.watermark_net_output) +
                                 torch_to_np(self.clean_net_output),
                                 0, 1), self.image])


def remove_watermark(image_name, image, fg):
    results = []
    for i in range(5):
        s = Watermark(image_name+"_{}".format(i), image, watermark_hint=fg)
        s.optimize()
        s.finalize()
        results.append(s.best_result)

    save_image(image_name + "_watermark", median([best_result.watermark for best_result in results]))
    save_image(image_name + "_clean", median([best_result.clean for best_result in results]))
    save_image(image_name + "_original", image)
    save_image(image_name + "_final", (1 - fg) * image + fg * median([best_result.clean for best_result in results]))
    save_image(image_name + "_mask", median([best_result.mask for best_result in results]))
    save_image(image_name + "_hint", fg)
    recovered_mask = fg * median([best_result.mask for best_result in results])
    clear_image_places = np.zeros_like(recovered_mask)
    clear_image_places[recovered_mask < 0.03] = 1
    save_image(image_name + "_real_final", clear_image_places * image + (1 - clear_image_places) *
               median([best_result.clean for best_result in results]))
    recovered_watermark = fg * median([best_result.watermark * best_result.mask for best_result in results])
    save_image(image_name + "_recovered_watermark", recovered_watermark)
