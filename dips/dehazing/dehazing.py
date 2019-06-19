from collections import namedtuple

from cv2.ximgproc import guidedFilter

from net import *
from net.losses import StdLoss
from utils.imresize import imresize, np_imresize
from net.noise import get_noise
from net.ssim import SSIM
from utils.image_io import *
from skimage.measure import compare_psnr
import torch.nn as nn
from .ambient_model import AmbientModel
from .dark_channel_prior import get_atmosphere
import progressbar


DehazeResult = namedtuple("DehazeResult", ['learned', 't', 'a', 'psnr'])


class Dehaze(object):
    CROP_SIZE = 2048
    def __init__(self, image_name, image, num_iter=8000, plot_during_training=True,
                 show_every=500,
                 use_deep_channel_prior=True,
                 gt_ambient=None, clip=True):
        self.image_name = image_name
        self.image = image

        self.num_iter = num_iter
        self.plot_during_training = plot_during_training
        self.show_every = show_every
        self.use_deep_channel_prior = use_deep_channel_prior
        self.gt_ambient = gt_ambient  # np
        self.ambient_net = None
        self.image_net = None
        self.mask_net = None
        self.ambient_val = None
        self.mse_loss = None
        self.learning_rate = 0.001
        self.parameters = None
        self.current_result = None

        self.clip = clip
        self.blur_loss = None
        self.best_result = None
        self.image_net_input = None
        self.mask_net_input = None
        self.image_out = None
        self.mask_out = None
        self.done = False
        self.ambient_out = None
        self.total_loss = None
        self.input_depth = 1
        self.post = None
        self._init_all()

    def _init_images(self):
        self.original_image = self.image.copy()
        factor = 1
        image = self.image
        while image.shape[1] >= 800 or image.shape[2] >= 800:
            new_shape_x, new_shape_y = self.image.shape[1] / factor, self.image.shape[2] /factor
            new_shape_x -= (new_shape_x % 32)
            new_shape_y -= (new_shape_y % 32)
            image = np_imresize(self.image, output_shape=(new_shape_x, new_shape_y))
            factor += 1
        self.image = image
        self.image_torch = np_to_torch(self.image).type(torch.cuda.FloatTensor)

    def _is_learning_ambient(self):
        """
        true if the ambient is learned during the optimization process
        :return:
        """
        return not self.use_deep_channel_prior and not isinstance(self.gt_ambient, np.ndarray)

    def _init_nets(self):
        input_depth = self.input_depth
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'

        image_net = skip(
            input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.image_net = image_net.type(data_type)

        mask_net = skip(
            input_depth, 1,
            num_channels_down=[8, 16, 32, 64],
            num_channels_up=[8, 16, 32, 64],
            num_channels_skip=[0, 0, 0, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask_net = mask_net.type(data_type)

    def _init_ambient(self):
        if self._is_learning_ambient():
            ambient_net = AmbientModel((self.image.shape[1], self.image.shape[2]))
            self.ambient_net = ambient_net.type(torch.cuda.FloatTensor)
        else:
            if isinstance(self.gt_ambient, np.ndarray):
                atmosphere = self.gt_ambient
            else:
                # use_deep_channel_prior is True
                atmosphere = get_atmosphere(self.image)
            self.ambient_val = nn.Parameter(data=torch.cuda.FloatTensor(atmosphere.reshape((1, 3, 1, 1))),
                                            requires_grad=False)

    def _init_parameters(self):
        parameters = [p for p in self.image_net.parameters()] + \
                     [p for p in self.mask_net.parameters()]
        if self._is_learning_ambient():
            parameters += [p for p in self.ambient_net.parameters()]

        self.parameters = parameters

    def _init_loss(self):
        data_type = torch.cuda.FloatTensor
        self.mse_loss = torch.nn.MSELoss().type(data_type)
        # self.mse_loss = torch.nn.L1Loss().type(data_type)
        self.blur_loss = StdLoss().type(data_type)

    def _init_inputs(self):
        self.image_net_input = get_noise(self.input_depth, 'noise', (self.image.shape[1], self.image.shape[2]),
                                         var=1/10.).type(torch.cuda.FloatTensor).detach()

        self.mask_net_input = self.image_net_input

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_ambient()
        self._init_inputs()
        self._init_parameters()
        self._init_loss()

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
            if self.done:
                return
            optimizer.step()

    def _optimization_closure(self, step):
        """

        :param step: the number of the iteration

        :return:
        """
        image_net_input = self.image_net_input + (self.image_net_input.clone().normal_() / 30.)
        self.image_out = self.image_net(image_net_input)

        if isinstance(self.ambient_net, nn.Module):
            self.ambient_out = self.ambient_net()
        else:
            self.ambient_out = self.ambient_val
        self.mask_out = self.mask_net(self.mask_net_input)

        self.blur_out = self.blur_loss(self.mask_out)
        self.total_loss = self.mse_loss(self.mask_out * self.image_out + (1 - self.mask_out) * self.ambient_out,
                                 self.image_torch) + 0.01 * self.blur_out
        self.total_loss.backward(retain_graph=True)

    def _random_crop_optimization_closure(self, step):
        """

        :param step: the number of the iteration

        :return:
        """
        x, y = np.random.randint(0, self.image_torch.shape[2] - self.CROP_SIZE), np.random.randint(0, self.image_torch.shape[3] - self.CROP_SIZE)
        image_net_input = self.image_net_input + (self.image_net_input.clone().normal_() / 30.)
        # crops:
        image_net_input = image_net_input[:, :, x:x+self.CROP_SIZE, y:y+self.CROP_SIZE]
        mask_net_input = self.mask_net_input[:, :, x:x+self.CROP_SIZE, y:y+self.CROP_SIZE]
        image_torch = self.image_torch[:, :, x:x+self.CROP_SIZE, y:y+self.CROP_SIZE]

        self.image_out = self.image_net(image_net_input)

        if isinstance(self.ambient_net, nn.Module):
            self.ambient_out = self.ambient_net()
        else:
            self.ambient_out = self.ambient_val
        self.mask_out = self.mask_net(mask_net_input)

        self.blur_out = self.blur_loss(self.mask_out)
        self.total_loss = self.mse_loss(self.mask_out * self.image_out + (1 - self.mask_out) * self.ambient_out,
                                 image_torch) + 0.01 * self.blur_out
        self.total_loss.backward(retain_graph=True)

    def _reconstruct_all(self):
        print("Reconstructing...")
        image_net = self.image_net
        mask_net = self.mask_net
        d = {str(j):{str(i):[] for i in range(self.image_torch.shape[3])} for j in range(self.image_torch.shape[2])}
        self.final_image = np.zeros_like(self.image)
        self.final_t_map = np.zeros((1,self.image.shape[1], self.image.shape[2]))
        for x in range(0, self.image_torch.shape[2], self.CROP_SIZE // 2):
            for y in range(0, self.image_torch.shape[3], self.CROP_SIZE // 2):
                if y + self.CROP_SIZE > self.image_torch.shape[3]:
                    y = self.image_torch.shape[3] - self.CROP_SIZE
                if x + self.CROP_SIZE > self.image_torch.shape[2]:
                    x = self.image_torch.shape[2] - self.CROP_SIZE
                image_net_input = self.image_net_input[:, :, x:x + self.CROP_SIZE, y:y + self.CROP_SIZE]
                mask_net_input = self.mask_net_input[:, :, x:x + self.CROP_SIZE, y:y + self.CROP_SIZE]
                image_out = np.clip(torch_to_np(image_net(image_net_input)), 0, 1)
                mask_out = np.clip(torch_to_np(mask_net(mask_net_input)), 0, 1)
                for j in range(self.CROP_SIZE):
                    for i in range(self.CROP_SIZE):
                        one = image_out[:,j:j + 1, i: i + 1]
                        two = mask_out[:, j:j + 1, i: i + 1]
                        d[str(j + x)][str(i + y)].append((one, two))
        print("Assigning...")
        for x in progressbar.progressbar(range(0, self.image_torch.shape[2])):
            for y in range(0, self.image_torch.shape[3]):
                try:
                    self.final_image[:, x:x+1, y:y+1] = median([v[0] for v in d[str(x)][str(y)]])
                    self.final_t_map[:, x:x+1, y:y+1] = median([v[1] for v in d[str(x)][str(y)]])
                except ValueError as e:
                    print(e)
                    print(x, y)
                    raise e
        print("Done!")

    def _obtain_current_result(self):
        image_out_np = np.clip(torch_to_np(self.image_out), 0, 1)
        mask_out_np = np.clip(torch_to_np(self.mask_out), 0, 1)
        ambient_out_np = np.clip(torch_to_np(self.ambient_out), 0, 1)
        psnr = compare_psnr(self.image, mask_out_np * image_out_np + (1 - mask_out_np) * ambient_out_np)
        self.current_result = DehazeResult(learned=image_out_np, t=mask_out_np, a=ambient_out_np, psnr=psnr)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result

    def _plot_closure(self, step):
        """

         :param step: the number of the iteration

         :return:
         """
        print('Iteration %05d    Loss %f  %f current_psnr: %f max_psnr %f' % (step, self.total_loss.item(),
                                                                              self.blur_out.item(),
                                                                           self.current_result.psnr,
                                                                           self.best_result.psnr), '\r', end='')
        if step % self.show_every == self.show_every - 1:
            mask_out_np = self.t_matting(self.best_result.t)
            post = (self.image - ((1 - mask_out_np) * self.current_result.a)) / mask_out_np

            plot_image_grid("image_ambient", [self.current_result.learned,
                                              self.current_result.a * np.ones_like(self.current_result.learned)])
            plot_image_grid("mask", [mask_out_np, self.current_result.t])
            # original_image = t*image + (1-t)*A
            # image = (original_image - (1 - t) * A) * (1/t)
            plot_image_grid("fixed_image", [self.image, np.clip(post, 0, 1)])

    def finalize(self):
        self.final_image = np_imresize(self.best_result.learned, output_shape=self.original_image.shape[1:])
        self.final_t_map = np_imresize(self.best_result.t, output_shape=self.original_image.shape[1:])
        mask_out_np = self.t_matting(self.final_t_map)
        self.post = np.clip((self.original_image - ((1 - mask_out_np) * self.best_result.a)) / mask_out_np, 0, 1)
        # save_image(self.image_name + "_original", np.clip(self.original_image, 0, 1))
        # save_image(self.image_name + "_learned", self.final_image)
        save_image(self.image_name + "_t", mask_out_np)
        save_image(self.image_name + "_final", self.post)
        save_image(self.image_name + "_a", np.clip(self.best_result.a * np.ones_like(self.final_image), 0, 1))

    def t_matting(self, mask_out_np):
        refine_t = guidedFilter(self.original_image.transpose(1, 2, 0).astype(np.float32),
                                mask_out_np[0].astype(np.float32), 50, 1e-4)
        if self.clip:
            return np.array([np.clip(refine_t, 0.03, 1)])
        else:
            return np.array([np.clip(refine_t, 0, 1)])


def dehaze(image_name, image, num_iter=4000, plot_during_training=True,
           show_every=500,
           use_deep_channel_prior=True,
           gt_ambient=None):
    dh = Dehaze(image_name + "_0", image, num_iter, plot_during_training, show_every, use_deep_channel_prior,
                gt_ambient, clip=False)
    dh.optimize()
    dh.finalize()
    if use_deep_channel_prior:
        assert not gt_ambient
        gt_ambient = dh.best_result.a
        use_deep_channel_prior = False
    # for i in range(2):
    #     assert dh.post.shape == image.shape, (dh.post.shape, image.shape)
    #     dh = Dehaze(image_name + "_{}".format(i+1), dh.post, num_iter, plot_during_training, show_every,
    #                 use_deep_channel_prior, gt_ambient, clip=False)
    #     dh.optimize()
    #     dh.finalize()
    t = np.array([np.mean((image - dh.best_result.a) / (dh.post - dh.best_result.a), axis=0)])
    save_image(image_name + "_original", np.clip(image, 0, 1))
    save_image(image_name + "_t", np.clip(t, 0, 1))
    # save_image(image_name + "_final", dh.post)
    save_image(image_name + "_a",  np.clip(dh.best_result.a * np.ones_like(dh.post), 0, 1))