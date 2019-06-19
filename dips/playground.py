from skimage.measure import compare_psnr

from net import *
from net.layers import weights_init
from net.noise import get_noise
from net.unet_model import UNet
from separation import mix_images
from utils.image_io import *

PSNR = []


def optimization_closure(iter_number, original_image_torch, image_net, original_input,
                         reg_noise_std, loss_function):
    """

    :param original_image_torch: the original image as pytorch tensor
    :param float reg_noise_std: additional noise for the input
    :param loss_function: the loss function for optimization
    :return: the loss, left output of the left net, right output of the right net
    """
    original_input = original_input
    image_out = image_net(original_input)
    total_loss = loss_function(image_out, original_image_torch)
    total_loss.backward()
    return total_loss, image_out


def plot_closure(iter_number, losses, image_out, original_image, show_every=1000):
    """

    :param iter_number: the number of the iteration
    :param losses: the total loss the the current iteration step
    :param original_image: original numpy image
    :param int show_every:

    :return:
    """
    global PSNR
    image_out_np = torch_to_np(image_out)
    psrn_gt = compare_psnr(original_image, image_out_np)
    PSNR.append(psrn_gt)
    print('Iteration {:5d}    Loss {:5f}   PSRN_gt: {:f}'.format(iter_number,
                                                                            losses.item(), psrn_gt),
          '\r', end='')
    if iter_number % show_every == 0:
        save_image('a_{}'.format(iter_number), image_out_np)


image2 = prepare_image("data/f16.png")
image1 = prepare_image("data/kate.png")
# image = prepare_image('data/separation/solid/m.jpg')
# im1 = prepare_image('data/separation/e.jpg')
# im2 = prepare_image('data/separation/c.jpg')

image = mix_images(image2, image1)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
data_type = torch.cuda.FloatTensor
input_type = 'noise'
# input_type = 'meshgrid'
pad = 'reflection'
optimize_over = 'net'

image1_torch = np_to_torch(image1).type(data_type)
image2_torch = np_to_torch(image2).type(data_type)


image_torch = np_to_torch(image).type(data_type)


input_depth = 3
image_net_input = get_noise(input_depth,
                            input_type,
                            (image.shape[1], image.shape[2])).type(data_type).detach()

learning_rate = 0.001

optimizer = 'adam'

reg_noise_std = 1. / 30.

num_iter = 500

loss_function = torch.nn.L1Loss().type(data_type)
# SAME TILL NOW

# image_net_input = image1_torch
# image_net = skip(
#     input_depth, 3,
#     num_channels_down=[8, 16, 16, 16, 16],
#     num_channels_up=[8, 16, 16, 16, 16],
#     num_channels_skip=[0, 0, 0, 0, 0],
#     upsample_mode='bilinear',
#     need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
image_net = UNet(input_depth, 3)

image_net = image_net.type(data_type)
# image_net.apply(weights_init)


parameters = get_params(optimize_over, image_net, image_net_input)


# loss_function = torch.nn.MSELoss().type(data_type)

optimize(optimizer, parameters, optimization_closure, plot_closure, learning_rate, num_iter,
         {'original_image_torch': image1_torch,
          'image_net': image_net,
          'original_input': image_net_input,
          'reg_noise_std': reg_noise_std,
          'loss_function': loss_function},
         {'original_image': image1,
          'show_every': 500})


print(image_net.x5.shape)
i = torch_to_np(image_net.x5)
i = i[:1, :, :]
i = i / np.max(i)
save_image("maube", i)
exit()
psnr1 = PSNR[:]

PSNR = []

# image_net_input = image2_torch
image_net = skip(
    input_depth, 3,
    num_channels_down=[8, 16, 16, 16, 16],
    num_channels_up=[8, 16, 16, 16, 16],
    num_channels_skip=[0, 0, 0, 0, 0],
    upsample_mode='bilinear',
    need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')


image_net = image_net.type(data_type)
image_net.apply(weights_init)

parameters = get_params(optimize_over, image_net, image_net_input)


optimize(optimizer, parameters, optimization_closure, plot_closure, learning_rate, num_iter,
         {'original_image_torch': image2_torch,
          'image_net': image_net,
          'original_input': image_net_input,
          'reg_noise_std': reg_noise_std,
          'loss_function': loss_function},
         {'original_image': image2,
          'show_every': 500})

psnr2 = PSNR[:]

# ----

PSNR = []

# image_net_input = image2_torch
image_net = skip(
    input_depth, 3,
    num_channels_down=[8, 16, 16, 16, 16],
    num_channels_up=[8, 16, 16, 16, 16],
    num_channels_skip=[0, 0, 0, 0, 0],
    upsample_mode='bilinear',
    need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')


image_net = image_net.type(data_type)
image_net.apply(weights_init)

parameters = get_params(optimize_over, image_net, image_net_input)

optimize(optimizer, parameters, optimization_closure, plot_closure, learning_rate, num_iter,
         {'original_image_torch': image_torch,
          'image_net': image_net,
          'original_input': image_net_input,
          'reg_noise_std': reg_noise_std,
          'loss_function': loss_function},
         {'original_image': image,
          'show_every': 500})

psnr = PSNR[:]

save_graphs('exp1', {'combined': psnr,
                     '1': psnr1,
                     '2': psnr2})