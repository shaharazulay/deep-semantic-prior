import numpy as np
from skimage.measure import compare_psnr
from torch import nn
from utils.image_io import torch_to_np, plot_image_grid, crop_image, get_image, pil_to_np


def deblurring_optimization_closure(iter_number, original_image_torch, image_net,
                                    image_net_original_input,
                                    reg_noise_std, loss_function, blur_layer):
    """

    :param original_image_torch: the original image as pytorch tensor
    :param image_net: the left net
    :param blur_layer: the learnable bluring layer
    :param image_net_original_input: the original noise for the left net
    :param float reg_noise_std: additional noise for the input
    :param loss_function: the loss function for optimization
    :return: the loss, left output of the left net, right output of the right net
    """
    if reg_noise_std > 0:
        image_net_input = image_net_original_input + (image_net_original_input.clone().normal_() * reg_noise_std)
    else:
        image_net_input = image_net_original_input

    image_out = image_net(image_net_input)

    # blurred_image = image_out 
    blurred_image = blur_layer(image_out)
    # TODO: make the image consistent in it colors (not too dark) 
    total_loss = loss_function(blurred_image, original_image_torch)

    total_loss.backward(retain_graph=True)

    return total_loss, image_out, blurred_image, blur_layer.mask


def deblurring_plot_closure(iter_number, total_loss, image_out, blurred_image, blur_mask, original_image, show_every=1000):
    """

    :param iter_number: the number of the iteration
    :param total_loss: the total loss the the current iteration step
    :param image_out: torch tensor left output
    :param original_image: original numpy image
    :param blur_mask: the learned torch mask
    :param blurred_image: the blurred image that is obtained
    :param int show_every:

    :return:
    """
    image_out_np = torch_to_np(image_out)
    blurred_image_np = torch_to_np(blurred_image)
    psrn_gt = compare_psnr(original_image, blurred_image_np)
    print('Iteration %05d    Loss %f  PSRN_gt: %f ' % (iter_number, total_loss.item(), psrn_gt), '\r', end='')
    if iter_number % show_every == 0:
        plot_image_grid([np.clip(image_out_np, 0, 1),
                         np.clip(original_image, 0, 1)], 4, 5)
        plot_image_grid([np.clip(original_image, 0, 1),
                         np.clip(blurred_image_np, 0, 1)], 4, 5)
