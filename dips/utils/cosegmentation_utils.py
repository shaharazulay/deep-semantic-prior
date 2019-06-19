import numpy as np
from skimage.measure import compare_psnr
from torch import nn
from net.downsampler import *
from net.layers import bn, GrayscaleLayer
from utils.image_io import torch_to_np, plot_image_grid, crop_image, get_image, pil_to_np




def cosegmentation_optimization_closure(iter_number, original_images_torch, same_net, other_nets,
                                      mask_nets, net_inputs, reg_noise_std,
                                      loss_function, blur_function, gngc_loss):
    """
    :param original_images_torch: the original image as pytorch tensor
    :param mask_nets: the mask net
    :param net_inputs: the inputs
    :param float reg_noise_std: additional noise for the input
    :param loss_function: the loss function for optimization
    :return: the loss, left output of the left net, right output of the right net
    """
    net_inputs = [net_input + net_input.clone().normal_() * reg_noise_std for net_input in net_inputs]

    mask_outs = [mask_net(net_input) for net_input, mask_net in zip(net_inputs, mask_nets)]
    other_outs = [other_net(net_input) for net_input, other_net in zip(net_inputs, other_nets)]
    same_outs = [same_net(net_input) for net_input in net_inputs]
    gngc = sum([gngc_loss(same_out, other_out) for same_out, other_out in zip(same_outs, other_outs)])
    blur = sum([blur_function(mask_out) for mask_out in mask_outs])
    mse = sum([loss_function(mask_out * same_out +(1 - mask_out) *  other_out, original_image_torch)
                    for mask_out, other_out, same_out, original_image_torch in zip(mask_outs, other_outs,
                                                                                   same_outs, original_images_torch)])

    total_loss = mse + 0.001 * gngc +  0.0001 * blur
    total_loss.backward(retain_graph=True)

    return (total_loss, gngc), other_outs, mask_outs, same_outs


def cosegmentation_plot_closure(iter_number, losses, other_outs, mask_outs, same_outs, original_images, show_every=1000):
    """

    :param iter_number: the number of the iteration
    :param int show_every:

    :return:
    """
    # TODO: handle other ratios
    other_outs_np = [torch_to_np(other_out) for other_out in other_outs]
    same_outs_np = [torch_to_np(same_out) for same_out in same_outs]
    mask_outs_np = [torch_to_np(mask_out) for mask_out in mask_outs]
    print(('Iteration %05d    Loss ' + (" %f " * len(losses))) % (
        iter_number, *[l.item() for l in losses]), '\r', end='')
    if iter_number % show_every == 0:
        for i, (other_out_np, same_out_np, mask_out_np) in enumerate(zip(other_outs_np, same_outs_np, mask_outs_np)):
            plot_image_grid("segment_{}_{}".format(iter_number, i),
                            [np.clip(other_out_np, 0, 1),
                             np.clip(same_out_np, 0, 1)])
            plot_image_grid("mask_{}_{}".format(iter_number, i),
                            [np.clip(mask_out_np, 0, 1), np.clip(1 - mask_out_np, 0, 1)])

