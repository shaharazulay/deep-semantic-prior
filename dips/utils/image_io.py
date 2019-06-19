import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skvideo.io

matplotlib.use('agg')


def crop_image(img, d=32):
    """
    Make dimensions divisible by d

    :param pil img:
    :param d:
    :return:
    """

    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
        int((img.size[0] - new_size[0]) / 2),
        int((img.size[1] - new_size[1]) / 2),
        int((img.size[0] + new_size[0]) / 2),
        int((img.size[1] + new_size[1]) / 2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped


def crop_np_image(img_np, d=32):
    return torch_to_np(crop_torch_image(np_to_torch(img_np), d))


def crop_torch_image(img, d=32):
    """
    Make dimensions divisible by d
    image is [1, 3, W, H] or [3, W, H]
    :param pil img:
    :param d:
    :return:
    """
    new_size = (img.shape[-2] - img.shape[-2] % d,
                img.shape[-1] - img.shape[-1] % d)
    pad = ((img.shape[-2] - new_size[-2]) // 2, (img.shape[-1] - new_size[-1]) // 2)

    if len(img.shape) == 4:
        return img[:, :, pad[-2]: pad[-2] + new_size[-2], pad[-1]: pad[-1] + new_size[-1]]
    assert len(img.shape) == 3
    return img[:, pad[-2]: pad[-2] + new_size[-2], pad[-1]: pad[-1] + new_size[-1]]


def get_params(opt_over, net, net_input, downsampler=None):
    """
    Returns parameters that we want to optimize over.
    :param opt_over: comma separated list, e.g. "net,input" or "net"
    :param net: network
    :param net_input: torch.Tensor that stores input `z`
    :param downsampler:
    :return:
    """

    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params


def get_image_grid(images_np, nrow=8):
    """
    Creates a grid from a list of images by concatenating them.
    :param images_np:
    :param nrow:
    :return:
    """
    images_torch = [torch.from_numpy(x).type(torch.FloatTensor) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()


def plot_image_grid(name, images_np, output_path, interpolation='lanczos'):
    """
    Draws images in a grid

    Args:
        images_np: list of images, each image is np.array of size 3xHxW or 1xHxW
        nrow: how many images will be in one row
        interpolation: interpolation used in plt.imshow
    """
    assert len(images_np) == 2 
    #output_path = "/home/yossig/vision/results/"
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, 2)

    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

    plt.savefig(output_path + "{}.png".format(name))


def save_image(name, image_np, output_path):
    p = np_to_pil(image_np)
    # output_path = "/home/yossig/vision/results_small/"
    #output_path = "./data"
    # output_path = "/home/yossig/vision/results_yuval_800/"
    p.save(output_path + "{}.jpg".format(name))


def save_heatmap(name, image_np):
    cmap = plt.get_cmap('jet')

    rgba_img = cmap(image_np)
    rgb_img = np.delete(rgba_img, 3, 2)
    save_image(name, rgb_img.transpose(2, 0, 1))


def save_graph(name, graph_list, output_path):
    plt.clf()
    plt.plot(graph_list)
    #output_path = "/home/yossig/vision/results/"
    plt.savefig(output_path + name + ".png")


def save_graphs(name, graph_dict, output_path):
    """

    :param name:
    :param dict graph_dict: a dict from the name of the list to the list itself.
    :return:
    """
    plt.clf()
    fig, ax = plt.subplots()
    for k, v in graph_dict.items():
        ax.plot(v, label=k)
        # ax.semilogy(v, label=k)
    ax.set_xlabel('iterations')
    ax.set_ylabel(name)
    # ax.set_ylabel('L1-loss')
    # ax.set_ylabel('PSNR')
    #output_path = "/home/yossig/vision/results/"
    plt.legend()
    plt.savefig(output_path + name + ".png")


def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img


def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.

    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np


def prepare_image(file_name):
    """
    loads makes it divisible
    :param file_name:
    :return: the numpy representation of the image
    """
    img_pil = crop_image(get_image(file_name, -1)[0], d=32)
    return pil_to_np(img_pil)


def prepare_video(file_name):
    data = skvideo.io.vread("/home/yossig/vision/dpis/" + file_name)
    return crop_torch_image(data.transpose(0, 3, 1, 2).astype(np.float32) / 255.)


def save_video(name, video_np, output_path):
    #output_path = "/home/yossig/vision/results/"
    outputdata = video_np * 255
    outputdata = outputdata.astype(np.uint8)
    skvideo.io.vwrite(output_path + "{}.mp4".format(name), outputdata.transpose(0, 2, 3, 1))


def prepare_gray_image(file_name):
    img = prepare_image(file_name)
    return np.array([np.mean(img, axis=0)])


def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def median(img_np_list):
    """
    assumes C x W x H [0..1]
    :param img_np_list:
    :return:
    """
    assert len(img_np_list) > 0
    l = len(img_np_list)
    shape = img_np_list[0].shape
    result = np.zeros(shape)
    for c in range(shape[0]):
        for w in range(shape[1]):
            for h in range(shape[2]):
                result[c, w, h] = sorted(i[c, w, h] for i in img_np_list)[l//2]
    return result


def average(img_np_list):
    """
    assumes C x W x H [0..1]
    :param img_np_list:
    :return:
    """
    assert len(img_np_list) > 0
    l = len(img_np_list)
    shape = img_np_list[0].shape
    result = np.zeros(shape)
    for i in img_np_list:
        result += i
    return result / l


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]
