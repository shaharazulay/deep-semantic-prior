from dehazing import *
from dehazing.dehazing import dehaze
from utils.image_io import prepare_image, get_image, pil_to_np, crop_np_image, save_image
from glob import glob
import os.path
import scipy.io as sio
import numpy as np

from utils.imresize import imresize


def full_path_to_name(path):
    return path.split("/")[-1].split(".")[0]


def get_ambients_dict(ambients_path):
    """
    returns a dict from image_name to the ambient
    :param ambients_path:
    :return:
    """
    ambients_dict = {}
    path = os.path.join(ambients_path, "estimated_airlight.mat")
    estimated_airlights = sio.loadmat(path)["estimatedAirlights"]
    image_files = sio.loadmat(os.path.join(ambients_path, "image_files.mat"))["imageFiles"]
    for f, estimated in zip(image_files, estimated_airlights):
        ambients_dict[f[0][0][0].split(".")[0]] = estimated
    return ambients_dict

def get_ambients_dict3(ambients_path):
    """
    returns a dict from image_name to the ambient
    :param ambients_path:
    :return:
    """
    ambients_dict = {}
    path = os.path.join(ambients_path, "OHAZE_airlights_2016_ICCP_Bahat.mat")
    estimated_airlights = sio.loadmat(path)["airlights"][0]
    for f in estimated_airlights:
        ambients_dict[f[0][0]] = f[1][0]
    return ambients_dict


def run_on_benchmark2(benchmark_path, ambient_path):
    """

    :param benchmark_path:
    :param ambient_path:
    :return:
    """
    ambients = get_ambients_dict(ambient_path)
    for image_path in glob(os.path.join(benchmark_path, "*")):
        image_name = full_path_to_name(image_path)
        print("Processing", image_name)
        image_pil = get_image(image_path, -1)[0]
        image = pil_to_np(image_pil)
        if image_name in ambients:
            print("Found ambient!")
            d = dehaze(image_name, image, 8000, plot_during_training=False, show_every=40001,
                       use_deep_channel_prior=False, gt_ambient=ambients[image_name])
            # # else:
            # d = dehaze(image_name, image, 4000, plot_during_training=False, show_every=40001,
            #            use_deep_channel_prior=True, gt_ambient=None)


def run_on_benchmark_right_yuval(benchmark_path, ambient_path):
    """

    :param benchmark_path:
    :param ambient_path:
    :return:
    """
    ambients = get_ambients_dict3(ambient_path)
    for image_path in glob(os.path.join(benchmark_path, "*")):
        image_name = full_path_to_name(image_path)
        print("Processing", image_name)
        image_pil = get_image(image_path, -1)[0]
        image = pil_to_np(image_pil)
        if image_name in ambients:
            print("Found ambient!")
            dehaze(image_name, image, 8000, plot_during_training=False, show_every=40001,
                       use_deep_channel_prior=False, gt_ambient=ambients[image_name])




def run_on_benchmark1(benchmark_path, ambient_path):
    """

    :param benchmark_path:
    :param ambient_path:
    :return:
    """
    ambients = get_ambients_dict(ambient_path)
    for image_path in glob(os.path.join(benchmark_path, "*")):
        image_name = full_path_to_name(image_path)
        print("Processing", image_name)
        image = prepare_image(image_path)

        if image_name in ambients:
            print("Found ambient!")
            d = dehaze(image_name, image, 8000, plot_during_training=False, show_every=40001,
                       use_deep_channel_prior=False, gt_ambient=ambients[image_name])
        else:
            d = dehaze(image_name, image, 4000, plot_during_training=False, show_every=40001,
                       use_deep_channel_prior=False, gt_ambient=None)


if __name__ == "__main__":
    # run_on_benchmark1(r"/home/yossig/data/dehazing", r"/home/yossig/vision/dehazing_ambient")
    # run_on_benchmark2(r"/home/yossig/data/ohazy/hazy", r"/home/yossig/vision/dehazing_ambient_ohaze")
    run_on_benchmark_right_yuval(r"/home/yossig/data/ohazy/hazy800bmp", r"/home/yossig/vision/dehazing_ambient_ohaze")