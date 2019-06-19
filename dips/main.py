import glob

from experiments.extending import SeparationExtendingExperiment
from segmentation import *
from separation import *
from separation.video_separation import ImageVideoSeparation, image_video_separation, image_video_separation_with_alpha, \
    alpha_video_video_separation, VideoVideoSeparation
from utils.image_io import prepare_image, median, save_image, prepare_gray_image, prepare_video
from utils.imresize import np_imresize
from watermarks.watermarks import Watermark, remove_watermark
from watermarks.many_watermarks import TwoImagesWatermark, ManyImagesWatermarkNoHint
import numpy as np

def separate_example():
    """
    runs a separation on specific two images
    :return:
    """
    # im1 = prepare_image('data/kate.png')
    # im2 = prepare_image('data/f16.png')
    # mixed = mix_images(im1, im2)
    
    # im1 = prepare_image('data/bear.jpg')
    # im2 = prepare_image('data/players.jpg')
    # mixed = prepare_image("data/separation/difference.bmp")

    # mixed, kernel, ratio = realistic_mix_images(im1, im2)

    # mixed = prepare_image('data/separation/postcard/ae-5-m-11.png')
    # mixed = prepare_image('data/separation/solid/m.jpg')

    # im1 = prepare_image('data/separation/g.jpg')
    # ----
    # im2 = prepare_image('data/separation/dorm1_input.png')
    #
    # s = Separation("dorm", im2, num_iter=4000)
    # s.optimize()
    # s.finalize()
    #
    # im2 = prepare_image('data/separation/dusk2_input.png')
    # im2 = np_imresize(im2, 0.5)
    #
    #
    # s = Separation("dusk", im2, num_iter=4000)
    # s.optimize()
    # s.finalize()
    im2 = prepare_image('data/separation/bus_station_input.png')

    s = Separation("bus_station", im2, num_iter=4000)
    s.optimize()
    s.finalize()

    im2 = prepare_image('data/separation/night3_input.png')
    im2 = np_imresize(im2, 0.5)


    s = Separation("night", im2, num_iter=4000)
    s.optimize()
    s.finalize()

    im2 = prepare_image('data/separation/dusk_input.png')

    s = Separation("dusj1", im2, num_iter=4000)
    s.optimize()
    s.finalize()



def experiment_example():
    im1 = prepare_image('data/experiments/texture3.jpg')
    im2 = prepare_image('data/experiments/texture1.jpg')
    mixed = (im1 + im2) / 2
    # mixed = prepare_gray_image('data/experiments/97033.jpg')
    # mixed = prepare_gray_image('data/separation/c.jpg')
    s = Separation("mixed", mixed, num_iter=8000)
    s.optimize()
    s.finalize()

def ambiguity_experiment_example():
    im1 = prepare_image('data/experiments/texture3.jpg')
    im2 = prepare_image('data/experiments/texture1.jpg')
    im3 = prepare_image('data/experiments/texture4.jpg')
    im4 = prepare_image('data/experiments/texture6.jpg')
    im1_new = im1
    im1_new[:,:, :im1.shape[2]//2] = im4[:,:, :im1.shape[2]//2]
    im2_new = im2
    # im4 = np_imresize(im4, output_shape=im2.shape)
    im2_new[:, :, :im2.shape[2] // 2] = im3[:,:, :im2.shape[2] // 2]
    save_image("input1", im1_new)
    save_image("input2", im2_new)
    mixed =(im1_new + im2_new) / 2
    save_image("mixed", mixed)
    exit()
    for i in range(10):
        # mixed = prepare_gray_image('data/experiments/97033.jpg')
        # mixed = prepare_gray_image('data/separation/c.jpg')
        s = Separation("mixed_{}".format(i), mixed, num_iter=8000)
        s.optimize()
        s.finalize()


def segment_example():
    # for i in range(1, 10):
    im = prepare_image('data/segmentation/zebra.png')
    fg = prepare_image('data/segmentation/zebra_fg - Copy.png')
    bg = prepare_image('data/segmentation/zebra_bg - Copy.png')
    # fg = prepare_image('data/segmentation/zebra_5_mask.bmp')
    # bg = 1 - prepare_image('data/segmentation/zebra_5_mask.bmp')
    # fg = prepare_image('data/segmentation/zebra_saliency.bmp')
    # fg[fg > 0.9] = 1
    # fg[fg <= 0.9] = 0
    # bg = 1 - prepare_image('data/segmentation/zebra_saliency.bmp')
    # bg[bg > 0.9] = 1
    # bg[bg <= 0.9] = 0
    s = Segmentation("zebra_{}".format(1), im, bg_hint=bg, fg_hint=fg)
    s.optimize()
    s.finalize()

    # im = prepare_image('data/segmentation/sheep.jpg')
    # fg = prepare_image('data/segmentation/sheep_fg.png')
    # bg = prepare_image('data/segmentation/sheep_bg.png')
    #
    # s = Segmentation("sheep", im, bg_hint=bg, fg_hint=fg)
    # s.optimize()
    # s.finalize()


    # im = prepare_image('data/segmentation/yaks.jpg')
    # fg = prepare_image('data/segmentation/yaks_fg.png')
    # bg = prepare_image('data/segmentation/yaks_bg.png')
    #
    # s = Segmentation("yaks", im, step_num=2, bg_hint=bg, fg_hint=fg)
    # s.optimize()
    # s.finalize()
    #
    # im = prepare_image('data/segmentation/pagoda.jpg')
    # fg = prepare_image('data/segmentation/pagoda_fg.png')
    # bg = prepare_image('data/segmentation/pagoda_bg.png')
    # s = Segmentation("pagoda", im, step_num=2, bg_hint=bg, fg_hint=fg)
    # s.optimize()
    # s.finalize()

    # im = prepare_image('data/elephant.jpg')
    # im = prepare_image('data/segmentation/pagoda.jpg')
    # im = prepare_image('data/segmentation/361010.jpg')
    # im = prepare_image('data/segmentation/image014.jpg')
    # im = prepare_image('data/segmentation/demo.png')
    # im = prepare_image('data/segmentation/image005.png')
    # im = prepare_image('data/segmentation/image015.png')
    # im = prepare_image('data/segmentation/img_1029.jpg')
    # im = np.clip(imresize(im.transpose(1, 2, 0), 0.5).transpose(2, 0, 1), 0, 1)
    # bg = np.clip(imresize(bg.transpose(1, 2, 0), 0.5).transpose(2, 0, 1), 0, 1)
    # fg = np.clip(imresize(fg.transpose(1, 2, 0), 0.5).transpose(2, 0, 1), 0, 1)
    # segment(im)
    # uneven_segment(im, show_every=500)
    # multiscale_segment(im)
    # uneven_multiscale_segment(im)

def dehazing_exmaple():
    # im = prepare_image('data/dehazing/forest.png')
    # im = prepare_image('data/dehazing/tiananmen.png')
    im = prepare_image('data/dehazing/cityscape.png')
    # im = prepare_image('data/dehazing/dubai.png')
    # im = prepare_image('data/dehazing/mountain.png')
    # im = prepare_image('data/dehazing/underwaterWaterTank.jpg')
    # dehaze(im, use_deep_channel_prior=True)


def watermark_example():

    # im = prepare_image('data/watermark/fotolia.jpg')
    # fg = prepare_image('data/watermark/fotolia_watermark.png')
    # remove_watermark("fotolia", im, fg)
    #
    # im = prepare_image('data/watermark/copyright.jpg')
    # fg = prepare_image('data/watermark/copyright_watermark.png')
    # remove_watermark("copyright", im, fg)
    #
    # im = prepare_image('data/watermark/small_portubation.jpg')
    # fg = prepare_image('data/watermark/small_portubation_watermark.png')
    # remove_watermark("small_portubation", im, fg)

    # im = prepare_image('data/watermark/cvpr1.jpg')
    # fg = prepare_image('data/watermark/cvpr1_watermark.png')
    # remove_watermark("cvpr1", im, fg)
    #
    # im = prepare_image('data/watermark/cvpr2.jpg')
    # fg = prepare_image('data/watermark/cvpr2_watermark.png')
    # remove_watermark("cvpr2", im, fg)

    # im = prepare_image('data/watermark/coco.jpg')
    # fg = prepare_image('data/watermark/coco_watermark.png')
    # remove_watermark("coco", im, fg)
    #
    # im = prepare_image('data/watermark/coco2.jpg')
    # fg = prepare_image('data/watermark/coco2_watermark.png')
    # remove_watermark("coco2", im, fg)
    # im = prepare_image('data/watermark/cvpr3.jpg')
    # fg = prepare_image('data/watermark/cvpr3_watermark.png')
    # remove_watermark("cvpr3", im, fg)

    # im = prepare_image('data/watermark/cvpr4.jpg')
    # fg = prepare_image('data/watermark/cvpr4_watermark.png')
    # remove_watermark("cvpr4", im, fg)
    # im = prepare_image('data/watermark/AdobeStock1.jpg')
    # fg = prepare_image('data/watermark/AdobeStock1_watermark.png')
    # remove_watermark("AdobeStock1", im, fg)
    # im = prepare_image('data/watermark/AdobeStock2.jpg')
    # fg = prepare_image('data/watermark/AdobeStock2_watermark.png')
    # remove_watermark("AdobeStock2", im, fg)
    # im = prepare_image('data/watermark/AdobeStock3.jpg')
    # fg = prepare_image('data/watermark/AdobeStock3_watermark.png')
    # remove_watermark("AdobeStock3", im, fg)
    # im = prepare_image('data/watermark/AdobeStock4.jpg')
    # fg = prepare_image('data/watermark/AdobeStock4_watermark.png')
    # remove_watermark("AdobeStock4", im, fg)
    im = prepare_image('data/watermark/AdobeStock5.jpg')
    fg = prepare_image('data/watermark/AdobeStock5_watermark.png')
    remove_watermark("AdobeStock5", im, fg)


def watermark2_example():
    im1 = prepare_image('data/watermark/fotolia1.jpg')
    im2 = prepare_image('data/watermark/fotolia2.jpg')
    fg = prepare_image('data/watermark/fotolia_many_watermark.png')
    results = []
    for i in range(7):
        # TODO: make it median
        s = TwoImagesWatermark("fotolia_example_{}".format(i), im1, im2, step_num=2, watermark_hint=fg)
        s.optimize()
        s.finalize()


def watermarks2_example_no_hint():
    # im1 = prepare_image('data/watermark/123RF_1.jpg')
    # im2 = prepare_image('data/watermark/123RF_2.jpg')
    # im3 = prepare_image('data/watermark/123RF_3.jpg')
    # im4 = prepare_image('data/watermark/123RF_4.jpg')
    # results = []
    # for i in range(7):
    #     # TODO: make it median
    #     s = ManyImagesWatermarkNoHint(["123rf_example_{}".format(i) for i in range(4)], [im1, im2, im3, im4])
    #     s.optimize()
    #     s.finalize()

    im1 = prepare_image('data/watermark/fotolia1.jpg')
    im2 = prepare_image('data/watermark/fotolia2.jpg')
    im3 = prepare_image('data/watermark/fotolia3.jpg')
    results = []
    for i in range(5):
        # TODO: make it median
        s = ManyImagesWatermarkNoHint(["fotolia_example_{}".format(i) for i in range(3)], [im1, im2, im3])
        s.optimize()
        results.append(s.best_result)
    # namedtuple("ManyImageWatermarkResult", ['cleans', 'mask', 'watermark', 'psnr'])
    obtained_watermark = median([result.mask * result.watermark for result in results])
    obtained_im1 = median([result.cleans[0] for result in results])
    obtained_im2 = median([result.cleans[1] for result in results])
    obtained_im3 = median([result.cleans[2] for result in results])
    # obtained_mask = median([result.mask for result in results])
    v = np.zeros_like(obtained_watermark)
    v[obtained_watermark < 0.03] = 1
    final_im1 = v * im1 + (1 - v) * obtained_im1
    final_im2 = v * im2 + (1 - v) * obtained_im2
    final_im3 = v * im3 + (1 - v) * obtained_im3
    save_image("fotolia1_final", final_im1)
    save_image("fotolia2_final", final_im2)
    save_image("fotolia3_final", final_im3)
    save_image("fotolia_final_watermark", obtained_watermark)
    # TODO: for watermark - zero everything under 0.03


def two_extending_experiment():
    im1 = prepare_image('data/kate.png')
    im2 = prepare_image('data/f16.png')
    t = SeparationExtendingExperiment("kate_f16", im1, im2, 2000, True)
    t.optimize()
    t.finalize()


def separate_image_video_example():
    # vid = prepare_video('data/separation/vid.avi')
    # vid = prepare_video('data/separation/half_horses.mp4')
    vid = prepare_video('data/separation/fountain_short.mp4')
    im = prepare_image('data/separation/d.jpg')
    im = np_imresize(im, output_shape=vid.shape[2:])
    mix = 0.5 * im + 0.5 * vid
    image_video_separation("tiger", mix)

    vid = prepare_video('data/separation/fountain_short.mp4')
    im = prepare_image('data/separation/f.jpg')
    im = np_imresize(im, output_shape=vid.shape[2:])
    mix = 0.5 * im + 0.5 * vid
    image_video_separation("misg", mix)

    vid = prepare_video('data/separation/horses_short.mp4')
    im = prepare_image('data/separation/g.jpg')
    im = np_imresize(im, output_shape=vid.shape[2:])
    mix = 0.5 * im + 0.5 * vid
    image_video_separation("cow", mix)


def separate_video_video_example():
    vid = prepare_video('data/separation/fountain_horses.mp4')
    s = VideoVideoSeparation("fountain_horses", vid)
    s.optimize()
    s.finalize()

def separate_alpha_video_example():
    vid = prepare_video('data/separation/vid.avi')
    image_video_separation_with_alpha("video_alpha", vid)


def separate_alpha_video_video_example():
    vid = prepare_video('data/separation/vid.avi')
    alpha_video_video_separation("video_alpha", vid)


def main():
    # bsd_experiment()
    # obj1_experiment()
    # obj1_experiment_extended()
    # separate_image_video_example()
    # separate_alpha_video_example()
    # separate_alpha_video_video_example()
    # haze_experiment()
    # patch_experiment()
    # extending_experiment()
    # two_extending_experiment()
    # transparency_experiment()
    # watermark_example()
    # watermark2_example()
    # watermarks2_example_no_hint()
    ambiguity_experiment_example()
    # segment_example()
    # separate_video_video_example()
    # entropy_experiment()
    # separate_example()
    # experiment_example()
    # cosegement_example()
    # deraining_example()
    # deblurring_example()
    # dehazing_exmaple()


if __name__ == "__main__":
    # arg_parser = get_arg_parser()
    # args = arg_parser.parse_args()
    main()