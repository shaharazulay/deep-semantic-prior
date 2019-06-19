import os.path
import glob

from segmentation import Segmentation
from utils.image_io import prepare_image, median, save_image


def segment_all(one_obj_path, output_path):
    for image in glob.glob(one_obj_path + "input/*"):
        name = image[len(one_obj_path + "input/"):-4]
        print("processing {}".format(name))
        #fg = image.replace("input/", "output_fg/").replace(".jpg", ".png")
        #bg = image.replace("input/", "output_bg/").replace(".jpg", ".png")
        masks = []
        im = prepare_image(image)
        #fg = prepare_image(fg)
        #bg = prepare_image(bg)
        fg = None
        bg = None
        for i in range(5):
            s = Segmentation("1obj_{}".format(i)+name,
                             im, bg_hint=bg, fg_hint=fg, plot_during_training=True, output_path=output_path)
            s.optimize()
            masks.append(s.best_result.mask)
        save_image("1obj_"+name+"_final_mask", median(masks), output_path)


if __name__ == "__main__":
    segment_all("./data/segmentation/", "./data/segmentation/output/")