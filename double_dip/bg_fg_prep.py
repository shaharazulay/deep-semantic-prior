import matplotlib.image as img
import glob

import numpy as np
from scipy.ndimage import filters, measurements, interpolation
from math import pi
import cv2

def image_histogram_equalization(image):
    return cv2.equalizeHist(image)

threshold = 15.5

for saliency in glob.glob("saliency/output_scaled/*.*"):
    print(saliency)
    # s = img.imread(saliency)
    s = cv2.imread(saliency,0)
    # print(s.shape) 
    image = image_histogram_equalization(s)
    print(image.max())
    image[image>255 - threshold] = 255
    image[image<=255 - threshold] = 0
    print(r"saliency/output_fg/" + saliency[len("saliency/output_scaled/"):])
    cv2.imwrite(r"saliency/output_fg/" + saliency[len("saliency/output_scaled/"):], image)
	
    image = image_histogram_equalization(s)
    print(image.max())
    v = np.zeros_like(image)
    v[image > threshold] = 0
    v[image <= threshold] = 255
    print(r"saliency/output_bg/" + saliency[len("saliency/output_scaled/"):])
    cv2.imwrite(r"saliency/output_bg/" + saliency[len("saliency/output_scaled/"):], v)
