import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


def normalize(img):
    normalized = img - np.mean(img)
    return normalized / np.std(img)


def read_images(img_name, template_name, data_dir):
    img = cv2.imread(os.path.join(data_dir, img_name), 0)
    template = cv2.imread(os.path.join(data_dir, template_name), 0)
    return img, template


def print_matching_result(image, matching_result, method):
    plt.subplot(212), plt.imshow(matching_result, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(211), plt.imshow(image, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(method)
    plt.show()
