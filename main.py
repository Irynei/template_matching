import cv2
import numpy as np
from utils import (
    print_matching_result,
    normalize,
    read_images
)

DATA_DIR = 'data'


def template_matching_opencv(img, template, methods=[]):
    """
    template matching using OpenCV lib. Plot matching result for every method specified in `methods`
    :param img: input image
    :param template: input template
    :param methods: list of opencv methods that are used in template matching
    """
    h, w = template.shape
    for method in methods:
        img_copy = img.copy()
        result = cv2.matchTemplate(img_copy, template, eval(method))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = min_loc if method == 'cv2.TM_SQDIFF' else max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img_copy, top_left, bottom_right, 255, 2)
        print_matching_result(img_copy, result, method)


def ssd(a, b):
    """ Sum of Square Distances """
    return np.sum(np.square(a - b))


def sad(a, b):
    """ Sum of Absolute Values """
    return np.sum(np.absolute(a - b))


def ncc(a, b):
    """ Normalized Cross-Correlation. `a` and `b` must be normalized """
    return np.sum(np.multiply(a, b))


def template_matching_numpy(img, template, methods):
    """
    Template matching using only numpy. Plot matching result for every method specified in `methods`
    :param img: input image
    :param template: input template
    :param methods: dict of method names as keys and specific params as values
    """
    image_h, image_w = image.shape
    template_h, template_w = template.shape
    default_result = np.zeros((image_h - template_h, image_w - template_w))
    results = {k: np.copy(default_result) for k in methods.keys()}
    # normalize images
    img_normalized = normalize(img)
    template_normalized = normalize(template)

    for i in range(image_h - template_h):
        for j in range(image_w - template_w):
            # for each provided method compute corresponding metric and store in results
            for method, value in methods.items():
                if value['normalized']:
                    image_patch = img_normalized[i:(i + template_h), j:(j + template_w)]
                    image_template = template_normalized
                else:
                    image_patch = img[i:(i + template_h), j:(j + template_w)]
                    image_template = template

                results[method][i, j] = value['func'](image_patch, image_template)

    # for each result get max(or min) and plot corresponding matching result
    for method, res in results.items():
        im_copy = img.copy()
        if method != 'Normalized Cross-Correlation':
            top_left = np.unravel_index(res.argmin(), res.shape)[::-1]
        else:
            top_left = np.unravel_index(res.argmax(), res.shape)[::-1]

        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
        cv2.rectangle(im_copy, top_left, bottom_right, 255, 2)
        print_matching_result(im_copy, res, method)


if __name__ == '__main__':
    image, template = read_images('0001.jpg', '0001_template.jpg', DATA_DIR)
    opencv_methods = ['cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF']
    custom_methods = {
        'Sum of Absolute Distances': {'func': sad, 'normalized': False},
        'Normalized Sum of Absolute Distances': {'func': sad, 'normalized': True},
        'Sum of Square Distances': {'func': ssd, 'normalized': False},
        'Normalized Cross-Correlation': {'func': ncc, 'normalized': True}
    }
    # template_matching_opencv(image, template, opencv_methods)
    template_matching_numpy(image, template, custom_methods)
