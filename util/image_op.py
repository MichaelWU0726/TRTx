import math

import cv2
import numpy as np


def resize_scale(image, ksize=512):
    h, w, _ = image.shape
    if h <= w:
        scale = ksize / w
        image_res = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        h_, w_ = image_res.shape[:2]
        image_border = cv2.copyMakeBorder(image_res, 0, ksize - h_, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    else:
        scale = ksize / h
        # w_ = math.ceil(w//scale)
        image_res = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        h_, w_ = image_res.shape[:2]
        image_border = cv2.copyMakeBorder(image_res, 0, 0, 0, ksize - w_, cv2.BORDER_CONSTANT, (0, 0, 0))
    return image_border, scale


def preprocess_image(image_path, ksize, dtype=np.float16):
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img, scale = resize_scale(img_raw, ksize=ksize)  # scale缩小比例
    img -= np.uint8([104, 117, 123])
    img = img.transpose(2, 0, 1)  # from HWC to CHW
    img = np.expand_dims(img, axis=0)
    if img.dtype != dtype:
        img = img.astype(dtype)
    return img, scale


def pre_extract(image, ksize=112, data_type=np.float32):
    if image.shape[0] != ksize:
        image = cv2.resize(image, (ksize, ksize))
    image = image.transpose(2, 0, 1)
    if image.dtype != data_type:
        image.astype(data_type)
    return image


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size
