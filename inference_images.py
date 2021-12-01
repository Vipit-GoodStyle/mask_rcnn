import cv2
import os
import sys
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn.config import Config
from mrcnn.visualize import *
import mrcnn.model as modellib
from mrcnn.model import MaskRCNN
import uuid
import argparse
import skimage
import colorsys
import tensorflow as tf
import numpy as np
import shutil
import random
import argparse
from skimage import io
import uuid

NUM_CATS = 46
IMAGE_SIZE = 512


class FashionConfig(Config):
    NAME = "fashion"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = NUM_CATS + 1
    BACKBONE = 'resnet50'

    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE
    IMAGE_RESIZE_MODE = 'none'
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)


config = FashionConfig()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
ap.add_argument("-o", "--output", required=True)
args = vars(ap.parse_args())

input_image_dir = args["input"]
output_image_dir = args["output"]

model = modellib.MaskRCNN(mode="inference", config=config, model_dir=config.MODEL_DIR)
model.load_weights(config.MODEL_PATH, by_name=True)


def resize_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img


def refine_masks(masks, rois):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m] == True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois


for filename in os.listdir(input_image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        image_path = os.path.join(input_image_dir, filename)
        print(image_path)
        img = io.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = model.detect([resize_image(image_path)])
        r = result[0]
        if r['masks'].size > 0:
            masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
            for m in range(r['masks'].shape[-1]):
                masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'),
                                            (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            y_scale = img.shape[0] / IMAGE_SIZE
            x_scale = img.shape[1] / IMAGE_SIZE
            rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)

            masks, rois = refine_masks(masks, rois)
        else:
            masks, rois = r['masks'], r['rois']

        save_masked_instances(output_image_dir, img, rois, masks, r['class_ids'], ['bg'] + config.LABEL_NAMES, r['scores'],
                              title='image_id',
                              figsize=(16, 16))
