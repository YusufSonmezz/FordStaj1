from constant import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import albumentations.pytorch
import os
import glob
import tqdm

#########
valid_size = 0.3
test_size = 0.1
#########

# Shows Images
def show_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Getting paths of images and masks
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

# Seperating valid and test data from whole dataset
indices = np.random.permutation(len(image_path_list))
test_indices  = int(len(indices) * test_size)
valid_test_indices = int(test_indices + len(indices) * valid_size)

train_path_list = image_path_list[valid_test_indices:]
label_path_list = mask_path_list[valid_test_indices:]

# Augmentation for IMAGES and MASKS
for Image in tqdm.tqdm(train_path_list):
    image = cv2.imread(Image)
    mask_path = Image.replace('images', 'masks').replace('jpg', 'png')
    mask  = cv2.imread(mask_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.CenterCrop(p = 1, height = image.shape[0], width = image.shape[1]),
        A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
        ])
    augmented_image = transform(image = image)['image']
    augmented_mask  = mask
    image_path_write = os.path.join(AUG_IMAGE, Image[-14:])
    mask_path_write = os.path.join(AUG_MASK, mask_path[-14:])
    cv2.imwrite(image_path_write, augmented_image.astype(np.uint8))
    cv2.imwrite(mask_path_write, augmented_mask.astype(np.uint8))
    


