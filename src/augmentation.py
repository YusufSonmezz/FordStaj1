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

Cone = False
Bridge = False
Tunel = False
All = True
#########

# Shows Images
def show_image(image):
    cv2.imshow("Image", cv2.resize(image, (500, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Getting paths of images and masks
if Cone:
    name = 'Cone'
elif Bridge:
    name = 'Bridge'
elif Tunel:
    name = 'Tunel'

if All == False:
    image_path_list = glob.glob(os.path.join(fr'C:\Users\Yusuf\Desktop\Aug_Datas\{name}', '*'))
    mask_path_list = glob.glob(os.path.join(fr'C:\Users\Yusuf\Desktop\aug_mask\{name}_mask', '*'))
else:
    image_path_list = glob.glob(os.path.join(fr'C:\Users\Yusuf\Desktop\Data_1\Images', '*'))
    mask_path_list = glob.glob(os.path.join(fr'C:\Users\Yusuf\Desktop\Data_1\Masks', '*'))
    
image_path_list.sort()
mask_path_list.sort()

# Seperating valid and test data from whole dataset
indices = np.random.permutation(len(image_path_list))
test_indices  = int(len(indices) * test_size)
valid_test_indices = int(test_indices + len(indices) * valid_size)

#train_path_list = image_path_list[valid_test_indices:]
#label_path_list = mask_path_list[valid_test_indices:]

# Augmentation for IMAGES and MASKS
for Image, Mask in zip(tqdm.tqdm(image_path_list), mask_path_list):
    image = cv2.imread(Image)
    mask  = cv2.imread(Mask)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if Cone:
        transform_image = A.Compose([
            A.HorizontalFlip(p = 1),
            A.ColorJitter(brightness = 0.5, p = 1)
            ])
        transform_mask = A.Compose([
            A.HorizontalFlip(p = 1),
            ])
        augmented_image = transform_image(image = image)['image']
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        augmented_mask = transform_mask(image = mask)['image']
    elif Bridge:
        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.4, p=0.9),
            ])
        augmented_image = transform(image = image)['image']
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        augmented_mask = mask
    elif Tunel:
        transform = A.Compose([
            A.HorizontalFlip(p = 1),
            ])
        augmented_image = transform(image = image)['image']
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        augmented_mask = transform(image = mask)['image']
    elif All:
        transform= A.Compose([
                #A.ColorJitter(brightness = 0.6, contrast = 0.6, p = 1),
                #A.HorizontalFlip(p = 1),
                #A.GaussNoise(var_limit = (10.0, 70.0), p = 1),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=4, p = 1)
            ])
        transform_mask = A.Compose([
            A.HorizontalFlip(p = 1),
            ])
        augmented_image = transform(image = image)['image']
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        #augmented_mask = transform_mask(image = mask)['image']
        augmented_mask = mask
    
    if All == False:
        image_path_write = os.path.join(AUG_IMAGE, Image[-14:-4] + fr'_{name}.jpg')
        mask_path_write = os.path.join(AUG_MASK, Mask[-14:-4] + fr'_{name}.png')
    else:
        image_path_write = os.path.join(AUG_IMAGE, Image[-14:-4] + fr'_2.jpg')
        mask_path_write = os.path.join(AUG_MASK, Mask[-14:-4] + fr'_2.png')
    
    cv2.imwrite(image_path_write, augmented_image.astype(np.uint8))
    cv2.imwrite(mask_path_write, augmented_mask.astype(np.uint8))
    


