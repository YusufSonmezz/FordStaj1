from model_unet import UNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
import os
import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim
import cv2
import torch
import tqdm
import matplotlib.pyplot as plt

######### PARAMETERS ##########
valid_size = 0.3
test_size  = 0.1
batch_size = 10
epochs = 25
cuda = True
input_shape = (224, 224)
n_classes = 2
###############################


######### DIRECTORIES #########
SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
AUG_IMAGE = os.path.join(DATA_DIR, 'aug_images')
AUG_MASK = os.path.join(DATA_DIR, 'aug_masks')
###############################

# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

aug_image_path_list = glob.glob(os.path.join(AUG_IMAGE, '*'))
aug_mask_path_list  = glob.glob(os.path.join(AUG_MASK,  '*'))

# DATA CHECK
image_mask_check(image_path_list, mask_path_list)

# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]

# DIFFERENT

test_input_path_list = glob.glob(os.path.join(r'C:\Users\Yusuf\Desktop\p1_test\img', '*'))
test_input_path_list.sort()

test_label_path_list = glob.glob(os.path.join(r'C:\Users\Yusuf\Desktop\p1_test\mask', '*'))
test_label_path_list.sort()


#model = UNet(input_size=input_shape, n_classes=2)
#model.load_state_dict(torch.load(r"C:\Users\Yusuf\Desktop\Models\10.09.21(1).pth.pth"))
#model.eval()



### PREDICTION 

def predict(test_input_path_list, test_label_path_list, model, i_model):
    
    total_train = 0
    accuracy = 0.0
    for i in tqdm.tqdm(range(len(test_input_path_list))):
        
        
      batch_test_path_list = test_input_path_list[i: i + 1]
      batch_test_label_path_list = test_label_path_list[i: i + 1]
      batch_test_input = tensorize_image(batch_test_path_list, input_shape, cuda)
      batch_label = tensorize_mask(batch_test_label_path_list, input_shape, n_classes, cuda)
      
      result = model(batch_test_input)
      
      _, predicted = torch.max(result.data, 1)
      _, batch_label_2 = torch.max(batch_label, 1)
      total_train += batch_label.numel()
      accuracy += predicted.eq(batch_label_2.data).sum().item()
      
      result = result.argmax(axis = 1)
      result = result.cpu()
      result_np = result.detach().numpy()
      result_np = np.squeeze(result_np, axis = 0)
      
      test_image = cv2.imread(batch_test_path_list[0])
      #test_image = cv2.resize(test_image, (224, 224))
      result_np = cv2.resize(result_np.astype(np.uint8), (1920, 1208))
      
      copy_image = test_image.copy()
      copy_image[result_np == 1, :] = (255, 0, 125)
      opac_image=(copy_image/2+test_image/2).astype(np.uint8)
      predict_name = batch_test_path_list[0]
      predict_path = predict_name.replace("img", "predicts")
      cv2.imwrite(predict_path, opac_image.astype(np.uint8))
    return 100 * accuracy / total_train * 2
epoch = 1
accuracies = []
for i in range(epoch):
    model = torch.load(fr"C:\Users\Yusuf\Desktop\EarlyStoppingModels\23.09.21(2)\23.09.21(2)_{i}.pth")
    model.eval()
    accuracies.append(predict(test_input_path_list, test_label_path_list, model, i))
print()
best_model = 0
model_numb = 0
for i in range(epoch):
    if accuracies[i] >= best_model:
        best_model = accuracies[i]
        model_numb = i
    print(f'\nThe Accuracy of {i} model TEST data is % {accuracies[i]}')
print(f'BEST MODEL accuracy is {best_model}')
model_numb = 20
model = torch.load(fr"C:\Users\Yusuf\Desktop\EarlyStoppingModels\23.09.21(1)\23.09.21(1)_{model_numb}.pth")
model.eval()
predict(test_input_path_list, test_label_path_list, model, model_numb)