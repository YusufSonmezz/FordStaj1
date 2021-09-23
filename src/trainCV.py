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
from sklearn.model_selection import KFold

def print_graph_2(loss_train, epoch, label):
    plt.plot(epoch, loss_train)
    plt.title(label)
    plt.show()

######### PARAMETERS ##########
valid_size = 0.15
test_size  = 0.15
batch_size = 8
epochs = 1
cuda = True
input_shape = (224, 224)
n_classes = 2
k_folds = 10
AUG_TRAIN = False
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

# For fold result
result = {}

# For model saving
models = {
    "train_loss": list(),
    "vall_loss": list()
    }

# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

if AUG_TRAIN:
    aug_image_path_list = glob.glob(os.path.join(AUG_IMAGE, '*'))
    aug_image_path_list.sort()
    aug_mask_path_list  = glob.glob(os.path.join(AUG_MASK,  '*'))
    aug_mask_path_list.sort()
    
    vip_aug_img = glob.glob(os.path.join(r'C:\Users\Yusuf\.spyder-py3\SadiEvrenSeker\Intership 1\data\vip_aug_img', '*'))
    vip_aug_mask = glob.glob(os.path.join(r'C:\Users\Yusuf\.spyder-py3\SadiEvrenSeker\Intership 1\data\vip_aug_mask', '*'))

    image_path_list += aug_image_path_list + vip_aug_img
    mask_path_list  += aug_mask_path_list + vip_aug_mask



# DATA CHECK
image_mask_check(image_path_list, mask_path_list)

# Define the K-fold Cross Validator
kfold = KFold(n_splits = k_folds, shuffle = True)


numb = 0
vall_loss = np.zeros((epochs * k_folds))
train_loss = np.zeros((epochs * k_folds))

train_accuracy = np.zeros((epochs * k_folds))
vall_accuracy = np.zeros((epochs * k_folds))
for fold, (train_ids, test_ids) in enumerate(kfold.split(image_path_list)):
    
    print(f'Fold {fold}')
    # TRAIN AND TEST SPLIT
    train_input_list = list(np.array(image_path_list)[train_ids])
    test_input_list = list(np.array(image_path_list)[test_ids])
    
    train_label_list = list(np.array(mask_path_list)[train_ids])
    test_label_list = list(np.array(mask_path_list)[test_ids])
    
    valid_input_list = test_input_list
    valid_label_list = test_label_list
    
    # DEFINE STEPS PER EPOCH
    steps_per_epoch = len(train_input_list) // batch_size
    
    # CALL MODEL
    model = torch.load(r"C:\Users\Yusuf\Desktop\Best Models among models\23.09.21(1)_20.pth")
    
    # DEFINE LOSS FUNCTION AND OPTIMIZER
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    
    # IF CUDA IS USED, IMPORT THE MODEL AND LOSS FUNCTION INTO CUDA
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    
    # TRAINING THE NEURAL NETWORK
    accuracy = 0
    total_train = 0
    for epoch in range(epochs):
        print()
        loss_value_graph = np.zeros((steps_per_epoch))
        epoch_value_graph = np.zeros((steps_per_epoch))
        running_loss = 0
        for ind in tqdm.tqdm(range(steps_per_epoch)):
            batch_input_path_list = train_input_list[batch_size*ind:batch_size*(ind+1)]
            batch_label_path_list = train_label_list[batch_size*ind:batch_size*(ind+1)]
            batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)
            batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)
            
            optimizer.zero_grad()
            model.zero_grad()
        
            outputs = model(batch_input)
            
            loss = criterion(outputs, batch_label)
            loss.backward()
            
            
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            _, batch_label_2 = torch.max(batch_label, 1)
            total_train += batch_label.numel()
            accuracy += (predicted.data == batch_label_2.data).sum().float()
            loss_float = loss.float()
            loss_value_graph[ind] = loss_float;
            epoch_value_graph[ind] = ind;
            running_loss += loss_float
            if ind == steps_per_epoch-1:
                train_loss[numb] = running_loss / steps_per_epoch
                result[fold] = 100 * accuracy / total_train * 2
                train_accuracy[numb] = 100 * accuracy / total_train * 2
                val_loss = 0
                accuracy = 0
                total_train = 0
                for (valid_input_path, valid_label_path) in zip(valid_input_list, valid_label_list):
                    batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                    batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                    outputs = model(batch_input)
                    loss = criterion(outputs, batch_label)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    _, batch_label_2 = torch.max(batch_label, 1)
                    total_train += batch_label.numel()
                    accuracy += predicted.eq(batch_label_2.data).sum().item()
                
                val_loss = val_loss / len(valid_input_list)
                print('\ntraining loss on epoch {}: {}'.format(epoch, running_loss / steps_per_epoch))
                print('validation loss on epoch {}: {}'.format(epoch, val_loss))
                print(f'\nThe Accuracy of TRAIN data is % {train_accuracy[numb]}')
                print(f'The Accuracy of VALIDATION data is % {100 * accuracy / total_train * 2}\n')
                model_name = '23.09.21(2)'
                vall_loss[numb] = val_loss
                vall_accuracy[numb] = 100 * accuracy / total_train * 2
                
                models['train_loss'].append(running_loss / steps_per_epoch)
                models['vall_loss'].append(val_loss)
                
        print_graph_2(loss_value_graph, epoch_value_graph, 'Epoch number {}'.format(numb))
        torch.save(model, fr'C:\Users\Yusuf\Desktop\EarlyStoppingModels\{model_name}\{model_name}_{numb}.pth')
        numb += 1

train_losses = models['train_loss']
val_losses = models['vall_loss']

for i in range(len(train_losses)):
    print(f'{i}  numarali epoch -> TRAIN LOSS..: {train_losses[i]}, VAL_LOSS..: {val_losses[i]}')


print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in result.items():
    print(f'Fold {key}: {value} %')
    sum += value
    print(f'Average: {sum/len(result.items())} %')

    































