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
import torch.nn.functional as F

def print_graph_2(loss, epoch, label):
    plt.plot(epoch, loss)
    plt.title(label)
    plt.show()

######### PARAMETERS ##########
valid_size = 0.3
test_size  = 0.1
batch_size = 10
epochs = 30
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

# SLICE AUGMENTATION DATASET TO ADD TRAIN DATASET
"""
aug_half = int(len(aug_image_path_list) / 2)
train_input_path_list = aug_image_path_list[:aug_half] + train_input_path_list[:] + aug_image_path_list[aug_half:]
train_label_path_list = aug_mask_path_list[:aug_half] + train_label_path_list[:] + aug_mask_path_list[aug_half:]
"""
# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list)//batch_size

# CALL MODEL
#model = UNet(input_size=input_shape, n_classes=2)
model = torch.load(r"C:\Users\Yusuf\Desktop\Models\09.08.21(2).pth")

# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.004, momentum = 0.9, nesterov=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

# TRAINING THE NEURAL NETWORK
vall_loss = np.zeros((epochs))
train_loss = np.zeros((epochs))
epoch_np = np.zeros((epochs))
for epoch in range(epochs):
    epoch_np[epoch] = epoch
    print()
    loss_value_graph = np.zeros((steps_per_epoch))
    epoch_value_graph = np.zeros((steps_per_epoch))
    running_loss = 0
    for ind in tqdm.tqdm(range(steps_per_epoch)):
        batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)
        batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)
        
        optimizer.zero_grad()
        
        outputs = model(batch_input)
        
        loss = criterion(outputs, batch_label)
        loss.backward()
        
        optimizer.step()
        
        loss_value_graph[ind] = loss.item();
        epoch_value_graph[ind] = ind;
        running_loss += loss.item()
        if ind == steps_per_epoch-1:
            train_loss[epoch] = running_loss
            print('training loss on epoch {}: {}'.format(epoch, running_loss))
            val_loss = 0
            for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                outputs = model(batch_input)
                loss = criterion(outputs, batch_label)
                val_loss += loss
                """
                if running_loss <= 8:
                    optimizer.param_groups[0]['lr'] = 0.001
                """
                
                break

            print('validation loss on epoch {}: {}'.format(epoch, val_loss))
            vall_loss[epoch] = val_loss
    print_graph_2(loss_value_graph, epoch_value_graph, 'Epoch number {}'.format(epoch))
    scheduler.step(running_loss)
            
def print_graph(train_loss, vall_loss, epoch):
    
    plt.plot(epoch, train_loss, color = "red", label = "Train")
    
    plt.plot(epoch, vall_loss, color = "blue", label = "Vall")
    
    plt.legend()
    
    plt.show()
    
print_graph(train_loss, vall_loss, epoch_np)




def show(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


torch.save(model, 'C:\\Users\\Yusuf\\Desktop\\Models\\09.08.21(3).pth')
print("###############################\n" + "Model Saved!\n"+ "##############################\n")
#best_model = torch.load(r'C:\Users\Yusuf\.spyder-py3\SadiEvrenSeker\Intership 1\model.pth')
#best_model.eval()

### PREDICTION 

def predict(test_input_path_list):
    
    for i in tqdm.tqdm(range(len(test_input_path_list))):
        
        
      batch_test_path_list = test_input_path_list[i: i + 1]
      batch_test = tensorize_image(batch_test_path_list, input_shape, cuda)
      
      result = model(batch_test)
      result = result.argmax(axis = 1)
      result = result.cpu()
      result_np = result.detach().numpy()
      result_np = np.squeeze(result_np, axis = 0)
      
      test_image = cv2.imread(batch_test_path_list[0])
      test_image = cv2.resize(test_image, (224, 224))
      
      copy_image = test_image.copy()
      copy_image[result_np == 1, :] = (255, 0, 125)
      opac_image=(copy_image/2+test_image/2).astype(np.uint8)
      predict_name = batch_test_path_list[0]
      predict_path = predict_name.replace("images", "predicts")
      cv2.imwrite(predict_path, opac_image.astype(np.uint8))
      
      
      
predict(test_input_path_list)
