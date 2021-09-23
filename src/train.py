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

def print_graph_2(loss_train, epoch, label):
    plt.plot(epoch, loss_train)
    plt.title(label)
    plt.show()

######### PARAMETERS ##########
valid_size = 0.15
test_size  = 0.15
batch_size = 8
epochs = 25
cuda = True
input_shape = (224, 224)
n_classes = 2
###############################

# For model saving
models = {
    "train_loss": list(),
    "vall_loss": list()
    }

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

# DATA CHECK
image_mask_check(image_path_list, mask_path_list)

aug_image_path_list = glob.glob(os.path.join(AUG_IMAGE, '*'))
aug_mask_path_list  = glob.glob(os.path.join(AUG_MASK,  '*'))


vip_aug_img = glob.glob(os.path.join(r'C:\Users\Yusuf\.spyder-py3\SadiEvrenSeker\Intership 1\data\vip_aug_img', '*'))
vip_aug_mask = glob.glob(os.path.join(r'C:\Users\Yusuf\.spyder-py3\SadiEvrenSeker\Intership 1\data\vip_aug_mask', '*'))

indices = np.random.permutation(len(image_path_list))

total_images = image_path_list + aug_image_path_list + vip_aug_img
#  
total_labels = mask_path_list + aug_mask_path_list + vip_aug_mask
# 

# SHUFFLE INDICES
indices = np.random.permutation(len(total_images))

image_path_list = list(np.array(total_images)[indices])
mask_path_list = list(np.array(total_labels)[indices])

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

# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list)//batch_size

# CALL MODEL
model = UNet(input_size=input_shape, n_classes=2)
#model = torch.load(r"C:\Users\Yusuf\Desktop\Best Models among models\18.09.21(1)_18.pth")
#model.eval()

# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay = 1e-6)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 2, verbose = True)

# IF CUDA IS USED, IMPORT THE MODEL AND LOSS FUNCTION INTO CUDA
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

# TRAINING THE NEURAL NETWORK
vall_loss = np.zeros((epochs))
train_loss = np.zeros((epochs))

train_accuracy = np.zeros((epochs))
vall_accuracy = np.zeros((epochs))
for epoch in range(epochs):
    print()
    indices = np.random.permutation(len(train_input_path_list))
    
    train_input_path_list = list(np.array(train_input_path_list)[indices])
    train_label_path_list = list(np.array(train_label_path_list)[indices])
    
    loss_value_graph = np.zeros((steps_per_epoch))
    epoch_value_graph = np.zeros((steps_per_epoch))
    running_loss = 0
    accuracy = 0
    total_train = 0
    for ind in tqdm.tqdm(range(steps_per_epoch)):
        batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)
        batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)
        
        optimizer.zero_grad()
        model.zero_grad()
        
        outputs = model(batch_input)
        
        loss = criterion(outputs, batch_label)
        loss.backward()
        
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        _, batch_label_2 = torch.max(batch_label.data, 1)
        total_train += batch_label.nelement()
        accuracy += (predicted.data == batch_label_2.data).sum().float()
        
        #loss_value_graph[ind] = loss.item();
        #epoch_value_graph[ind] = ind;
        running_loss += loss.float()
        if ind == steps_per_epoch-1:
            train_loss[epoch] = running_loss / steps_per_epoch
            train_accuracy[epoch] = 100 * accuracy / total_train * 2
            val_loss = 0
            accuracy = 0
            total_train = 0
            #model.eval()
            for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                outputs = model(batch_input)
                loss = criterion(outputs, batch_label)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                _, batch_label_2 = torch.max(batch_label, 1)
                total_train += batch_label.numel()
                accuracy += predicted.eq(batch_label_2.data).sum().item()
            
            
            val_loss = val_loss / len(valid_input_path_list)
            print(f'\nrunning loss on epoch {epoch}: {running_loss}')
            print('training loss on epoch {}: {}'.format(epoch, train_loss[epoch]))
            print('validation loss on epoch {}: {}'.format(epoch, val_loss))
            print(f'\nThe Accuracy of TRAIN data is % {train_accuracy[epoch]}')
            print(f'The Accuracy of VALIDATION data is % {100 * accuracy / total_train * 2}\n')
            model_name = '23.09.21(1)'
            vall_loss[epoch] = val_loss
            scheduler.step(vall_loss[epoch])
            vall_accuracy[epoch] = 100 * accuracy / total_train * 2
            models['train_loss'].append(running_loss / steps_per_epoch)
            models['vall_loss'].append(val_loss)
    
    #print_graph_2(loss_value_graph, epoch_value_graph, 'Epoch number {}'.format(epoch))
    torch.save(model, fr'C:\Users\Yusuf\Desktop\EarlyStoppingModels\{model_name}\{model_name}_{epoch}.pth')
            
def LossGraph(train_loss,vall_loss):
    plt.plot(range(epochs), train_loss, color = 'red', label = 'Train')
    plt.plot(range(epochs), vall_loss, color = 'blue', label = 'Vall')
    plt.title('Loss Graph')
    plt.legend()
    plt.show()
    
def AccuracyGraph(train_accuracy, vall_accuracy):
    plt.plot(range(epochs), train_accuracy, color = 'red', label = 'Train')
    plt.plot(range(epochs), vall_accuracy, color = 'blue', label = 'Vall')
    plt.title('Accuracy Graph')
    plt.legend()
    plt.show()


LossGraph(train_loss, vall_loss)
AccuracyGraph(train_accuracy, vall_accuracy)

train_losses = models['train_loss']
val_losses = models['vall_loss']

for i in range(len(train_losses)):
    print(f'{i}  numarali epoch -> TRAIN LOSS..: {train_losses[i]}, VAL_LOSS..: {val_losses[i]}')


def show(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

### PREDICTION 

def predict(test_input_path_list, test_label_path_list):
    accuracy = 0
    total_train = 0
    model.eval()
    for i in tqdm.tqdm(range(len(test_input_path_list))):
        
      batch_test_input_path_list = test_input_path_list[i: i + 1]
      batch_test_label_path_list = test_label_path_list[i: i + 1]
      batch_test = tensorize_image(batch_test_input_path_list, input_shape, cuda)
      batch_label = tensorize_mask(batch_test_label_path_list, input_shape, n_classes, cuda)
      
      result = model(batch_test)
      
      _, predicted = torch.max(result.data, 1)
      _, batch_label_2 = torch.max(batch_label, 1)
      total_train += batch_label.numel()
      accuracy += predicted.eq(batch_label_2.data).sum().item()
      
      result = result.argmax(axis = 1)
      result = result.cpu()
      result_np = result.detach().numpy()
      result_np = np.squeeze(result_np, axis = 0)
      
      test_image = cv2.imread(batch_test_input_path_list[0])
      test_image = cv2.resize(test_image, (224, 224))

      
      
      copy_image = test_image.copy()
      copy_image[result_np == 1, :] = (255, 0, 125)
      opac_image=(copy_image/2+test_image/2).astype(np.uint8)
      predict_name = batch_test_input_path_list[0]
      predict_path = predict_name.replace("images", "predicts")
      cv2.imwrite(predict_path, opac_image.astype(np.uint8))
    print(f'The Accuracy of TEST data is % {100 * accuracy / total_train * 2}\n')
      
      
      
predict(test_input_path_list, test_label_path_list)