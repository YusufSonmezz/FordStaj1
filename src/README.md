## Summary Of Project
This project aims detection of freespace in highway. In this project, datas which are limited has been processed and augmented, the UNet model has been used and with [train.py](https://github.com/YusufSonmezz/FordStaj1/blob/main/src/train.py), the model has been fitted.

## Content Of Project
* Convertion of Json files to Mask Images     : [json2mask.py](https://github.com/YusufSonmezz/FordStaj1/blob/main/src/json2mask.py)
* Checking Masks With Placing Them on Images : [mask_on_image.py](https://github.com/YusufSonmezz/FordStaj1/blob/main/src/mask_on_image.py)
* Preparing Datas to GPU supported Tensors : [preprocess.py](https://github.com/YusufSonmezz/FordStaj1/blob/main/src/preprocess.py)
* Creating The Model : [model_unet.py](https://github.com/YusufSonmezz/FordStaj1/blob/main/src/model_unet.py)
* Fitting The Model : [train.py](https://github.com/YusufSonmezz/FordStaj1/blob/main/src/train.py)
* Checking If The Model is Overfitted With K-Fold Cross Validation : [trainCV.py](https://github.com/YusufSonmezz/FordStaj1/blob/main/src/trainCV.py)
* Making Predictions With Unseen Images To Understand If Model is OK : [predict.py](https://github.com/YusufSonmezz/FordStaj1/blob/main/src/predict.py)

## Why PyTorch has been used in the project?
PyTorch is a Python library that provides three utulity:

1. Converting NumPy arrays to [Tensors](https://pytorch.org/docs/stable/tensors.html) to use strength of GPU
2. Sets proper and flexible environment to user and speed the computings up through GPU
3. Has a clean and understandable API which gets works easier

In addition, people who use PyTorch increasingly growes.
<br><br>
<img src = "https://github.com/YusufSonmezz/FordStaj1/blob/main/Intern%20Images/pytorchvstensorflow.png" width = "600" height = "300">

## Json To Mask
Json To Mask targets to make mask images. Mask images will be used in later stage while the model fitting.

Overview to JSON File: 
<br>
<img src = "https://github.com/YusufSonmezz/FordStaj1/blob/main/Intern%20Images/Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202021-10-03%20165648.png" width = "600" height = "400">
<br>
As you see above, Freespaces have exterior points. These points are used to detect which area on the image belongs to Freespace. <br> <br>
Overview to Code:
<br>
<img src = "https://github.com/YusufSonmezz/FordStaj1/blob/main/Intern%20Images/jsontomask.png">
<br>
**In the code**, 
- Exterior points are read from JSON file
- With `fillPolly` function from OpenCV, inside of these points paint with a color that is given
- Lastly, The new mask image is written to given direction with `imwrite` function

## Mask On Image
Mask On Image module gets mask images and placing them on the original images. The result is an image with mask and original image. Thus, we can check mask images if they are truly created or not.

Here, we get mask and image. `np.uint8` represents data type that is supported by OpenCV.
<br> <br>
<img src = "https://github.com/YusufSonmezz/FordStaj1/blob/main/Intern%20Images/mask_on_image.png"> <br>

Secondly, we put mask on image and opacify them to not lose image where they cross.
<br> <br>
<img src = "https://github.com/YusufSonmezz/FordStaj1/blob/main/Intern%20Images/opac_image.png"> <br>

Overview to result of [mask_on_image.py](https://github.com/YusufSonmezz/FordStaj1/blob/main/src/mask_on_image.py):
<br>
<img src = "https://github.com/YusufSonmezz/FordStaj1/blob/main/Intern%20Images/cfc_000249.png" widht = "700" height = "350">

## Preprocess 
Preprocess is one of the most important part of project. We convert datas to tensors here. Converting them has two layer. Firstly, converting numpy structure to tensor structure. Secondly, giving tensorize arrays to PyTorch library for making them tensors.
<br> <br>
In this section, two components have considered. 
1. Masks
2. Images

### Mask
Masks are 1D arrays. They contain 1 and 0 values. Therefore, we need to make them 2D arrays to prepare masks for training. The reason of converting 1D to 2D is that we have two component to evaluate. That means we have two area, the pixels that belong to freespace and not to belong freespace. One Hot Encoding technique has been used to converting process.

Overview to array to tensor for mask images:
<br>
```
# Create empty list
local_mask_list = []

# For each masks
for mask_path in mask_path_list:

    # Access and read mask
    mask = cv2.imread(mask_path, 0)

    # Resize the image according to defined shape
    mask = cv2.resize(mask, output_shape)

    # Apply One-Hot Encoding to image
    mask = one_hot_encoder(mask, n_class)

    # Change input structure according to pytorch input structure
    torchlike_mask = torchlike_data(mask)
    
    local_mask_list.append(torchlike_mask)

 mask_array = np.array(local_mask_list, dtype=np.int)
 torch_mask = torch.from_numpy(mask_array).float()
 if cuda:
    torch_mask = torch_mask.cuda()

return torch_mask
```

Overview to One Hot Encoder Tecnique:

```
# Define array whose dimensison is (width, height, number_of_class)
encoded_data = np.zeros((data.shape[0], data.shape[1], n_class), dtype=np.int)

for lbl in range(n_class):
      encoded_data[:, :, lbl][data == lbl] = 1

return encoded_data
```
Represantation for OneHotEncoding:
<br>
<img src = "https://github.com/YusufSonmezz/FordStaj1/blob/main/Intern%20Images/4HVuN.png" width = "900" height = "450"> <br> <br>

### Image
Images are already 3D arrays. They have Red, Green and Blue layers. When we read images, they are read as numpy arrays by OpenCV. NumPy arrays are not suitable for GPU architecture. What we want here is to change architecture of NumPy structure to Tensor structure that is for GPU. 

NumPy  structure -> *HxWxC* <br>
Tensor structure -> *CxHxW* <br>

Overview to converting Numpy to TensorLike data:
```
# Obtain channel value of the input
n_channels = data.shape[2]

# Create and empty image whose dimension is similar to input
torchlike_data_output = np.empty((n_channels, data.shape[0], data.shape[1]))

for c in range(n_channels):
      torchlike_data_output[c] = data[:, :, c]

return torchlike_data_output
```

## Model





































