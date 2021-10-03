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
As you see above, Freespaces have exterior points. These points are used to detect which area on the image belongs to Freespace. <br>
Overview to Code:
<br>
<img src = "https://github.com/YusufSonmezz/FordStaj1/blob/main/Intern%20Images/jsontomask.png">
<br>
**In the code**, 
- Exterior points are read from JSON file
- With `fillPolly` function from OpenCV, inside of these points paint with a color that is given
- Lastly, The new mask image is written the given direction with `imwrite` function

