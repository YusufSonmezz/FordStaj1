import numpy as np
import matplotlib.pyplot as plt



deneme1 = np.zeros((10))

print(deneme1)

deneme2 = np.zeros((10))

deneme3 = np.zeros((10))

epochs = 10

for i in range(epochs):
    deneme1[i] = (i + 2) * 2
    deneme2[i] = (i + 3) * 2
    deneme3[i] = i