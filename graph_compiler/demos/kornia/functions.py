import cv2
import numpy as np
import kornia
import torch


# load image
img_np = cv2.cvtColor(cv2.imread('demos/kornia/odsc.png'), cv2.COLOR_BGR2RGB)/255

# transform to kornia format
img_np = np.expand_dims(np.transpose(img_np, (2, 0, 1)), 0)

# composite function
def dilate_edges(img):
    edges = kornia.filters.canny(img, hysteresis=False)[1]
    return kornia.morphology.dilation(edges, torch.ones(7, 7))

# original torch
img = torch.tensor(img_np)