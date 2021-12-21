from PIL import Image
import json
import torch
from torchvision import transforms
import cv2
import numpy as np
import os
import torch.nn as nn

def show_cam_on_img(img, mask, img_path_save):
    heat_map = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heat_map = np.float32(heat_map) / 255

    cam = heat_map + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(img_path_save, np.uint8(255 * cam))

    
img_path_read = ""
img_path_save = ""




def main():
  img = cv2.imread(img_path_read, flags=1)

  img = np.float32(cv2.resize(img, (224, 224))) / 255

  # cam_all is the score tensor of shape (B, C, H, W), similar to y_raw in out Figure 1
  # cls_idx specifying the i-th class out of C class
  # visualize the 0's class heatmap
  cls_idx = 0
  cam = cam_all[cls_idx]


  # cam = nn.ReLU()(cam)
  cam = cam / torch.max(cam)
  
  cam = cv2.resize(np.array(cam), (224, 224))
  show_cam_on_img(img, cam, img_path_save)
  
