import torch
import torchvision.transforms as transforms
import torchvision
 
from torch.utils.data import DataLoader,Dataset
 
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
  def __init__(self, data_paths, labels, transform):
    self.transform=transform
    self.data_paths=data_paths
    self.labels=np.array(labels)
   
  def __len__(self):
    return len(self.labels)
 
  def __getitem__(self,index):
    image=cv2.imread(self.data_paths[index])
    image=self.transform(image)
    targets=self.labels[index]
     
    return (image, targets)

class ImageDatasetTest(Dataset):
  def __init__(self, img_path, length, transform):
    self.transform=transform
    self.img_path=img_path
    self.length = length
   
  def __len__(self):
    return self.length
 
  def __getitem__(self,index):
    image=cv2.imread(self.img_path+ str(index)+ '.jpeg')
    image=self.transform(image)

    return image