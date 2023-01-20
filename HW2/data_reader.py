import csv
import random
import cv2
import torch as th
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

from imageDataset import ImageDataset, ImageDatasetTest

th.manual_seed(13)

def read_image(img_path):
    img_path = "./" + img_path
    img = cv2.imread(img_path)
    img = img / 255
    img = np.transpose(img, [2, 0, 1])

    return img

transforms_func = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_transforms_func = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def data_reader_with_normalization(path, batch_size=256, split_value = 20000, enhance=False, enhance_dataset_path = './predicted_unlabeled.csv'):
    file = open(path, encoding="utf8")
    csv_data = csv.reader(file)
    next(csv_data)

    data_y_info = []
    for path, cls in csv_data:
        data_y_info.append((int(cls), './' + path))

    random.shuffle(data_y_info)
    data_Y, data_paths = zip(*data_y_info)

    train_dataset = ImageDataset(data_paths[:split_value], data_Y[:split_value],  train_transforms_func)
    val_dataset = ImageDataset(data_paths[split_value:], data_Y[split_value:], transforms_func)

    if enhance:
        data_Y = []
        data_paths = []
        file = open(enhance_dataset_path, encoding="utf8")
        csv_data = csv.reader(file)
        next(csv_data)
        for path, cls in csv_data:
            data_Y.append(int(cls))
            data_paths.append(path)
        enhance_train_dataset = ImageDataset(data_paths, data_Y, train_transforms_func)
        train_dataset = th.utils.data.ConcatDataset([train_dataset, enhance_train_dataset])

    train_loader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader= th.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

def read_val_data_with_normalization(path, length = 5000, batch_size=256):
    dataset = ImageDatasetTest(path, length, transforms_func)
    test_loader= th.utils.data.DataLoader(dataset, batch_size=batch_size)

    return test_loader

def data_reader(path, batch_size=256):
    data_X = []
    data_Y = []
    file = open(path, encoding="utf8")

    csv_data = csv.reader(file)
    next(csv_data)

    for path, cls in csv_data:
        img = read_image(path)
        data_X.append(img)
        data_Y.append(int(cls))

    file.close()


    data_X = th.tensor(np.array(data_X), dtype=th.float32)
    data_Y = th.tensor(np.array(data_Y), dtype=th.long)

    dataset = th.utils.data.TensorDataset(data_X, data_Y)

    train_set, val_set = th.utils.data.random_split(dataset, [20000, 3555])
    train_loader = th.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader= th.utils.data.DataLoader(val_set, batch_size=batch_size)

    return train_loader, val_loader

def read_val_data():
    data_X = []

    for i in range(5000):
        path = './task1/val_data/' +  str(i) + '.jpeg'
        img = read_image(path)
        data_X.append(img)

    return th.tensor(np.array(data_X), dtype=th.float32)

def write_submision(predictions, path, length = 5000):
    header = ['sample', 'label']
    data = []
    for i in range(length):
        data.append([str(i) + '.jpeg', str(predictions[i])])

    with open(path, 'w', encoding='UTF8', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

def write_labeled(predictions, path):
    header = ['sample', 'label']
    data = []
    for i in range(len(predictions)):
        data.append(['./task1/train_data/images/unlabeled/' +str(predictions[i][0]) + '.jpeg', str(predictions[i][1])])

    with open(path, 'w', encoding='UTF8', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

if __name__ == "__main__":
    train_loader, val_loader = data_reader_with_normalization("./task1/train_data/annotations.csv")
    for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            print(inputs, labels)
            break
    # test_loader = read_val_data_with_normalization()
    # for i, data in enumerate(test_loader, 0):
    #         inputs = data
    #         print(inputs)
    #         break
    # test = read_val_data_with_normalization()
    # print(len(test))
    # print(test[0].shape)
    
