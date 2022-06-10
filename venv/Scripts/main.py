import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision import transforms, models
import torchvision
import torch.optim as optim
import dataset as dt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import utils
import torch.nn as nn
import torch.nn.functional as F
import time
import cv2
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_curve
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import VGGNet
import MobileNet
import ResNet
import SqueezeNet


def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
    resize = transforms.Resize((224, 224))
    my_transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])

    dataset = dt.MusicDataset(my_transform)
    BATCH_SIZE = 16
    TRAIN_SIZE = 27000
    TEST_SIZE = 6273
    NUM_EPOCH = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ''' ----------------------------------------------------------------------------------------- '''

    net = VGGNet.VGGNet('models', device, dataset, BATCH_SIZE, TRAIN_SIZE, TEST_SIZE)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.get_parameters_models())
    net.train(NUM_EPOCH, loss_fn, optimizer,True)
    net.analize()

    ''' ----------------------------------------------------------------------------------------- '''

    net = ResNet.ResNet('models', device, dataset, BATCH_SIZE, TRAIN_SIZE, TEST_SIZE)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.get_parameters_models())
    net.train(NUM_EPOCH, loss_fn, optimizer,True)
    net.analize()

    ''' ----------------------------------------------------------------------------------------- '''

    net = SqueezeNet.SqueezeNet('models', device, dataset, BATCH_SIZE, TRAIN_SIZE, TEST_SIZE)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.get_parameters_models())
    net.train(NUM_EPOCH, loss_fn, optimizer,True)
    net.analize()

    ''' ----------------------------------------------------------------------------------------- '''

    net = MobileNet.MobileNet('models', device, dataset, BATCH_SIZE, TRAIN_SIZE, TEST_SIZE)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.get_parameters_models())
    net.train(NUM_EPOCH, loss_fn, optimizer,True)
    net.analize()

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()



