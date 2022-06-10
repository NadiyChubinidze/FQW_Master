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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Analyst import Analyst 

class MobileNet():
    def __init__(self, root, device, dataset, batch_size, train_size, test_size):
        self.root = root
        self.device = device
        self.model = models.mobilenet_v3_small(pretrained=False, num_classes=24)
        self.batch_size = batch_size
        self.target_names = dataset.class_map.values()
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        self.classes = dataset.class_map
        self.model = self.model.to(device)
        self.file = os.path.join(root, 'MobileNet', 'INFO.txt')
        self.FIX_STEP = False
        file = open(self.file, 'w+')
        file.close()

    def get_parameters_models(self):
        return self.model.parameters()

    def name(self):
        return 'MobileNet'

    def train_one_epoch(self):
        for data, targets in tqdm(iter(self.train_loader)):
            data = data.to(self.device)
            targets = targets.to(self.device)
            results = self.model(data)
            loss = self.loss_fn(results, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, num_epochs, loss_fn, optimizer,fix_step=False):
        self.FIX_STEP = fix_step
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.model.train()
        start_time = time.time()
        for epoch in range(num_epochs):
            self.train_one_epoch()
            model_scripted = torch.jit.script(self.model)
            model_scripted.save(
                os.path.join(self.root, 'MobileNet', 'MobileNet_' + str(epoch) + '_' + str(self.batch_size) + '.pt'))
            self.eval_network(epoch)
            if self.FIX_STEP:
                self.analize(str(epoch))

        elapsed_time = time.time() - start_time
        file = open(self.file, 'a')
        file.write((str('-' * 90) + '\n') * 3)
        file.write("Spent " + str(elapsed_time) + " seconds training for " + str(num_epochs) + " epoch(s).\n")
        print("Spent " + str(elapsed_time) + " seconds training for " + str(num_epochs) + " epoch(s).")
        file.close()

    def eval_network(self,epoch):
        start_time = time.time()
        num_correct = 0
        total_guesses = 0

        self.model.eval()
        with torch.no_grad():
            for data, targets in iter(self.val_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)
                results = self.model(data)
                best_guesses = torch.argmax(results, 1)
                num_correct += torch.eq(targets, best_guesses).sum().item()
                total_guesses += self.batch_size

            elapsed_time = time.time() - start_time
            file = open(self.file, 'a')
            file.write(str('-' * 90) + '\n')
            file.write("Correctly guessed " + str(num_correct / total_guesses * 100) + "% of the dataset. Epoch: "+ str(epoch+1) + '\n')
            print("Correctly guessed " + str(num_correct / total_guesses * 100) + "% of the dataset")
            file.write("Evaluated in " + str(elapsed_time) + " seconds")
            file.write("\n")
            print("Evaluated in " + str(elapsed_time) + " seconds")
            file.close()
            
    def analize(self, file_name = 'Total'):
        analizer = Analyst(os.path.join('analize', self.name()),self.model, self.device, self.val_loader, self.batch_size, self.classes)
        analizer.get_pred_and_graundtruth()
        analizer.create_roc(file_name)
        analizer.get_f_measure(file_name)
        analizer.create_macro_roc(file_name)