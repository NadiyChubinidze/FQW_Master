import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, classification_report
import torch
import os
import dataset as dt
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import numpy as np


class Analyst():
    def __init__(self, root, model, device, val_loader, batch_size, classes):
        self.root = root
        self.model = model
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.device = device
        self.y = []
        self.y_pred = []
        self.target_names = classes.values()
        self.target_classes = list(classes.keys())
        self.fpr = dict()
        self.tpr = dict()

    def get_pred_and_graundtruth(self):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for data, targets in iter(self.val_loader):
                data = data.to(self.device)
                results = self.model(data)
                for i in results.cpu():
                    self.y_pred.append(i.numpy())
                self.y += targets.tolist()

        self.y_pred = np.array(self.y_pred)

    def create_roc(self,file_name):
        y_every = label_binarize(self.y, classes=self.target_classes)
        n_classes = y_every.shape[1]

        roc_auc = dict()
        plt.clf()
        for i in range(n_classes):
            self.fpr[i], self.tpr[i], _ = roc_curve(y_every[:, i], self.y_pred[:, i])
            roc_auc[i] = auc(self.fpr[i], self.tpr[i])
            plt.plot(self.fpr[i], self.tpr[i])

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.savefig(os.path.join(self.root, file_name + '.png'))
    
    def get_f_measure(self,file_name):
        y_pred = [np.argmax(x) for x in self.y_pred]
        report = classification_report(self.y,y_pred,target_names = self.target_names)

        file_analitic = open(os.path.join(self.root,file_name + '.txt'), 'w+')
        file_analitic.write('-' * 100 + '\n')
        file_analitic.write(self.root + "\n")
        file_analitic.write('-' * 100 + '\n')
        file_analitic.write(report)
        file_analitic.close()

    def create_macro_roc(self,file_name):
        n_classes = len(self.target_classes)
        all_fpr = np.unique(np.concatenate([self.fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr,self.fpr[i],self.tpr[i])
        mean_tpr/=n_classes

        plt.clf()
        plt.plot(all_fpr, mean_tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.savefig(os.path.join(self.root, file_name + '_macro.png'))
