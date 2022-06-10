import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

class MusicDataset(Dataset):
    def __init__(self, transform):
        self.transform = transform
        self.imgs_path = "data\\"
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        self.class_map = {}
        index = 0
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            self.class_map[index] = class_name
            for img_path in glob.glob(class_path + "\\*.png"):
                self.data.append([img_path, index])
            index+=1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_index = self.data[idx]
        img = Image.open(img_path)
        class_id = class_index
        img_tensor = self.transform(img)
        class_id = torch.tensor(class_id)
        return img_tensor, class_id