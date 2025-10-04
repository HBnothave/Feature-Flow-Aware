import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class HistopathologyDataset(Dataset):
    def __init__(self, data_dir, dataset_name, transform=None):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.transform = transform
        self.images = []
        self.labels = []
        self.num_classes = self._load_dataset()

    def _load_dataset(self):
        if self.dataset_name == 'BreakHis':
            classes = ['A', 'F', 'PT', 'TA', 'DC', 'LC', 'MC', 'PC']
            for cls_idx, cls in enumerate(classes):
                cls_dir = os.path.join(self.data_dir, cls)
                for img_name in os.listdir(cls_dir):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(cls_idx)
            return len(classes)
        
        elif self.dataset_name == 'NCT-CRC-HE-100K':
            classes = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
            for cls_idx, cls in enumerate(classes):
                cls_dir = os.path.join(self.data_dir, cls)
                for img_name in os.listdir(cls_dir):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(cls_idx)
            return len(classes)
        
        elif self.dataset_name == 'GasHisSDB-160':
            classes = ['Normal', 'Abnormal']
            for cls_idx, cls in enumerate(classes):
                cls_dir = os.path.join(self.data_dir, cls)
                for img_name in os.listdir(cls_dir):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(cls_idx)
            return len(classes)
        
        elif self.dataset_name == 'ROSE':
            classes = ['Negative', 'Positive']
            for cls_idx, cls in enumerate(classes):
                cls_dir = os.path.join(self.data_dir, cls)
                for img_name in os.listdir(cls_dir):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(cls_idx)
            return len(classes)
        
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label