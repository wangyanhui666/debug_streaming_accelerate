import os

import torch
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        try:
            feature_file = self.features_files[idx]
            label_file = self.labels_files[idx]

            features = np.load(os.path.join(self.features_dir, feature_file))
            labels = np.load(os.path.join(self.labels_dir, label_file))
            data = {'features': features, 'labels': labels}
        except OSError as e:
            print(f"Error reading file: {e}")
            feature_file = self.features_files[0]
            label_file = self.labels_files[0]

            features = np.load(os.path.join(self.features_dir, feature_file))
            labels = np.load(os.path.join(self.labels_dir, label_file))
            data = {'features': features, 'labels': labels}
        return data