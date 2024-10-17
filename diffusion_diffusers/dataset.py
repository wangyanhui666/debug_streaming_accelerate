import os

import torch
import numpy as np
from torch.utils.data import Dataset

class CustomDataset_fastdit(Dataset):
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
    

class CustomDataset(Dataset):
    def __init__(self, features_dir):
        # features_dir, _features/_labels
        L = os.listdir(features_dir)
        print(f'---> Folders in {features_dir}: {L}')
        for name in L:
            if name.endswith('_features'):
                self.features_dir = os.path.join(features_dir, name)
            elif name.endswith('_labels'):
                self.labels_dir = os.path.join(features_dir, name)


        self.features_files = sorted(os.listdir(self.features_dir), key=lambda x:int(x.split('_')[0])*8+int(x[-5]))[:-1]
        self.labels_files = sorted(os.listdir(self.labels_dir), key=lambda x:int(x.split('_')[0])*8+int(x[-5]))[:-1]
        print(f'---> Features files: {len(self.features_files)}')
        assert len(self.features_files) == len(self.features_files) == 1281167 # ImageNet

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return  {'features': torch.from_numpy(features).squeeze(0), 'labels': torch.from_numpy(labels).squeeze(0)}