import os
from typing import Tuple, List, Any, Dict

import pandas as pd
import torch

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class LSCPlantsDataset(Dataset):
    transformer = transforms.ToTensor()

    def __init__(self, path: str, split_path: str) -> None:
        self.datasets = []
        for i in range(1, 5):
            self.datasets.append(pd.read_csv(os.path.join(path, f'A{i}', f'A{i}.csv'),
                                             names=['img', 'leaves']))
        self.train, self.dev, self.test = self.make_split_(path, split_path)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError('Should not be called, use get_train, get_dev and get_test functions instead')

    def __len__(self) -> int:
        total_length = 0
        for dataset in self.datasets:
            total_length += len(dataset)
        return total_length

    def get_train(self) -> Dataset:
        return LSCPlantsDatasetSplit(self, 'train')

    def get_dev(self) -> Dataset:
        return LSCPlantsDatasetSplit(self, 'dev')

    def get_test(self) -> Dataset:
        return LSCPlantsDatasetSplit(self, 'test')

    def make_split_(self, rel_path: str, split_path: str) -> Tuple[List[Any], List[Any], List[Any]]:
        split = pd.read_csv(split_path)
        split.split[split.split == 'train'] = 0
        split.split[split.split == 'dev'] = 1
        split.split[split.split == 'test'] = 2
        split = split.replace('data/', '', regex=True)

        data_split = ([], [], [])
        for row in split.iterrows():
            img_path = os.path.join(rel_path, row[1]['img_path'])
            folder_index = int(os.path.dirname(img_path)[-1]) - 1
            _, img_name = os.path.split(img_path)
            df = self.datasets[folder_index]
            data_split[row[1]['split']].append({
                'img': img_path,
                'sem': os.path.join(rel_path, row[1]['sem_path']),
                'inst': os.path.join(rel_path, row[1]['inst_path']),
                'leaves': df.loc[df.img == img_name].leaves.item()}
            )
        return data_split


class LSCPlantsDatasetSplit(Dataset):
    def __init__(self, dataset: LSCPlantsDataset, mode):
        mapping = {'train': dataset.train, 'dev': dataset.dev, 'test': dataset.test}
        if mode in mapping:
            self.data = mapping[mode]
        else:
            raise ValueError('Wrong argument!')

    def __getitem__(self, index):
        img = LSCPlantsDataset.transformer((Image.open(self.data[index]['img']).crop((0, 30, 500, 530))))
        sem = LSCPlantsDataset.transformer(Image.open(self.data[index]['sem']).crop((0, 30, 500, 530)))
        inst = LSCPlantsDataset.transformer(Image.open(self.data[index]['inst']).crop((0, 30, 500, 530)))
        return img, sem, inst, self.data[index]['leaves']

    def __len__(self):
        return len(self.data)
