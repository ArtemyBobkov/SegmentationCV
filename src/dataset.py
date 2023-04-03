import os

import pandas as pd
import torch

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from os import listdir


class LSCPlantsDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.filepaths = []
        self.img = []
        self.sem = []
        self.inst = []
        self.leaves = []
        for i in range(1, 5):
            for file in listdir(os.path.join(path, f'A{i}')):
                full_name = os.path.join(path, f'A{i}', file)
                if file.endswith('.csv'):
                    df = pd.read_csv(full_name)
                    self.filepaths.extend(map(lambda t: os.path.join(path, f'A{i}', t), df[0].tolist()))
                    self.leaves.extend(df[1].tolist())
                elif full_name.count('rgb') > 0:
                    self.img.append(transforms.ToTensor()(Image.open(full_name)))
                elif full_name.count('fg') > 0:
                    self.sem.append(transforms.ToTensor()(Image.open(full_name)))
                elif full_name.count('label') > 0:
                    self.inst.append(transforms.ToTensor()(Image.open(full_name)))
                else:
                    raise KeyError(f'File {full_name} does not match any conditions!')

    def __getitem__(self, index: int):
        return {'path': self.filepaths[index],
                'img': self.img[index],
                'sem': self.sem[index],
                'inst': self.inst[index],
                'leaves': self.leaves[index]}

    def __len__(self) -> int:
        if len(self.filepaths) == len(self.img) == len(self.sem) == \
                len(self.inst) == len(self.leaves):
            return len(self.filepaths)
        else:
            raise AssertionError('Sizes of all lists does not match')

    def make_split(self, split_path, rel_path):
        split = pd.read_csv(split_path)
        split.split[split.split == 'train'] = 0
        split.split[split.split == 'dev'] = 1
        split.split[split.split == 'test'] = 2

        data_split = [[], [], []]
        for row in split:
            ind = self.filepaths.index(os.path.join(rel_path, row['img_path']))
            data_split[row['split']].append([self.img[ind], self.sem[ind], self.inst[ind], self.leaves[ind]])
        return data_split
