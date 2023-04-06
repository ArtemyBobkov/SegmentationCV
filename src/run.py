import sys

import lightning.pytorch as pl

from datetime import datetime

from lightning.pytorch.loggers import CSVLogger

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from dataset import LSCPlantsDataset
from UNetModel import UNetModel

PATH_TO_DATA = '../CVPPPSegmData/data'
PATH_TO_SPLIT = '../CVPPPSegmData/split.csv'
PATH_TO_MODELS = '../configs'

if __name__ == '__main__':
    dataset = LSCPlantsDataset(PATH_TO_DATA, PATH_TO_SPLIT)
    if sys.argv[1] == 'train':
        train = dataset.get_train()
        model = UNetModel(PATH_TO_MODELS, 'resnet18-f37072fd.pth')
        logger = CSVLogger("../logs", name="UNet_logs_" + datetime.now().isoformat())
        train_loader = DataLoader(train, batch_size=4, shuffle=True)
        trainer = pl.Trainer(max_epochs=2, logger=logger, default_root_dir=PATH_TO_MODELS, val_check_interval=0.25)
        dev_loader = DataLoader(dataset.get_dev(), batch_size=4, shuffle=True)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=dev_loader)
    if sys.argv[1] == 'test':
        test = dataset.get_test()
        model = UNetModel.load_from_checkpoint(
            path_to_models=PATH_TO_MODELS,
            backbone='resnet18-f37072fd.pth',
            checkpoint_path=sys.argv[2]
        )
        test_loader = DataLoader(test, batch_size=4)
        pl.Trainer().predict(model, dataloaders=test_loader)
    else:
        raise ValueError


