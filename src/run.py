import lightning.pytorch as pl

from datetime import datetime

from lightning.pytorch.loggers import CSVLogger

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dataset import LSCPlantsDataset
from UNetModel import UNetModel

PATH_TO_DATA = '../CVPPPSegmData/data'
PATH_TO_SPLIT = '../CVPPPSegmData/split.csv'
PATH_TO_MODELS = '../configs'

if __name__ == '__main__':
    dataset = LSCPlantsDataset(PATH_TO_DATA, PATH_TO_SPLIT)
    train = dataset.get_train()
    # plt.imshow(train[10][0].permute((1, 2, 0)))
    # plt.show()
    model = UNetModel(PATH_TO_MODELS, 'resnet18-f37072fd.pth')
    # model.load_state_dict(torch.load('../configs/resnet18-f37072fd.pth'))
    logger = CSVLogger("../logs", name="UNet_logs_" + datetime.now().isoformat())
    train_loader = DataLoader(train, batch_size=4)
    trainer = pl.Trainer(max_epochs=2, logger=logger, default_root_dir=PATH_TO_MODELS)
    trainer.fit(model, train_dataloaders=train_loader)
