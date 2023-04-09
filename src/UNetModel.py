import os
from typing import Optional, Any

import torch
import lightning.pytorch as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from torchvision.models import resnet18
from torchvision.transforms import CenterCrop, Resize

from metrics import DiffFgDICE, DiffFgMSE
from matplotlib import pyplot as plt


class UNetModel(pl.LightningModule):
    transformer = Resize(512)

    def __init__(self, path_to_models, backbone='resnet18'):
        super().__init__()
        self.base_model_ = resnet18(pretrained=True)
        self.base_model_.load_state_dict(torch.load(os.path.join(path_to_models, backbone)))
        self.backbone_layers_ = list(self.base_model_.children())

        self.construct_encoder()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.construct_decoder()
        self.step = 0
        self.dice = 0
        self.mse = 0
        self.test_count = 0
        self.save_hyperparameters(logger=False)

    def construct_encoder(self):
        self.conv_enc0 = torch.nn.Conv2d(3, 64, 1)
        self.conv_enc1 = self.backbone_layers_[4]
        self.connect1 = self.connect_layers(64, 64)
        self.conv_enc2 = self.backbone_layers_[5]
        self.connect2 = self.connect_layers(128, 128)
        self.conv_enc3 = self.backbone_layers_[6]
        self.connect3 = self.connect_layers(256, 256)
        self.conv_enc4 = self.backbone_layers_[7]
        self.connect4 = self.connect_layers(512, 512)
        self.conv_enc5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            torch.nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU()
        )

    def construct_decoder(self):
        self.conv_dec1 = self.conv_decoder_layer(1024 + 512, 512, (3, 3))
        self.conv_dec2 = self.conv_decoder_layer(512 + 256, 256, (3, 3))
        self.conv_dec3 = self.conv_decoder_layer(256 + 128, 128, (3, 3))
        self.conv_dec4 = self.conv_decoder_layer(128 + 64, 64, (3, 3))
        self.conv_last = torch.nn.Conv2d(64, 1, 1, (1, 1))

    def forward(self, x):
        x1 = self.conv_enc1(self.conv_enc0(x))
        x2 = self.conv_enc2(self.connect1(x1))
        x3 = self.conv_enc3(self.connect2(x2))
        x4 = self.conv_enc4(self.connect3(x3))
        x5 = self.conv_enc5(self.connect4(x4))

        x = self.decode(x4, x5, self.conv_dec1, 64)
        x = self.decode(x3, x, self.conv_dec2, 128)
        x = self.decode(x2, x, self.conv_dec3, 256)
        x = self.decode(x1, x, self.conv_dec4, 512)
        x = self.conv_last(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch
        x = UNetModel.transformer(x)
        y = UNetModel.transformer(y)
        y_hat = self(x)
        loss = 1 - DiffFgDICE(y_hat, y)
        self.step += 1
        if self.step % 10 == 0:
            self.visualize_train_results(y_hat, y)
        print('loss', loss)
        self.log('train loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y, _, _ = batch
        x = UNetModel.transformer(x)
        y = UNetModel.transformer(y)
        y_hat = self(x)
        dices = []
        for pred, true in zip(y_hat, y):
            dices.append(DiffFgDICE(pred, true).item())
            self.dice += DiffFgDICE(pred, true).item()
            self.mse += DiffFgMSE(pred, true).item()
            self.test_count += 1
        self.visualize_test_result(y_hat, x, dices, batch_idx)
        print(self.dice / self.test_count)
        print(self.mse / self.test_count)
        return y_hat

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y, _, _ = batch
        x = UNetModel.transformer(x)
        y = UNetModel.transformer(y)
        y_hat = self(x)
        dices = []
        for pred, true in zip(y_hat, y):
            dices.append(DiffFgDICE(pred, true).item())
            self.dice += DiffFgDICE(pred, true).item()
            self.test_count += 1
        self.visualize_test_result(y_hat, y, dices)
        print(self.dice / self.test_count)
        return y_hat

    def configure_optimizers(self, optimizer=torch.optim.Adam, lr=3e-4):
        return optimizer(self.parameters(), lr=lr)

    def connect_layers(self, in_channels, out_channels, kernel=(1, 1), padding=0, stride=1):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding, stride=stride),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        )

    def conv_decoder_layer(self, in_channels, out_channels, kernel=(1, 1), padding=(1, 1), stride=1):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding, stride=stride),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=padding, stride=stride),
            torch.nn.ReLU(),
        )

    def decode(self, x_skipped, x_encoded, layer, out_size):
        x = self.upsample(x_encoded)
        new_tensor = torch.cat([CenterCrop(out_size)(x_skipped), x], dim=1)
        return layer(new_tensor)

    def visualize_train_results(self, y_hat, y):
        fig, axis = plt.subplots(1, 2, figsize=(12, 6))
        axis[0].imshow(y_hat.detach().cpu()[0].permute((1, 2, 0)))
        axis[1].imshow(y.detach().cpu()[0].permute((1, 2, 0)))
        plt.show()

    def visualize_test_result(self, y_hat, x, metrics, batch_idx=0):
        y_hat = y_hat.detach().cpu()
        x = x.detach().cpu()

        for j in range(max(1, y_hat.shape[0] // 2)):
            fig, axis = plt.subplots(1, 4, figsize=(24, 6))
            y_show = y_hat[j:j + 2]
            x_show = x[j:j + 2]
            metrics_show = metrics[j:j + 2]
            for i in range(4):
                if i % 2 == 1:
                    axis[i].imshow(y_show[i // 2].permute(1, 2, 0))
                    axis[i].set_title(f'Predicted segments, score = {metrics_show[i // 2]}')
                else:
                    axis[i].imshow(x_show[i // 2].permute(1, 2, 0))
                    axis[i].set_title('Real image')
            plt.savefig(f'../test_results/{batch_idx}_{j}.png', dpi=800)
            plt.show()
