import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
import torch
# import torchmetrics
from VAE.losses import VAE_loss
import math


class ConvTransposeBlock(pl.LightningModule):
    def __init__(self, in_channel, out_channel, kernel, stride, padding, out_padding = 1) -> None:
        super().__init__()

        self.deconv = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel, stride=stride, padding=padding, output_padding=out_padding)
        self.activation = nn.PReLU()
        self.batch_norm = nn.BatchNorm2d(num_features=out_channel)


    def forward(self, x):
        x = self.deconv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        return x

class ConvolutionBlock(pl.LightningModule):
    def __init__(self, input_channels, out_channels, kernel, stride, padding) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.activation = nn.PReLU()
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x) 

        return x

class LinearBlock(pl.LightningModule):
    def __init__(self, in_dim, out_dim, p = 0.5) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.PReLU()
        self.drp_out = nn.Dropout1d(p)

    def forward(self, x):
        x = self.linear(x)
        x = self.drp_out(x)
        x = self.activation(x)

        return x

class Encoder(pl.LightningModule):
    def __init__(self, channel = 3, latent_dim = 64):
        super().__init__()

        self.conv1 = ConvolutionBlock(input_channels=channel, out_channels=8, kernel= 3, stride=2, padding=1)
        self.conv2 = ConvolutionBlock(input_channels=8, out_channels=16, kernel= 3, stride=2, padding=1)
        self.conv3 = ConvolutionBlock(input_channels=16, out_channels=32, kernel= 3, stride=2, padding=1)
        self.conv4 = ConvolutionBlock(input_channels=32, out_channels=64, kernel= 3, stride=2, padding=1)
        self.conv5 = ConvolutionBlock(input_channels=64, out_channels=128, kernel= 3, stride=2, padding=1)
        self.conv6 = ConvolutionBlock(input_channels=128, out_channels= 256, kernel= 3, stride=2, padding=1)

        self.flatten = nn.Flatten()
        
        self.fc1 = LinearBlock(1024, 512)
        self.fc2 = LinearBlock(512, 256)
        self.fc3 = LinearBlock(256, 128)
        self.fc4 = nn.Linear(128, latent_dim)

        self.mu = nn.Linear(128, latent_dim)
        self.log_var = nn.Linear(128, latent_dim)
        

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = self.conv5(x)
        # print(x.shape)
        x = self.conv6(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        mu =  self.mu(x)
        log_var =  self.log_var(x)
        x = torch.sigmoid(self.fc4(x))
        # print(x.shape)

        return x, mu, log_var
    
class Decoder(pl.LightningModule):
    def __init__(self, latent_dim = 64, output_channel = 3) -> None:
        super().__init__()

        self.fc1 = LinearBlock(latent_dim, 128)
        self.fc2 = LinearBlock(128, 256)
        self.fc3 = LinearBlock(256, 512)
        self.fc4 = LinearBlock(512, 1024)

        self.unflatten = nn.Unflatten(1, (256, 2, 2))

        self.deconv1 = ConvTransposeBlock(256, 128, 3, 2, 1)
        self.deconv2 = ConvTransposeBlock(128, 64, 3, 2, 1)
        self.deconv3 = ConvTransposeBlock(64, 32, 3, 2, 1)
        self.deconv4 = ConvTransposeBlock(32, 16, 3, 2, 1)
        self.deconv5 = ConvTransposeBlock(16, 8, 3, 2, 1)
        self.deconv6 = nn.ConvTranspose2d(8, output_channel, 3, 2, 1, 1)


    def forward(self, x):
        x = self.fc1(x)
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        x = self.fc4(x)
        # print(x.shape)
        x = self.unflatten(x)
        # print(x.shape)
        x = self.deconv1(x)
        # print(x.shape)
        x = self.deconv2(x)
        # print(x.shape)
        x = self.deconv3(x)
        # print(x.shape)
        x = self.deconv4(x)
        # print(x.shape)
        x = self.deconv5(x)
        # print(x.shape)
        x = self.deconv6(x)
        # print(x.shape)
        
        return x

class VAE(pl.LightningModule):
    def __init__(self, lr = 3e-04, latent_dim = 64, img_samples = 25) -> None:
        super().__init__()
        self.encoder = Encoder(3, latent_dim)
        self.decoder = Decoder(latent_dim, 3)
        self.criterion = VAE_loss()
        self.lr = lr
        self.latent_dim = latent_dim
        self.img_samples = img_samples
        self.training_step_losses = []
        self.val_step_losses = []

    def forward(self, x):
        x, mu, log_var = self.encoder(x)
        recon = self.decoder(x)

        return recon, mu, log_var
    
    def _common_step(self, batch, batch_idx):
        imgs, _ = batch
        recon, mu, log_var = self.forward(imgs)
        loss = self.criterion(recon, imgs, mu, log_var)

        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.training_step_losses.append(loss)

        if batch_idx % 500 == 0:
            imgs = x[:25]
            grid = torchvision.utils.make_grid(imgs)
            self.logger.experiment.add_image('CIFAR_training_Images', grid, self.global_step)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.val_step_losses.append(loss)
       
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_losses).mean()
        self.log('train_epoch_loss', avg_loss)
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_step_losses).mean()
        self.log('train_epoch_loss', avg_loss)

        gaussian_noise = torch.randn(self.img_samples, self.latent_dim).to(self.device)
        new_imgs = self.decoder(gaussian_noise)
        grid = torchvision.utils.make_grid(tensor=new_imgs, nrow=int(math.sqrt(self.img_samples)))

        self.logger.experiment.add_image('CIFAR_new_images', grid, self.global_step)

    def configure_optimizers(self):
        # return super().configure_optimizers(
        return optim.Adam(self.parameters(), lr=self.lr)
        

if __name__ == '__main__':
    x = torch.randn(64, 3, 128, 128)
    model = Encoder()
    mod = Decoder()
    x = model(x)
    x = mod(x)
    print(x.shape)
    