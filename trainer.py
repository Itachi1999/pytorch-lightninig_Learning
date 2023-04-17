from dataModule import MnistDataModule
from lightningModule import Model
import pytorch_lightning as pl
import config

dataModule = MnistDataModule(config.DATA_DIR, transform=config.TRANSFORM, batch_size=config.BATCH_SIZE)
model = Model(input_dim=config.INPUT_SIZE, num_classes=config.NUM_CLASSES, lr = config.LR)
trainer = pl.Trainer(accelerator = config.ACCELERATOR, devices = config.DEVICES, precision = config.PRECISION, min_epochs = 1, max_epochs = config.NUM_EPOCHS)
trainer.fit(model, datamodule = dataModule)
trainer.validate(model, datamodule = dataModule)
trainer.test(model, datamodule = dataModule)