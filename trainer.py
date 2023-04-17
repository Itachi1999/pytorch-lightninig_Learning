from dataModule import MnistDataModule
from lightningModule import Model
import pytorch_lightning as pl
import config
from callbacks import myPrintingCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger('tb_logger/', 'KMNIST_exp', version = '0_0')

dataModule = MnistDataModule(config.DATA_DIR, transform=config.TRANSFORM, batch_size=config.BATCH_SIZE)

model = Model(input_dim=config.INPUT_SIZE, num_classes=config.NUM_CLASSES, lr = config.LR)

trainer = pl.Trainer(accelerator = config.ACCELERATOR, devices = config.DEVICES, precision = config.PRECISION, min_epochs = 1, max_epochs = config.NUM_EPOCHS, callbacks=[myPrintingCallback(), EarlyStopping(monitor='val_loss')], logger=logger)

trainer.fit(model, datamodule = dataModule)
trainer.validate(model, datamodule = dataModule)
trainer.test(model, datamodule = dataModule)