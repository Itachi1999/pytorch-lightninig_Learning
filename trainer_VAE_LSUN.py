from VAE.dataModule import CIFAR10_DataModule
from VAE.model import VAE
from VAE.callbacks import EarlyStopping, myPrintingCallBack
import VAE.config as cf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

callbacks = [myPrintingCallBack(), EarlyStopping(monitor='val_loss')]
logger = TensorBoardLogger('tb_logger', 'CIFAR_VAE', '0_0')
dataModule = CIFAR10_DataModule(cf.DATA_DIR, cf.BATCH_SIZE, cf.TRANSFORM)
model = VAE(cf.LR, cf.LATENT_DIM)

trainer = pl.Trainer(accelerator=cf.ACCELERATOR, devices=cf.DEVICES, precision=cf.PRECISION, logger=logger, callbacks=callbacks, min_epochs=1, max_epochs=cf.NUM_EPOCHS)

trainer.fit(model=model, datamodule=dataModule)
trainer.validate(model=model, datamodule=dataModule)

