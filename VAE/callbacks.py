from pytorch_lightning.callbacks import Callback, EarlyStopping

class myPrintingCallBack(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        # return super().on_train_start(trainer, pl_module)
        print("Start Training")

    def on_train_end(self, trainer, module):
        print("Training End")