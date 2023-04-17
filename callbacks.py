from pytorch_lightning.callbacks import Callback, EarlyStopping

class myPrintingCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
    
    def on_train_start(self, trainer, pl_module) -> None:
        print(f"Training Start")

    def on_train_end(self, trainer, pl_module) -> None:
        print(f"Training is done.")

    