# The model of the MNIST Classifier is done using lightning modules with extra facilities
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import torchvision
from metrics import myAccuracy

# Dataset and Dataloader
class Model(pl.LightningModule):
    def __init__(self, input_dim, num_classes, lr) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 50)
        self.linear2 = nn.Linear(50, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy('multiclass', num_classes = num_classes)
        self.f1_score = torchmetrics.F1Score('multiclass', num_classes = num_classes)
        self.myAcc = myAccuracy()
        self.lr = lr

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.softmax(self.linear2(x), dim = 1)

        return x
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.shape[0], -1)
        scores = self.forward(x)
        loss = self.criterion(scores, y)
        #predictions = torch.max(scores, dim = 1)

        #In this state, we'll see the training pictures
        if batch_idx % 100 == 0:
            imgs = x[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
            self.logger.experiment.add_image('MNIST_Images', grid, self.global_step)

        return loss, scores, y


    def training_step(self, batch, batch_idx):
        # We can do much more
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        my_acc = self.myAcc(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({'training_loss': loss, 'training_acc': accuracy, 'myAccuracy': my_acc, 'training_f1': f1_score}, prog_bar = True, on_epoch = True)
        #self.log('training_loss', loss)
        return loss        

    def validation_step(self, batch, batch_idx):
        # We can do much more
        loss, scores, y = self._common_step(batch, batch_idx)
        acc = self.accuracy(scores, y)
        self.log_dict({'val_loss': loss, 'accuracy': acc})
        return loss
    
    def test_step(self, batch, batch_idx):
        # We can do much more
        loss, scores, y = self._common_step(batch, batch_idx)
        acc = self.accuracy(scores, y)
        self.log_dict({'test_loss': loss, 'accuracy': acc})
        return loss
    
    def predict_step(self, batch, batch_idx: int):
        _, pred, _ = self._common_step(batch, batch_idx)
        return pred
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = self.lr)