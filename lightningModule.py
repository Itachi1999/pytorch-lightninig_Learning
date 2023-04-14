# The model of the MNIST Classifier is done using lightning modules with extra facilities

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import pytorch_lightning as pl

#Hyperparameters
batch_size = 32
lr = 3e-04
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Selected Device: {device}")
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)
epochs = 3

# Dataset and Dataloader
train_ds = ImageFolder('Datasets/MNIST/train/', transform = transform)
val_ds = ImageFolder('Datasets/MNIST/test/', transform = transform)

train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
val_dl = DataLoader(val_ds, batch_size = batch_size, shuffle = True)


# Model Definition

# class Model(nn.Module):
#     def __init__(self, input_dim, num_classes) -> None:
#         super().__init__()
#         self.linear1 = nn.Linear(input_dim, 50)
#         self.linear2 = nn.Linear(50, num_classes)
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x):
#         x = F.relu(self.linear1(x))
#         x = F.softmax(self.linear2(x), dim = 1)

#         return x

class Model(pl.LightningModule):
    def __init__(self, input_dim, num_classes) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 50)
        self.linear2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.softmax(self.linear2(x), dim = 1)

        return x
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.shape[0], -1)
        scores = self.forward(x)
        loss = self.criterion(scores, y)
        predictions = torch.max(scores, dim = 1)
        return loss, predictions, y


    def training_step(self, batch, batch_idx):
        # We can do much more
        loss, pred, y = self._common_step(batch, batch_idx)
        self.log('training_loss', loss)
        return loss        

    def validation_step(self, batch, batch_idx):
        # We can do much more
        loss, pred, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        # We can do much more
        loss, pred, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx: int):
        _, pred, _ = self._common_step(batch, batch_idx)
        return pred
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = 1e-04)

num_classes = len(train_ds.classes)
#print(num_classes)

#print(next(iter(train_dl))[0].shape)

model = Model(28*28*3, num_classes).to(device)
optimiser = optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()


# Training Loop

for epoch in range(epochs):
    running_loss = 0
    counter = 0

    for batch_idx, (img, label) in enumerate(tqdm(train_dl)):
        img = img.to(device)
        img = img.reshape(img.shape[0], -1)
        label = label.to(device)

        scores = model(img)
        #_, predictions = torch.max(scores, dim = 1)
        loss = criterion(scores, label)

        running_loss += loss
        counter += 1

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"For Epoch: {epoch + 1}, loss: {running_loss / counter}")


def calculate_accuracy(model, val_dl):
    
    print(f"Validation Loop")
    correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for (img, label) in tqdm(val_dl):
            img = img.to(device)
            img = img.reshape(img.shape[0], -1)
            label = label.to(device)

            scores = model(img)
            _, predictions = torch.max(scores, dim = 1)

            correct += (predictions == label).sum()
            num_samples += predictions.shape[0]

    
    model.train()
    return correct / num_samples



print(f"Accuracy on training samples: {calculate_accuracy(model, train_dl) * 100: .2f}")
print(f"Accuracy on validation samples: {calculate_accuracy(model, val_dl) * 100: .2f}")


