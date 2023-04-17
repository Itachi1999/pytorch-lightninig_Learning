import torch
import torchvision.transforms as transforms

# Training Hyperparameters
INPUT_SIZE = 128 * 128
NUM_CLASSES = 10
LATENT_DIM = 64
BATCH_SIZE = 32
LR = 3e-04
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 20
print(f"Selected Device: {DEVICE}")


# Dataset
TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(128),
        transforms.Normalize(0.5, 0.5),
    ]
)

DATA_DIR = 'Datasets/'

# Computation
ACCELERATOR = 'gpu'
PRECISION = '16-mixed'
DEVICES = [0]