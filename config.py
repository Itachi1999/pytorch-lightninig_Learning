import torch
import torchvision.transforms as transforms

# Training Hyperparameters
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
BATCH_SIZE = 32
LR = 3e-04
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 3
print(f"Selected Device: {DEVICE}")


# Dataset
TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

DATA_DIR = 'Datasets/KMNIST/'

# Computation
ACCELERATOR = 'gpu'
PRECISION = '16-mixed'
DEVICES = [0]