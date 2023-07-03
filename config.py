import torch
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/maps/train"
VAL_DIR = "data/maps/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 600
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"


input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(0.2),
    transforms.Resize(IMAGE_SIZE, antialias=True),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

target_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE, antialias=True),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

