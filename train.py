import torch
import config

from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from discriminator_model import Discriminator
from generator_model import Generator
from dataset import MapDataset

from utils import save_checkpoint, save_image, save_some_examples, load_checkpoint


def train_fun(gen, disc, gen_optim, disc_optim, BCELoss, l1_loss, loader):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

    y_fake = gen(x)
    D_real = disc(x, y)
    D_real_loss = BCELoss(D_real, torch.ones_like(D_real))
    D_fake = disc(x, y_fake.detach())
    D_fake_loss = BCELoss(D_fake, torch.zeros_like(D_fake))
    D_loss = (D_real_loss + D_fake_loss) / 2

    disc_optim.zero_grad()
    D_loss.backward()
    disc_optim.step()

    D_fake = disc(x, y_fake)
    G_fake_loss = BCELoss(D_fake, torch.ones_like(D_fake))
    L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
    G_loss = G_fake_loss + L1

    gen_optim.zero_grad()
    G_loss.backward()
    gen_optim.step()

    if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

def main():
    gen = Generator(config.CHANNELS_IMG).to(config.DEVICE)
    disc = Discriminator(config.CHANNELS_IMG).to(config.DEVICE)
    LR = config.LEARNING_RATE
    gen_optimizer = Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
    disc_optimizer = Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, gen_optimizer, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, disc_optimizer, config.LEARNING_RATE,
        )

    train_dataset = MapDataset(
            root_dir=config.TRAIN_DIR, input_transform=config.input_transform, target_transform=config.target_transform
            )
    train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True
    )

    val_dataset = MapDataset(
            root_dir=config.VAL_DIR, input_transform=config.input_transform, target_transform=config.target_transform
            )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train_fun(gen, disc, gen_optimizer, disc_optimizer, L1_LOSS, BCE, train_loader)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, gen_optimizer, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, disc_optimizer, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")



if __name__ == "__main__":
    main()