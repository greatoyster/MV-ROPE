import argparse
import torch
import torchvision.transforms as transforms
from torch.optim import Adam
from nocs import NOCSNet
from data import NOCSDataset
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import os

try:
    import ipdb as pdb
except:
    import pdb

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nocs trainer")
    # Add arguments
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument(
        "--checkpoint", type=str, default="none", help="load checkpoint model"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--num_epochs", type=int, default=48)
    parser.add_argument("--epoch_length", type=int, default=5000)

    args = parser.parse_args()

    net = NOCSNet(args=args).cuda()
    net.train()

    if args.checkpoint != "none":
        print("model load checkpoint")

    # TODO: add augment transforms
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    ##### Real #####
    net.load_state_dict(
        torch.load(
            "/home/yangjq/Projects/DROID_NOCS/NOCS/logs/init_train/model_40_loss_0.0274_.pth"
        )
    )
    dataset = NOCSDataset("/home/yangjq/Datasets/DROID_DATA/nocs/real", "train")
    trainset_len = int(len(dataset) * 0.9)
    valset_len = len(dataset) - trainset_len
    trainset, valset = random_split(dataset, [trainset_len, valset_len])
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    ##### Real #####

    train_epoch_length = min(5000, len(trainloader))
    val_epoch_length = min(4000, len(valloader))

    # WARNING: must set shuffle to True

    optimizer = Adam(net.parameters(), 1e-3)
    scheduler = StepLR(optimizer, step_size=int(args.num_epochs / 8), gamma=0.32)

    log_path = os.path.join("logs", args.exp_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_dir=log_path, comment=args.exp_name)

    print(f"Start training experiment: {args.exp_name}")
    for epoch in range(args.num_epochs):  # loop over the dataset multiple times
        print(f"epoch: {epoch} ...")
        tqdm_bar = tqdm(total=train_epoch_length, desc="Training epoch: %d" % epoch)
        net.train()
        for i, data in enumerate(trainloader):
            if i >= train_epoch_length:
                break
            tqdm_bar.update()
            # get the inputs; data is a list of [inputs, labels]
            images, masks, nocs_gts, match = data
            images = images.cuda()
            nocs_gts = nocs_gts.cuda()
            masks = masks.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss, _ = net(images, masks, match, nocs_gts)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                writer.add_scalar("loss", loss, epoch * train_epoch_length + i)
        scheduler.step()
        if epoch % int(args.num_epochs / 20) == 0 or (epoch + 1) == args.num_epochs:
            with torch.no_grad():
                print(f"validate at epoch {epoch} ...")
                # net.eval()
                # we need the loss value to validate
                total_loss = 0.0
                for i, data in enumerate(valloader):
                    if i >= val_epoch_length:
                        break
                    images, masks, nocs_gts, match = data
                    images = images.cuda()
                    nocs_gts = nocs_gts.cuda()
                    masks = masks.cuda()
                    loss, _ = net(images, masks, match, nocs_gts)
                    total_loss += loss.item()
                total_loss /= val_epoch_length
            torch.save(
                net.state_dict(),
                os.path.join(log_path, f"model_{epoch}_loss_{total_loss:.4f}_.pth"),
            )

    print("Finished Training")
