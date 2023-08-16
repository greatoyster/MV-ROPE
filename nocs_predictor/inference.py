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
import cv2
import numpy as np

try:
    import ipdb as pdb
except:
    import pdb

from tqdm import tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nocs inference")
    # Add arguments
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, default="test")

    args = parser.parse_args()

    net = NOCSNet(args=args).cuda()
    net.train()

    # TODO: add augment transforms
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    ##### Real #####
    net.load_state_dict(torch.load(args.checkpoint))
    dataset = NOCSDataset("/home/yangjq/Datasets/DROID_DATA/nocs/real", "test")
    testloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    # WARNING: must set shuffle to True

    log_path = os.path.join("logs", args.exp_name)
    output_path = os.path.join(log_path, "nocs_output")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    # for i in (1, 2, 3, 4, 5, 6):
    for i in range(35, 36):
        os.makedirs(os.path.join(output_path, f"scene_{i}"), exist_ok=True)

    print(f"Start testing experiment: {args.exp_name}")

    with torch.no_grad():
        net.eval()
        for i, data in tqdm(
            enumerate(testloader), total=len(testloader), desc="testing..."
        ):
            images, masks, match, scene_id, frame_id = data
            images = images.cuda()
            masks = masks.cuda()
            nocs_images = net(images, masks, match)

            for j in range(len(nocs_images)):
                filename = os.path.join(
                    output_path, scene_id[j], f"{frame_id[j]}_coord.pred.png"
                )
                cv_image = nocs_images[j].permute(1, 2, 0).numpy()
                cv_image *= 255
                cv_image = np.clip(cv_image, 0, 255)
                cv_image = cv_image.astype(np.uint8)
                cv2.imwrite(filename, cv_image)
        print("Finished Testing")
