import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import crop

try:
    import ipdb as pdb
except:
    import pdb

from unet import UNet_encoder
import numpy as np


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, mask):
        loss = ((pred - target) * mask).abs().sum() / (3 * mask.sum())

        return loss


class SymmetricMaskedL1Loss(nn.Module):
    def __init__(self, rot_mats):
        super(SymmetricMaskedL1Loss, self).__init__()
        self.rot_mats = rot_mats
        assert self.rot_mats.requires_grad == False

    def forward(self, pred, target, mask, cat_ids):
        pred -= 0.5
        target -= 0.5
        B, _, H, W = pred.shape
        loss = 0
        for i in range(B):
            if cat_ids[i] in (1, 2, 4):
                rot_pred = torch.matmul(self.rot_mats, pred[i].reshape(3, -1)).reshape(
                    self.rot_mats.shape[0], 3, H, W
                )
                loss += ((rot_pred - target[i]) * mask[i]).abs().sum(
                    dim=(1, 2, 3)
                ).min() / (3.0 * mask[i].sum())
            else:
                loss += ((pred[i] - target[i]) * mask[i]).abs().sum() / (
                    3.0 * mask[i].sum()
                )

        return loss / B


def get_bounding_box(mask: torch.Tensor):
    """mask: (h0, w0) binary tensor"""

    # Find the coordinates of the non-zero elements in the mask
    nonzero_indices = torch.nonzero(mask)

    if len(nonzero_indices) == 0:
        # No non-zero elements found, return None
        return None

    # Calculate the bounding box coordinates
    min_vals, _ = torch.min(nonzero_indices, dim=0)
    max_vals, _ = torch.max(nonzero_indices, dim=0)

    return min_vals[0], min_vals[1], max_vals[0], max_vals[1]


def crop_and_resize_roi(x, mask, return_bbox=False):
    """
    x: (c, h0, w0) tensor
    mask: (h0, w0) binary tensor
    resized patch: (1, c, h1, w1) tensor patch
    """
    bbox = get_bounding_box(mask)

    # Crop the region into a 32x32 patch
    cropped_patch = crop(
        x, bbox[0], bbox[1], bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
    )
    resized_patch = torch.nn.functional.interpolate(
        cropped_patch.unsqueeze(0), size=(32, 32), mode="nearest"
    )

    if return_bbox:
        return resized_patch, bbox
    return resized_patch


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, stride=stride
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(
            planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride
        )
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes // 4)
            self.norm2 = nn.BatchNorm2d(planes // 4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes // 4)
            self.norm2 = nn.InstanceNorm2d(planes // 4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


DIM = 32


class BasicEncoder(nn.Module):
    def __init__(
        self,
        output_dim=128,
        norm_fn="batch",
        dropout=0.0,
        multidim=False,
        upsample=False,
    ):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.multidim = multidim
        self.upsample = upsample

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=DIM)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(DIM)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(DIM)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, DIM, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = DIM
        self.layer1 = self._make_layer(DIM, stride=1)
        self.layer2 = self._make_layer(2 * DIM, stride=2)
        self.layer3 = self._make_layer(4 * DIM, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(4 * DIM, output_dim, kernel_size=1)

        if self.multidim:
            self.layer4 = self._make_layer(256, stride=2)
            self.layer5 = self._make_layer(512, stride=2)

            self.in_planes = 256
            self.layer6 = self._make_layer(256, stride=1)

            self.in_planes = 128
            self.layer7 = self._make_layer(128, stride=1)

            self.up1 = nn.Conv2d(512, 256, 1)
            self.up2 = nn.Conv2d(256, 128, 1)
            self.conv3 = nn.Conv2d(128, output_dim, kernel_size=1)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        if upsample:
            self.up_deconv1 = nn.ConvTranspose2d(
                output_dim, output_dim, kernel_size=2, stride=2
            )
            self.up_deconv2 = nn.ConvTranspose2d(
                output_dim, output_dim, kernel_size=2, stride=2
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        b, c1, h1, w1 = x.shape
        x = x.view(b, c1, h1, w1)

        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu1(x)  # 1/2

        x2 = self.layer1(x1)  # 1/2
        x3 = self.layer2(x2)  # 1/4
        x4 = self.layer3(x3)  # 1/8

        x5 = self.conv2(x4)  # 1/16
        backbone_feats = None
        if self.upsample:
            # TODO: update more reasonable upsample module
            backbone_feats = nn.functional.interpolate(
                x5, scale_factor=8, mode="bilinear", align_corners=False
            )
        else:
            backbone_feats = x5
        _, c2, h2, w2 = backbone_feats.shape
        return backbone_feats.view(b, c2, h2, w2)


class SamePad2d(nn.Module):
    """
    Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = (
            (out_width - 1) * self.stride[0] + self.kernel_size[0] - in_width
        )
        pad_along_height = (
            (out_height - 1) * self.stride[1] + self.kernel_size[1] - in_height
        )
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)

    def __repr__(self):
        return self.__class__.__name__


class NOCSPredictor(nn.Module):
    def __init__(self, **kwargs):
        super(NOCSPredictor, self).__init__()

        self.num_classes = 6

        self.padding = SamePad2d(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(self.padding(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(self.padding(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(self.padding(x))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(self.padding(x))
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)

        x = torch.sigmoid(x)

        return x


class NOCSNet(nn.Module):
    def __init__(self, args):
        super(NOCSNet, self).__init__()

        # self.image_encoder = BasicEncoder(
        #     output_dim=256, norm_fn='batch', upsample=True)

        self.image_encoder = UNet_encoder(n_channels=3, n_classes=256)

        self.nocs_predictor_x = NOCSPredictor()
        self.nocs_predictor_y = NOCSPredictor()
        self.nocs_predictor_z = NOCSPredictor()

        def rotation_y_matrix(theta):
            rotation_matrix = np.array(
                [
                    np.cos(theta),
                    0,
                    np.sin(theta),
                    0,
                    1,
                    0,
                    -np.sin(theta),
                    0,
                    np.cos(theta),
                ]
            )
            rotation_matrix = np.reshape(rotation_matrix, (3, 3))
            return rotation_matrix

        M = 20
        rot_mats = np.array(
            [rotation_y_matrix(float(i) * np.float32(2 * np.pi / M)) for i in range(M)]
        )
        self.loss_func = SymmetricMaskedL1Loss(
            torch.from_numpy(rot_mats).float().cuda().requires_grad_(False)
        )

    def forward(self, images, masks, match, nocs_gt=None):
        if self.training:
            assert nocs_gt is not None
            nocs_gt_patches = []
        else:
            bboxes = []
            patch_images_ids = []
        mask_patches = []
        feat_patches = []
        cat_ids = []
        inst_ids = []
        # use image feature extractor to output feature map
        feats = self.image_encoder(images)  # b,c,h,w full resolution
        # calculate bounding box region and crop it
        for i in range(masks.shape[0]):
            mask_ids = masks[i].unique(sorted=True)[:-1]

            for obj_id in mask_ids:
                cat_id = int(match[i][obj_id.item() - 1].item())
                inst_ids.append(obj_id.item())
                if cat_id == 0:
                    continue
                inst_mask = masks[i] == obj_id
                cropped_feat = crop_and_resize_roi(feats[i], inst_mask)
                if self.training:
                    cropped_mask = crop_and_resize_roi(
                        inst_mask[None, :, :].float(), inst_mask
                    )
                else:
                    patch_images_ids.append(i)
                    cropped_mask, bbox = crop_and_resize_roi(
                        inst_mask[None, :, :].float(), inst_mask, True
                    )
                    bboxes.append(bbox)
                feat_patches.append(cropped_feat)

                if self.training:
                    cropped_nocs_gt = crop_and_resize_roi(nocs_gt[i], inst_mask)
                    nocs_gt_patches.append(cropped_nocs_gt)
                mask_patches.append(cropped_mask)
                cat_ids.append(cat_id)

        if len(cat_ids) == 0:
            if self.training:
                zero_loss = torch.zeros(1, device=feats.device, requires_grad=True)
                return zero_loss, None
            else:
                return None
        feat_patches = torch.cat(feat_patches, dim=0)

        # apply nocs regional cnn
        nocs_pred_x = self.nocs_predictor_x(feat_patches)  # (b, num_class, 32, 32)
        nocs_pred_y = self.nocs_predictor_y(feat_patches)  # (b, num_class, 32, 32)
        nocs_pred_z = self.nocs_predictor_z(feat_patches)  # (b, num_class, 32, 32)

        nocs_pred = torch.cat(
            [
                nocs_pred_x.unsqueeze(2),
                nocs_pred_y.unsqueeze(2),
                nocs_pred_z.unsqueeze(2),
            ],
            dim=2,
        )
        # b, num_class, 3 32, 32
        nocs_pred = torch.stack(
            [nocs_pred[i, cat_ids[i] - 1, ...] for i in range(len(cat_ids))]
        )
        mask_patches = torch.cat(mask_patches, dim=0)
        if self.training:
            nocs_gt_patches = torch.cat(nocs_gt_patches, dim=0)
            loss = self.loss_func(nocs_pred, nocs_gt_patches, mask_patches, cat_ids)
            return loss, nocs_pred
        else:
            nocs_images = torch.zeros_like(images, device="cpu")
            assert (
                len(nocs_pred) == len(patch_images_ids) == len(mask_patches)
            ), "data length mismatch"
            for i in range(len(patch_images_ids)):
                ith_nocs_pred = torch.nn.functional.interpolate(
                    nocs_pred[i].unsqueeze(0),
                    size=(
                        bboxes[i][2] - bboxes[i][0] + 1,
                        bboxes[i][3] - bboxes[i][1] + 1,
                    ),
                    mode="bilinear",
                )
                ith_nocs_mask = masks[
                    patch_images_ids[i],
                    bboxes[i][0] : bboxes[i][2] + 1,
                    bboxes[i][1] : bboxes[i][3] + 1,
                ].unsqueeze(0)
                ith_nocs_mask = ith_nocs_mask == inst_ids[i]
                masked_nocs_pred = ith_nocs_pred * ith_nocs_mask
                nocs_images[
                    patch_images_ids[i],
                    :,
                    bboxes[i][0] : bboxes[i][2] + 1,
                    bboxes[i][1] : bboxes[i][3] + 1,
                ] += masked_nocs_pred[0].cpu()
            return nocs_images


class CoordBinValues(nn.Module):
    """
    Module to convert NOCS bins to values in range [0,1]
    """

    def __init__(self, coord_num_bins):
        super(CoordBinValues, self).__init__()
        self.coord_num_bins = coord_num_bins

    def forward(self, mrcnn_coord_bin):
        mrcnn_coord_bin_value = mrcnn_coord_bin.argmax(dim=2) / self.coord_num_bins

        return mrcnn_coord_bin_value
