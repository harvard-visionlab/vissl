# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

__all__ = ['AlexNetColorization', 'alexnet_colorization']


def parse_out_keys_arg(
    out_feat_keys: List[str], all_feat_names: List[str]
) -> Tuple[List[str], int]:
    """
    Checks if all out_feature_keys are mapped to a layer in the model.
    Returns the last layer to forward pass through for efficiency.
    Allow duplicate features also to be evaluated.
    Adapted from (https://github.com/gidariss/FeatureLearningRotNet).
    """

    # By default return the features of the last layer / module.
    if out_feat_keys is None or (len(out_feat_keys) == 0):
        out_feat_keys = [all_feat_names[-1]]

    if len(out_feat_keys) == 0:
        raise ValueError("Empty list of output feature keys.")
    for _, key in enumerate(out_feat_keys):
        if key not in all_feat_names:
            raise ValueError(
                f"Feature with name {key} does not exist. "
                f"Existing features: {all_feat_names}."
            )

    # Find the highest output feature in `out_feat_keys
    max_out_feat = max(all_feat_names.index(key) for key in out_feat_keys)

    return out_feat_keys, max_out_feat


def get_trunk_forward_outputs_module_list(
    feat: torch.Tensor,
    out_feat_keys: List[str],
    feature_blocks: nn.ModuleList,
    all_feat_names: List[str] = None,
) -> List[torch.Tensor]:
    """
    Args:
        feat: model input.
        out_feat_keys: a list/tuple with the feature names of the features that
            the function should return. By default the last feature of the network
            is returned.
        feature_blocks: list of feature blocks in the model
        feature_mapping: name of the layers in the model
    Returns:
        out_feats: a list with the asked output features placed in the same order as in
        `out_feat_keys`.
    """
    out_feat_keys, max_out_feat = parse_out_keys_arg(
        out_feat_keys, all_feat_names)
    out_feats = [None] * len(out_feat_keys)
    for f in range(max_out_feat + 1):
        feat = feature_blocks[f](feat)
        key = all_feat_names[f]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat
    return out_feats


class Flatten(nn.Module):
    """
    Flatten module attached in the model. It basically flattens the input tensor.
    """

    def __init__(self, dim=-1):
        super(Flatten, self).__init__()
        self.dim = dim

    def forward(self, feat):
        """
        flatten the input feat
        """
        return torch.flatten(feat, start_dim=self.dim)

    def flops(self, x):
        """
        number of floating point operations performed. 0 for this module.
        """
        return 0


class AlexNetColorization(nn.Module):
    def __init__(self):
        super().__init__()
        conv1_bn_relu = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )

        pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        conv2_bn_relu = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        conv3_bn_relu = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )

        conv4_bn_relu = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )

        conv5_bn_relu = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        flatten = Flatten()

        self._feature_blocks = nn.ModuleList(
            [
                conv1_bn_relu,
                pool1,
                conv2_bn_relu,
                pool2,
                conv3_bn_relu,
                conv4_bn_relu,
                conv5_bn_relu,
                pool3,
                flatten,
            ]
        )
        self.all_feat_names = [
            "conv1",
            "pool1",
            "conv2",
            "pool2",
            "conv3",
            "conv4",
            "conv5",
            "pool5",
            "flatten",
        ]
        assert len(self.all_feat_names) == len(self._feature_blocks)

    def forward(self, x, out_feat_keys=None):
        feat = x
        # In case of LAB image, we take only "L" channel as input. Split the data
        # along the channel dimension into [L, AB] and keep only L channel.
        feat = torch.split(feat, [1, 2], dim=1)[0]
        out_feats = get_trunk_forward_outputs_module_list(
            feat, out_feat_keys, self._feature_blocks, self.all_feat_names
        )
        return out_feats[0] if out_feat_keys is None else out_feats


class ImgPil2LabTensor(object):
    """
    Convert a PIL image to LAB tensor of shape C x H x W
    This transform was proposed in Colorization - https://arxiv.org/abs/1603.08511
    The input image is PIL Image. We first convert it to tensor
    HWC which has channel order RGB. We then convert the RGB to BGR
    and use OpenCV to convert the image to LAB. The LAB image is
    8-bit image in range > L [0, 255], A [0, 255], B [0, 255]. We
    rescale it to: L [0, 100], A [-128, 127], B [-128, 127]
    The output is image torch tensor.
    """

    def __init__(self, indices=[]):
        self.indices = indices

    def __call__(self, image):
        img_tensor = np.array(image)
        # PIL image tensor is RGB. Convert to BGR
        img_bgr = img_tensor[:, :, ::-1]
        img_lab = self._convertbgr2lab(img_bgr.astype(np.uint8))
        # convert HWC -> CHW. The image is LAB.
        img_lab = np.transpose(img_lab, (2, 0, 1))
        # torch tensor output
        img_lab_tensor = torch.from_numpy(img_lab).float()
        return img_lab_tensor

    def _convertbgr2lab(self, img):
        # opencv is not a hard dependency for VISSL so we do the import locally
        import cv2

        # img is [0, 255] , HWC, BGR format, uint8 type
        assert len(img.shape) == 3, "Image should have dim H x W x 3"
        assert img.shape[2] == 3, "Image should have dim H x W x 3"
        assert img.dtype == np.uint8, "Image should be uint8 type"
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # 8-bit image range -> L [0, 255], A [0, 255], B [0, 255]. Rescale it to:
        # L [0, 100], A [-128, 127], B [-128, 127]
        img_lab = img_lab.astype(np.float32)
        img_lab[:, :, 0] = (img_lab[:, :, 0] * (100.0 / 255.0)) - 50.0
        img_lab[:, :, 1:] = img_lab[:, :, 1:] - 128.0
        ############################ debugging ####################################
        # img_lab_bw = img_lab.copy()
        # img_lab_bw[:, :, 1:] = 0.0
        # img_lab_bgr = cv2.cvtColor(img_lab_bw, cv2.COLOR_Lab2BGR)
        # img_lab_bgr = img_lab_bgr.astype(np.float32)
        # img_lab_RGB = img_lab_bgr[:, :, [2, 1, 0]]        # BGR to RGB
        # img_lab_RGB = img_lab_RGB - np.min(img_lab_RGB)
        # img_lab_RGB /= np.max(img_lab_RGB) + np.finfo(np.float64).eps
        # plt.imshow(img_lab_RGB)
        # n = np.random.randint(0, 1000)
        # np.save(f"/tmp/lab{n}.npy", img_lab_bgr)
        # print("SAVED!!")
        ######################### debugging over ##################################
        return img_lab


def alexnet_colorization():
    model = AlexNetColorization()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        ImgPil2LabTensor()
    ])

    return model, preprocess
