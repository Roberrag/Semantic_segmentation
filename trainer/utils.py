import os
import sys
import random
# import shutil
# import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .configuration import SystemConfig, TrainerConfig, DataloaderConfig



def init_semantic_segmentation_dataset(dataframe, data_path, imgs_folder="imgs/imgs", masks_folder="masks/masks", isTest = False):
    dataset = []
    if isTest == False:

        for ids in dataframe.ImageID:

            dataset.append({
                "image" : os.path.join(data_path, imgs_folder, str(ids)+ ".jpg"),
                "mask" : os.path.join(data_path, masks_folder, str(ids)+ ".png"),
            })
    else:
        for ids in dataframe.ImageID:
            dataset.append({
                "image": os.path.join(data_path, imgs_folder, str(ids) + ".jpg"),
                "idx": str(ids),
                "size": (cv2.imread(os.path.join(data_path, imgs_folder, str(ids) + ".jpg"))).shape
            })
    
    return dataset


def draw_semantic_segmentation_samples(dataset, n_samples=3):
    """ Draw samples from semantic segmentation dataset.

    Arguments:
        dataset (iterator): dataset class.
        plt (matplotlib.pyplot): canvas to show samples.
        n_samples (int): number of samples to visualize.
    """
    fig, ax = plt.subplots(nrows=n_samples, ncols=2, sharey=True, figsize=(10, 10))
    for i, sample in enumerate(dataset):
        if i >= n_samples:
            break
        ax[i][0].imshow(sample["image"])
        ax[i][0].set_xlabel("image")
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])

        ax[i][1].imshow(sample["mask"])
        ax[i][1].set_xlabel("mask")
        ax[i][1].set_xticks([])
        ax[i][1].set_yticks([])

    plt.tight_layout()
    plt.gcf().canvas.draw()
    plt.show()
    plt.close(fig)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count

def patch_configs(epoch_num_to_set=TrainerConfig.epoch_num, batch_size_to_set=DataloaderConfig.batch_size):
    """ Patches configs if cuda is not available

    Returns:
        returns patched dataloader_config and trainer_config

    """
    # default experiment params
    num_workers_to_set = DataloaderConfig.num_workers

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        batch_size_to_set = 16
        num_workers_to_set = 2
        epoch_num_to_set = 1

    dataloader_config = DataloaderConfig(batch_size=batch_size_to_set, num_workers=num_workers_to_set)
    trainer_config = TrainerConfig(device=device, epoch_num=epoch_num_to_set, progress_bar=True)
    return dataloader_config, trainer_config

def setup_system(system_config: SystemConfig) -> None:
    torch.manual_seed(system_config.seed)
    np.random.seed(system_config.seed)
    random.seed(system_config.seed)
    torch.set_printoptions(precision=10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_config.seed)
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic

def draw_semantic_segmentation_batch(images, masks_gt, masks_pred=None, n_samples=3):
    """ Draw batch from semantic segmentation dataset.

    Arguments:
        images (torch.Tensor): batch of images.
        masks_gt (torch.LongTensor): batch of ground-truth masks.
        plt (matplotlib.pyplot): canvas to show samples.
        masks_pred (torch.LongTensor, optional): batch of predicted masks.
        n_samples (int): number of samples to visualize.
    """
    nrows = min(images.size(0), n_samples)
    ncols = 2 if masks_pred is None else 3
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=(10, 10))
    for i in range(nrows):
        img = images[i].permute(1, 2, 0).detach().cpu().numpy()
        img = np.clip(img, 0, 1)
        ax[i][0].imshow(img)
        ax[i][0].set_xlabel("image")
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])
        gt_mask = masks_gt[i].detach().cpu().numpy()
        ax[i][1].imshow(gt_mask)
        ax[i][1].set_xlabel("ground-truth mask")
        ax[i][1].set_xticks([])
        ax[i][1].set_yticks([])
        if masks_pred is not None:
            pred = masks_pred[i].detach().cpu().numpy()
            ax[i][2].imshow(pred)
            ax[i][2].set_xlabel("predicted mask")
            ax[i][2].set_xticks([])
            ax[i][2].set_yticks([])

    plt.tight_layout()
    plt.gcf().canvas.draw()
    plt.show()
    plt.close(fig)

