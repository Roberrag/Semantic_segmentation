from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import cv2

from trainer.utils import init_semantic_segmentation_dataset


class SemSegDataset(Dataset):
    """ Generic Dataset class for semantic segmentation datasets.

        Arguments:
            data_path (string): Path to the dataset folder.
            images_folder (string): Name of the folder containing the images (related to the data_path).
            masks_folder (string): Name of the folder containing the masks (related to the data_path).
            csv_path (string): train or test csv file name
            image_ids (list): List of images.
            train_val_test (string): 'train', 'val' or 'test'
            transforms (callable, optional): A function/transform that inputs a sample
                and returns its transformed version.
            class_names (list, optional): Names of the classes.
    """

    def __init__(
            self,
            data_path,
            images_folder,
            masks_folder,
            csv_path,
            #         image_ids,
            train_val_test,
            transforms=None,
            class_names=None,
            isTest=False
    ):
        self.isTest = isTest
        self.transforms = transforms
        csv_path = csv_path + ".csv"
        if train_val_test == 'train' or train_val_test == 'val':
            train_csv_path = os.path.join(data_path, csv_path)
            data = pd.read_csv(train_csv_path)
            len_data = len(data.ImageID)
            num_data_train = int(len_data // 1.25)  # 80% of data
            num_data_test = len_data - num_data_train
            train_data_ids = data.iloc[:num_data_train]
            test_data_ids = data.iloc[num_data_train:]
            if train_val_test == "train":
                self.dataset = init_semantic_segmentation_dataset(train_data_ids, data_path, images_folder,
                                                                  masks_folder, isTest)
            elif train_val_test == "val":
                self.dataset = init_semantic_segmentation_dataset(test_data_ids, data_path, images_folder, masks_folder,
                                                                  isTest)
        elif train_val_test == "test":
            test_csv_path = os.path.join(data_path, csv_path)
            data = pd.read_csv(test_csv_path)
            self.dataset = init_semantic_segmentation_dataset(data, data_path, images_folder, masks_folder, isTest)

    def get_num_clases(self):
        return 12

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.isTest == True:
            sample = {
                "image": cv2.imread(self.dataset[idx]["image"]),
                "size": cv2.imread(self.dataset[idx]["image"]).shape,
                "idx": self.dataset[idx]["idx"]
            }
        else:
            sample = {
                "image": cv2.imread(self.dataset[idx]["image"]),
                "mask": cv2.imread(self.dataset[idx]["mask"], cv2.IMREAD_GRAYSCALE)
            }
        if self.transforms is not None:
            sample = self.transforms(**sample)
            if self.isTest == False:
                sample["mask"] = sample["mask"].long()
        return sample