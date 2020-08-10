import os

import torch

from trainer.configuration import OptimizerConfig, TrainerConfig, DataloaderConfig
from trainer.datasets import SemSegDataset
from trainer.deeplab import Deeplabv3
from trainer.trainermodel import Training_model
from trainer.utils import patch_configs, draw_semantic_segmentation_batch
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, Resize, CenterCrop, Normalize, RandomCrop, HorizontalFlip, ShiftScaleRotate, HueSaturationValue
from albumentations.pytorch import ToTensorV2, ToTensor


if __name__ == '__main__':
    # create test dataset
    data_path = "Data/"
    images_folder = "imgs/imgs"
    masks_folder = "masks/masks"
    csv_path = "train"
    train_val_test = "train"
    test_dataset = SemSegDataset(data_path, images_folder, masks_folder, csv_path, train_val_test)

    # run the experiment
    dataloader_config, trainer_config = patch_configs(epoch_num_to_set=TrainerConfig.epoch_num,
                                                      batch_size_to_set=DataloaderConfig.batch_size)
    optimizer_config = OptimizerConfig(learning_rate=OptimizerConfig.learning_rate,
                                       lr_step_milestones=OptimizerConfig.lr_step_milestones,
                                       weight_decay=OptimizerConfig.weight_decay)
    train = Training_model(dataloader_config=dataloader_config, optimizer_config=optimizer_config)
    experiment = train.run(trainer_config)

    # Run inference
    # load model
    model_dir = 'model/'
    model_file_name = "Deeplabv3_best.pth"
    model_path = os.path.join(model_dir, model_file_name)
    # loading the model and getting model parameters by using load_state_dict
    num_classes = 12
    model = Deeplabv3(num_classes_output=num_classes, pretrained=True, trained=True)
    # print(model)
    device = torch.device('cuda')
    model.load_state_dict(torch.load(model_path, map_location=device))

    loader_test = DataLoader(
        SemSegDataset(

            data_path,
            images_folder,
            masks_folder,
            csv_path,
            "val",
            transforms=Compose([
                Resize(224, 224),
                CenterCrop(224, 224),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()])
        ),
        batch_size=3,
        shuffle=False,
        num_workers=3,
        pin_memory=True
    )
    output = {"mask": []}
    images = next(iter(loader_test))
    for i in range(3):
        with torch.no_grad():
            preds = model(images["image"])[i]
            om = torch.argmax(preds.squeeze(), dim=0)
            output["mask"].append(om)
    draw_semantic_segmentation_batch(images["image"], images["mask"], output["mask"])


