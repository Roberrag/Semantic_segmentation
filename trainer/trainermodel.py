import torch

from trainer import Trainer
from trainer.configuration import SystemConfig, DatasetConfig, DataloaderConfig, OptimizerConfig, TrainerConfig
from trainer.datasets import SemSegDataset
from trainer.deeplab import Deeplabv3
from trainer.dicecoefficient import DiceCoefficient
from trainer.lossfunction import SemanticSegmentationLoss
from trainer.utils import setup_system
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, Resize, CenterCrop, Normalize, RandomCrop, HorizontalFlip, ShiftScaleRotate, HueSaturationValue
from albumentations.pytorch import ToTensorV2, ToTensor
# optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from operator import itemgetter
from trainer import Trainer, hooks

class Training_model:
    def __init__(
            self,
            # init configs
            system_config: SystemConfig = SystemConfig(),
            dataset_config: DatasetConfig = DatasetConfig(),
            dataloader_config: DataloaderConfig = DataloaderConfig(),
            optimizer_config: OptimizerConfig = OptimizerConfig(),
    ):
        # create test dataset
        data_path = "Data/"
        images_folder = "imgs/imgs"
        masks_folder = "masks/masks"
        csv_path = "train"
        train_val_test = "train"
        # apply system settings
        self.system_config = system_config
        setup_system(system_config)
        # define train dataloader
        self.loader_train = DataLoader(
            # define our dataset
            SemSegDataset(
                data_path,
                images_folder,
                masks_folder,
                csv_path,
                "train",
                # define augmentations
                transforms=Compose([
                    Resize(330, 330),
                    CenterCrop(330, 330),
                    HorizontalFlip(),
                    HueSaturationValue(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            ),
            batch_size=dataloader_config.batch_size,
            shuffle=True,
            num_workers=dataloader_config.num_workers,
            pin_memory=True
        )
        # define test dataloader
        self.loader_test = DataLoader(
            SemSegDataset(

                data_path,
                images_folder,
                masks_folder,
                csv_path,
                "val",
                transforms=Compose([
                    Resize(330, 330),
                    CenterCrop(330, 330),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()])
            ),
            batch_size=dataloader_config.batch_size,
            shuffle=False,
            num_workers=dataloader_config.num_workers,
            pin_memory=True
        )
        num_classes = 12
        # define model
        self.model = Deeplabv3(
            num_classes_output=num_classes, pretrained=True, trained=True
        )

        # define loss
        self.loss_fn = SemanticSegmentationLoss(num_classes=num_classes, ignore_indices=num_classes)
        #         self.loss_fn = nn.CrossEntropyLoss(ignore_index=num_classes)

        # define metrics function as intersection over union
        self.metric_fn = DiceCoefficient(
            num_classes=num_classes, reduced_probs=False, normalized=False
        )
        # define optimizer and its params
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay
        )
        # define learning rate scheduler
        self.lr_scheduler = MultiStepLR(
            self.optimizer, milestones=optimizer_config.lr_step_milestones, gamma=optimizer_config.lr_gamma
        )

    # run training
    def run(self, trainer_config: TrainerConfig) -> dict:
        # apply system settings
        setup_system(self.system_config)
        # move training to the chosen device
        device = torch.device(trainer_config.device)
        # send data to chosen device
        self.model = self.model.to(device)
        self.loss_fn = self.loss_fn.to(device)

        # define trainer
        model_trainer = Trainer(
            model=self.model,
            loader_train=self.loader_train,
            loader_test=self.loader_test,
            loss_fn=self.loss_fn,
            metric_fn=self.metric_fn,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            data_getter=itemgetter("image"),
            target_getter=itemgetter("mask"),
            stage_progress=trainer_config.progress_bar,
            get_key_metric=itemgetter("dice"),
            model_saving_frequency=trainer_config.model_saving_frequency,
            save_dir=trainer_config.model_dir
        )

        # define hook to run after each epoch
        model_trainer.register_hook("end_epoch", hooks.end_epoch_hook_semseg)
        # run the training
        self.metrics = model_trainer.fit(trainer_config.epoch_num)
        return self.metrics