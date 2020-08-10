# create resnet encoder module
import torch.nn as nn
# model zoo with pretrained models
import torchvision.models as models
from torchvision import transforms

class Deeplabv3(nn.Module):
    """ ResNet encoder.

        Arguments:

            pretrained (bool): if pretrained == True, ImageNet weights will be load.
    """

    def __init__(self, num_classes_output, pretrained=True, trained=True):
        super().__init__()
        # get Torchvision deeplab pretrained model
        if pretrained:
            self.module = models.segmentation.deeplabv3_resnet101(pretrained=1).train()
        else:
            self.module = models.segmentation.deeplabv3_resnet101(pretrained=0).train()
        # modify output size ussing num of classes
        if trained:
            self.module.aux_classifier[4] = nn.Conv2d(256, num_classes_output, kernel_size=1, stride=1)
            self.module.classifier[4] = nn.Conv2d(256, num_classes_output, kernel_size=1, stride=1)
        else:
            self.module.classifier[4] = nn.Conv2d(256, num_classes_output, kernel_size=1, stride=1)

    # define forward pass
    def forward(self, x):
        classifier = self.module(x)['out']

        return classifier