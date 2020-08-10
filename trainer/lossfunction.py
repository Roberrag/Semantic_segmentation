# define soft-Jaccard loss
# import primitives to build our custom model
import torch.nn as nn
class SoftJaccardLoss(nn.Module):
    """
        Implementation of the Soft-Jaccard Loss function.

        Arguments:
            num_classes (int): number of classes.
            eps (float): value of the floating point epsilon.
    """
    def __init__(self, num_classes, eps=1e-5):
        super().__init__()
        # init fields
        self.num_classes = num_classes
        self.eps = eps

    # define forward pass
    def forward(self, pred_logits, targets):
        """
            Compute Soft-Jaccard Loss.

            Arguments:
                pred_logits (torch.FloatTensor): tensor of predicted logits. The shape of the tensor is (B, num_classes, H, W).
                targets (torch.LongTensor): tensor of ground-truth labels. The shape of the tensor is (B, 1, H, W).
        """
        # get predictions from logits
        preds = pred_logits.softmax(dim=1)
#         print(preds[:,11])
        loss = 0
        # iterate over all classes
        for cls in range(self.num_classes):
#             print(cls)
            # get ground truth for the current class
            target = (targets == cls).float()

            # get prediction for the current class
            pred = preds[:, cls]

            # calculate intersection of predictions and targets
            intersection = (pred * target).sum()

            # compute iou
            iou = (intersection + self.eps) / (pred.sum() + target.sum() - intersection + self.eps)

            # compute negative logarithm from the obtained dice coefficient
            loss = loss - iou.log()

            # get mean loss by class value
            loss = loss / self.num_classes

        return loss


# define Focal loss
class FocalLoss(nn.Module):
    """
        Implementation of the Focal Loss function.

        Arguments:
            num_classes (int): number of classes.
                valid_labels: 0,..., num_classes - 1
                ignore_index: num_classes
            gamma (float): parameter which control slope of loss function.
    """
    def __init__(self, num_classes, gamma=2.0, ignore_indices=-1):
        super().__init__()
        # init fields
        self.num_classes = num_classes
        self.gamma = gamma
        self.loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_indices)

    # define forward pass
    def forward(self, pred_logits, target):
        """
            Compute Focal Loss.

            Arguments:
                pred_logits (torch.FloatTensor): tensor of predicted logits. The shape of the tensor is
                (B, num_classes, H, W).
                targets (torch.LongTensor): tensor of ground-truth labels. The shape of the tensor is
                (B, 1, H, W).
        """
        loss_ce = self.loss_fn(pred_logits, target)
        loss_focal = (1.0 - loss_ce.mul(-1).exp()).pow(self.gamma) * loss_ce
        return loss_focal.mean()

# define semantic segmentation loss
class SemanticSegmentationLoss(nn.Module):
    """
        Implementation of the multi-objective loss function for semantic segmentation.

        Arguments:
            num_classes (int): number of classes.
            jaccard_alpha (float): weight of the SoftJaccardLoss
    """
    def __init__(self, num_classes, jaccard_alpha=0.9, ignore_indices=-1):
        super().__init__()
        # init fields
        self.jaccard_alpha = jaccard_alpha
        self.jaccard = SoftJaccardLoss(num_classes)
        self.focal = FocalLoss(num_classes=num_classes, ignore_indices=ignore_indices)

    # define forward pass
    def forward(self, pred_logits, target):
        """
            Compute Focal Loss.

            Arguments:
                pred_logits (torch.FloatTensor): tensor of predicted logits. The shape of the tensor is (B, num_classes, H, W).
                targets (torch.LongTensor): tensor of ground-truth labels. The shape of the tensor is (B, 1, H, W).
        """
        # our loss is a weighted sum of two losses
        jaccard_loss = self.jaccard_alpha * self.jaccard(pred_logits, target)
        focal_loss = self.focal(pred_logits, target)
        loss = jaccard_loss + focal_loss
        return loss