import torch
from torch import nn
import torch.nn.functional as F

# inspired from the torch geometry version


class DiceLoss(nn.Module):

    def __init__(self, use_background=True):
        super().__init__()
        self.use_background = use_background
        self.eps: float = 1e-12

    def forward(self, input, target):

        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        if self.use_background:

            target_one_hot = self.one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)
        else:
            target_one_hot = self.one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)
            target_one_hot = target_one_hot[:, 1:, :, :]
            input_soft = input_soft[:, 1:, :, :]

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)

    def one_hot(self, labels, num_classes, device, dtype):

        batch_size, height, width = labels.shape
        one_hot = torch.zeros(batch_size, num_classes, height, width,
                              device=device, dtype=dtype)

        return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + self.eps