import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TextLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def ohem_L2(self, predict, target, weight, negative_ratio):
        # the amount of positive and negative pixels
        regionPos = weight > 0
        regionNeg = weight == 0
        sumPos = regionPos.sum()
        sumNeg = regionNeg.sum()
        # the amount of hard negative pixels
        # print(negative_ratio*sumPos,sumNeg,'aaa')
        sumhardNeg = min(negative_ratio*sumPos, sumNeg)
        if sumPos == 0:
            sumhardNeg = 1
        # set loss on ~(top-sumhardNeg) negative pixels to 0
        loss_pos = F.mse_loss(predict[regionPos],
                              target[regionPos], reduction='none')
        lossNeg = F.mse_loss(predict[regionNeg],
                             target[regionNeg], reduction='none')
        loss_pos = loss_pos.sum(dim=1)
        lossNeg = lossNeg.sum(dim=1)
        lossNeg, _ = torch.topk(lossNeg, sumhardNeg)
        # weight for positive and negative pixels
        weightPos = weight[regionPos]

        # total loss
        loss = ((loss_pos*weightPos).sum()+lossNeg.sum()) / \
            (weightPos.sum()+sumhardNeg)
        return loss

    def forward(self, predict, vec_mask, weight,negative_ratio):
        """
        calculate textsnake loss
        :param predict: (Variable), network predict, (BS, 4, H, W)
        :param vec_mask: (Variable), vec word target, (BS, 2, H, W)
        :param weight: (Variable), weight word loss, (BS, H, W)
        """
        #batch_size = predict.size(0)

        word_predict = predict.permute(
            0, 2, 3, 1).contiguous().view(-1, 2)  # (BSxHxW,2)
        vec_mask = vec_mask.permute(
            0, 2, 3, 1).contiguous().view(-1, 2)  # (BSxHxW,2)
        weight = weight.contiguous().view(-1)  # (BSxHxW,)
        loss_word = self.ohem_L2(word_predict, vec_mask, weight,negative_ratio)

        return loss_word  # /batch_size
