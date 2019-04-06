import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TextLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def ohem_L2(self, predict, target, weight, batch_size, negative_ratio=3.):
        # the amount of positive and negative pixels
        regionPos = weight > 0
        regionNeg = weight == 0
        sumPos = regionPos.sum()
        sumNeg = regionNeg.sum()
        # the amount of hard negative pixels
        # print(negative_ratio*sumPos,sumNeg,'aaa')
        sumhardNeg = min(negative_ratio*sumPos, sumNeg)
        # set loss on ~(top-sumhardNeg) negative pixels to 0
        loss_pos = F.mse_loss(predict[regionPos],
                              target[regionPos], reduction='none')
        lossNeg = F.mse_loss(predict[regionNeg],
                             target[regionNeg], reduction='none')
        lossNeg = lossNeg.sum(dim=1)
        lossNeg, _ = torch.topk(lossNeg, sumhardNeg)
        # weight for positive and negative pixels
        weightPos = torch.stack(
            [weight[regionPos], torch.clone(weight[regionPos])], dim=1)
        # weightNeg = (lossHard != 0).float()
        # weightNeg = torch.stack([weightNeg, torch.clone(weightNeg)], dim=1)

        # total loss
        # loss = torch.sum((distL1**2)*(weightPos + weightNeg)) / torch.sum(weightPos + weightNeg)
        # print(loss_pos.size(),weightPos.size(),lossNeg.size())
        # print((loss_pos*weightPos).sum(),lossNeg.sum())
        # print(batch_size,'bbb')
        loss = ((loss_pos*weightPos).sum()+lossNeg.sum()) / \
            (weightPos.sum()+sumhardNeg*2)/batch_size
        return loss

    def forward(self, predict, vec_mask, weight):
        """
        calculate textsnake loss
        :param predict: (Variable), network predict, (BS, 7, H, W)
        :param tr_mask: (Variable), TR target, (BS, H, W)
        :param tcl_mask: (Variable), TCL target, (BS, H, W)u
        :param sin_map: (Variable), sin target, (BS, H, W)
        :param cos_map: (Variable), cos target, (BS, H, W)
        :param radii_map: (Variable), radius target, (BS, H, W)
        :param train_mask: (Variable), training mask, (BS, H, W)
        :return: loss_tr, loss_tcl, loss_radii, loss_sin, loss_cos
        """
        # print(vec_mask.dtype,'aasvasfvasv',vec_mask[:, 0].size())
        # print(predict.size())
        # x_pred = predict[:, 0].contiguous().view(-1)  # (BSxHxW,)
        # y_pred = predict[:, 1].contiguous().view(-1)  # (BSxHxW,)
        # x_mask = vec_mask[:, 0].contiguous().view(-1)  # (BSxHxW,)
        # y_mask = vec_mask[:, 1].contiguous().view(-1)  # (BSxHxW,)
        # # print(x_pred.size(), x_mask.size(),weight.dtype)
        # weight = weight[:].contiguous().view(-1).float()# (BSxHxW,)
        # # print(x_pred.size(), x_mask.size(),weight.dtype)
        # loss_x = self.ohem_L1(x_pred, x_mask, weight)
        # loss_y = self.ohem_L1(y_pred, y_mask, weight)
        batch_size = predict.size(0)
        predict = predict.permute(
            0, 2, 3, 1).contiguous().view(-1, 2)  # (BSxHxW,2)
        vec_mask = vec_mask.permute(
            0, 2, 3, 1).contiguous().view(-1, 2)  # (BSxHxW,2)
        weight = weight.contiguous().view(-1)  # (BSxHxW,)
        loss = self.ohem_L2(predict, vec_mask, weight, batch_size)
        return loss
