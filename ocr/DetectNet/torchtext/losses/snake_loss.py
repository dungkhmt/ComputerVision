import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TextLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def ohem(self, predict, target, train_mask, negative_ratio=3.):
        pos = (target * train_mask).byte()
        neg = ((1 - target) * train_mask).byte()
        # print(predict.dtype, target.dtype, train_mask.dtype)
        n_pos = pos.float().sum()
        n_neg = min(int(neg.float().sum().item()),
                    int(negative_ratio * n_pos.float()))
        # print('aaaaa',predict[pos].size(),target[pos].size())
        # print(pos.size(),predict[pos][0][:],target[pos][0])
        # print(neg.size(),predict[neg][0][:],target[neg][0])
        loss_pos = F.cross_entropy(predict[pos], target[pos], reduction='sum')
        loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
        loss_neg, _ = torch.topk(loss_neg, n_neg)
        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    def ohem_L1(self, predict, target, tr_mask, train_mask, negative_ratio=3.):
        pos = (tr_mask * train_mask).byte()
        neg = ((1 - tr_mask) * train_mask).byte()
        # print(predict.dtype, target.dtype, train_mask.dtype)
        n_pos = pos.float().sum()
        n_neg = min(int(neg.float().sum().item()),
                    int(negative_ratio * n_pos.float()))
        # print('aaaaa',predict[pos].size(),target[pos].size())
        # print(pos.size(),predict[pos][0][:],target[pos][0])
        # print(neg.size(),predict[neg][0][:],target[neg][0])
        loss_pos = F.smooth_l1_loss(predict[pos], target[pos], reduction='sum')
        loss_neg = F.smooth_l1_loss(
            predict[neg], target[neg], reduction='none')
        loss_neg, _ = torch.topk(loss_neg, n_neg)
        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    def forward(self, predict, tr_mask, tcl_mask, sin_map, cos_map, radii_map, train_mask, x_mask, y_mask):
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
        # print(predict.size())
        tr_pred = predict[:, :2].permute(
            0, 2, 3, 1).contiguous().view(-1, 2)  # (BSxHxW, 2)
        # print(tr_pred.size(),tr_pred.dtype)
        tcl_pred = predict[:, 2:4].permute(
            0, 2, 3, 1).contiguous().view(-1, 2)  # (BSxHxW, 2)
        # sin_pred = predict[:, 4].contiguous().view(-1)  # (BSxHxW,)
        # cos_pred = predict[:, 5].contiguous().view(-1)  # (BSxHxW,)

        # regularize sin and cos: sum to 1
        # scale = torch.sqrt(1.0 / (sin_pred ** 2 + cos_pred ** 2))
        # sin_pred = sin_pred * scale
        # cos_pred = cos_pred * scale

        # radii_pred = predict[:, 6].contiguous().view(-1)  # (BSxHxW,)

        x_pred = predict[:, 4].contiguous().view(-1)
        y_pred = predict[:, 5].contiguous().view(-1)
        x_mask = x_mask.contiguous().view(-1)
        y_mask = y_mask.contiguous().view(-1)

        train_mask = train_mask.view(-1)  # (BSxHxW,)

        tr_mask = tr_mask.contiguous().view(-1)
        tcl_mask = tcl_mask.contiguous().view(-1)
        # radii_map = radii_map.contiguous().view(-1)
        # sin_map = sin_map.contiguous().view(-1)
        # cos_map = cos_map.contiguous().view(-1)
        # print(tr_pred.size(), tr_mask.size(),train_mask.size())
        # loss_tr = F.cross_entropy(tr_pred[train_mask], tr_mask[train_mask].long())
        loss_tr = self.ohem(tr_pred, tr_mask.long(), train_mask.long())

        loss_tcl = self.ohem(tcl_pred, tcl_mask.long(), train_mask.long())
        # loss_tcl = F.cross_entropy(tcl_pred[train_mask * tr_mask],
        #                            tcl_mask[train_mask * tr_mask].long())
        # loss_tcl = F.cross_entropy(tcl_pred,
        #                            tcl_mask.long())

        # geometry losses
        # ones = radii_map.new(radii_pred[tcl_mask].size()).fill_(1.).float()
        # loss_radii = F.smooth_l1_loss(
        #     radii_pred[tcl_mask] / radii_map[tcl_mask], ones)
        # print(torch.max(radii_pred),torch.max(radii_map))
        # loss_sin = F.smooth_l1_loss(sin_pred[tcl_mask], sin_map[tcl_mask])
        # loss_cos = F.smooth_l1_loss(cos_pred[tcl_mask], cos_map[tcl_mask])

        # print(x_pred.dtype, x_mask.dtype)
        # loss_x = F.smooth_l1_loss(x_pred[tcl_mask], x_mask[tcl_mask])
        # loss_y = F.smooth_l1_loss(y_pred[tcl_mask], y_mask[tcl_mask])
        loss_x = self.ohem_L1(x_pred, x_mask, tr_mask, train_mask)
        loss_y = self.ohem_L1(y_pred, y_mask, tr_mask, train_mask)

        # ones_x = x_mask.new(x_mask[tcl_mask].size()).fill_(1.).float()
        # loss_x = F.smooth_l1_loss(x_pred[tcl_mask] / x_mask[tcl_mask], ones_x)
        # ones_y = y_mask.new(y_mask[tcl_mask].size()).fill_(1.).float()
        # loss_y = F.smooth_l1_loss(y_pred[tcl_mask] / y_mask[tcl_mask], ones_y)
        return loss_tr, loss_tcl, loss_x/2, loss_y/2
