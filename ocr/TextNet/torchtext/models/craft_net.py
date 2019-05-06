import torch.nn as nn
import torch
import torch.nn.functional as F
try:
    from vgg import *
    from resnet import *
    from senet import *
except:
    from .vgg import *
    from .resnet import *
    from .senet import *


class MergeUpsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.conv1x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3x3_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.deconv = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv3x3_1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3x3_2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.deconv(x)
        return x


class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels, multiply):
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.multiply = multiply
        if multiply > 1:
            ratio = multiply/2
            self.deconv = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=int(4*ratio), stride=int(2*ratio), padding=int(1*ratio))

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv3x3(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.deconv(x)
        return x


resnet = {'resnet34': resnet34, 'resnet50': resnet50,
          'resnet101': resnet101, 'resnet152': resnet152}
vgg = {'vgg16': vgg16}

se_net = {'se_resnext50_32x4d': se_resnext50_32x4d,
          'se_resnext101_32x4d': se_resnext101_32x4d}


class CRAFTNet(nn.Module):

    def __init__(self, backbone='vgg16', output_channel=2):
        super().__init__()

        self.backbone_name = backbone
        self.output_channel = output_channel

        if backbone[0:3] == 'vgg':
            self.backbone = vgg[backbone]()
            self._make_upsample(expansion=1)
        elif backbone[0:6] == 'resnet':
            self.backbone = resnet[backbone]()
            expansion = 1 if backbone in ['resnet18', 'resnet34'] else 4
            self._make_upsample(expansion=expansion)
        elif backbone == 'se_resnext50_32x4d' or backbone == 'se_resnext101_32x4d':
            self.backbone = se_net[backbone]()
            self._make_upsample(expansion=4)

    def _make_upsample(self, expansion=1):
        if self.backbone_name == 'vgg16':
            self.deconv5 = nn.ConvTranspose2d(
                512, 512, kernel_size=4, stride=2, padding=1)
            self.merge4 = MergeUpsample(512 + 512, 256)
            self.merge3 = MergeUpsample(256 + 256, 128)
            self.merge2 = MergeUpsample(128 + 128, 64)
            self.merge1 = MergeUpsample(64 + 64, 64)
            self.predict = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(64, self.output_channel,
                          kernel_size=1, stride=1, padding=0)
            )
        else:
            self.deconv5 = Upsample(512*expansion, int(512*expansion/2), 2)
            self.merge4 = MergeUpsample(
                512*expansion/2+256*expansion, 256*expansion/2)
            self.merge3 = MergeUpsample(
                256*expansion/2 + 128*expansion, 128*expansion/2)
            self.merge2 = MergeUpsample(
                128*expansion/2 + 64*expansion, 64*expansion/2)
            self.merge1 = MergeUpsample(64*expansion/2 + 64, 64)
            self.predict = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(32, self.output_channel,
                          kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x):
        C1, C2, C3, C4, C5 = self.backbone(x)
        up5 = self.deconv5(C5)
        up4 = self.merge4(C4, up5)
        up3 = self.merge3(C3, up4)
        up2 = self.merge2(C2, up3)
        up1 = self.merge1(C1, up2)
        predict = self.predict(up1)
        return predict


if __name__ == '__main__':
    import torch
    input = torch.randn((1, 3, 512, 512)).cuda()
    net = CRAFTNet(backbone='vgg16').cuda()
    print(net(input).size())
    import torchsummary
    # with torch.no_grad():
    print(torchsummary.summary(net, (3, 64, 64), batch_size=1,device='cuda'))
    exit()
