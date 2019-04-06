import torch.nn as nn
import torch
import torch.nn.functional as F
from vgg import *
from resnet import *


class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
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


class TextNet(nn.Module):

    def __init__(self, backbone='vgg16', output_channel=6):
        super().__init__()

        self.backbone_name = backbone
        self.output_channel = output_channel
        self.relu = nn.ReLU()

        if backbone[0:3] == 'vgg':
            self.backbone = vgg[backbone]()
            self.deconv5 = nn.ConvTranspose2d(
                512, 512, kernel_size=4, stride=2, padding=1)
            self.merge4 = Upsample(512 + 512, 256)
            self.merge3 = Upsample(256 + 256, 128)
            self.merge2 = Upsample(128 + 128, 64)
            self.merge1 = Upsample(64 + 64, 32)
            self.conv1x1 = nn.Conv2d(
                32, 32, kernel_size=1, stride=1, padding=0)
            self.conv3x3 = nn.Conv2d(
                32, self.output_channel, kernel_size=3, stride=1, padding=1)
        elif backbone[0:6] == 'resnet':
            self.backbone = resnet[backbone]()
            expansion = 1 if backbone in ['resnet18', 'resnet34'] else 4
            self.deconv5 = nn.ConvTranspose2d(
                512*expansion, 512*expansion, kernel_size=4, stride=2, padding=1)
            self.merge4 = Upsample(512*expansion+256*expansion, 256*expansion)
            self.merge3 = Upsample(
                256*expansion + 128*expansion, 128*expansion)
            self.merge2 = Upsample(128*expansion + 64*expansion, 64*expansion)
            self.merge1 = Upsample(64*expansion + 64, 32)
            self.conv1x1 = nn.Conv2d(
                32, 32, kernel_size=1, stride=1, padding=0)
            self.conv3x3 = nn.Conv2d(
                32, self.output_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        C1, C2, C3, C4, C5 = self.backbone(x)
        up5 = self.deconv5(C5)
        up5 = self.relu(up5)

        # print('c4 up5',C4.size(),up5.size())
        up4 = self.merge4(C4, up5)
        up4 = self.relu(up4)

        # print('c3 up4',C3.size(),up4.size())
        up3 = self.merge3(C3, up4)
        up3 = self.relu(up3)

        # print('c2 up3',C2.size(),up3.size())
        up2 = self.merge2(C2, up3)
        up2 = self.relu(up2)

        # print('c1 up2',C1.size(),up2.size())
        up1 = self.merge1(C1, up2)
        up1 = self.relu(up1)
        
        up1 = self.conv1x1(up1)
        up1 = self.conv3x3(up1)

        return up1


if __name__ == '__main__':
    import torch
    input = torch.randn((3, 3, 512, 512))
    net = TextNet(backbone='resnet50').cuda()
    print(net(input.cuda()).size())
    import torchsummary
    print(torchsummary.summary(net, (3, 512, 512),batch_size=2))
    exit()
