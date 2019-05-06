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


class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels, multiply):
        super().__init__()
        self.conv5x5 = nn.Conv2d(
            in_channels, in_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1x1_1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1_2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.multiply = multiply
        if multiply > 1:
            ratio = multiply/2
            self.deconv = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=int(4*ratio), stride=int(2*ratio), padding=int(1*ratio))

    def forward(self, x):
        x = self.conv5x5(x)
        x = self.relu(x)
        x = self.conv1x1_1(x)
        x = self.relu(x)
        x = self.conv1x1_2(x)
        x = self.relu(x)
        if self.multiply > 1:
            x = self.deconv(x)
        return x


resnet = {'resnet34': resnet34, 'resnet50': resnet50,
          'resnet101': resnet101, 'resnet152': resnet152}
vgg = {'vgg16': vgg16}

se_net = {'se_resnext50_32x4d': se_resnext50_32x4d}


class TextFieldNet(nn.Module):

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
        elif backbone == 'se_resnext50_32x4d':
            self.backbone = se_net[backbone]()
            self._make_upsample(expansion=4)

    def _make_upsample(self, expansion=1):
        if self.backbone_name == 'vgg16':
            self.up5x4 = Upsample(512, 256, 4)
            self.up4x2 = Upsample(512, 256, 2)
            self.up3x1 = Upsample(256, 256, 1)
            self.up2x2 = nn.Sequential(
                nn.Conv2d(256*3, 512, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 2, kernel_size=1,
                          stride=1, padding=0, bias=False)
            )
            self.up1x4 = nn.ConvTranspose2d(
                2, 2, kernel_size=8, stride=4, padding=2, bias=False)
        else:
            pass

    def forward(self, x):
        # print('xxxxxxxxxxxx size',x.size())
        C1, C2, C3, C4, C5 = self.backbone(x)
        # print('c1', C1.size(), 'c2', C2.size(), 'c3',
        #       C3.size(), 'c4', C4.size(), 'c5', C5.size())
        up5 = self.up5x4(C5)
        # print('up5',up5.size())
        up4 = self.up4x2(C4)
        # print('up4',up4.size())
        up3 = self.up3x1(C3)

        concat = torch.cat([up5, up4, up3], dim=1)
        # concat = self.concatx2(concat)
        # print('concat',concat.size())
        up2 = self.up2x2(concat)
        # print('up2',up2.size())
        up1 = self.up1x4(up2)
        # print(up1.size())
        return up1


if __name__ == '__main__':
    import torch
    # input = torch.randn((1, 3, 768, 768))
    net = TextFieldNet(backbone='vgg16')
    # print(net(input).size())
    import torchsummary
    with torch.no_grad():
        print(torchsummary.summary(net, (3, 64, 64), batch_size=1, device='cpu'))
    exit()
