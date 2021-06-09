import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self, n_classes, encoder_dim, img_size, low_level_dim=256, rates=[1, 6, 12, 18]
    ):
        super(Decoder, self).__init__()
        self.img_size = img_size

        self.aspp1 = ASPP_module(encoder_dim, 256, rate=rates[0])
        self.aspp2 = ASPP_module(encoder_dim, 256, rate=rates[1])
        self.aspp3 = ASPP_module(encoder_dim, 256, rate=rates[2])
        self.aspp4 = ASPP_module(encoder_dim, 256, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(encoder_dim, 256, 1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # adopt [1x1, 256] for channel reduction.
        self.conv1 = nn.Conv2d(256 * 5, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(low_level_dim, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, n_classes, kernel_size=1, stride=1),
        )

    def forward(self, x, low_level_feat):
        # x = output of encoder
        # low_level_feat = end of 'layer1' encoder
        input_size = (low_level_feat.shape[-2] * 4, low_level_feat.shape[-1] * 4)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(
            x, size=low_level_feat.size()[2:], mode="bilinear", align_corners=True
        )

        low_level_feat = self.conv2(low_level_feat)
        low_level_feat = self.bn2(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)

        return x


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=rate,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    # n_classes, encoder_dim, img_size, low_level_dim=256, rates=[1, 6, 12, 18]
    model = Decoder(21, 256, (512, 512), low_level_dim=256, rates=[1, 6, 12, 18])
    model.eval()
    print(model)
    x = torch.randn(1, 256, 512 // 2 ** 5, 512 // 2 ** 5)
    low_level_feat = torch.randn(1, 256, 512, 512)
    with torch.no_grad():
        output = model.forward(x, low_level_feat)
    print(output.size())
