import torch
import torch.nn as nn
import torch.nn.functional as F
from network.Res2Net_v1b import res2net50_v1b_26w_4s


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class FeatureAggregationDecoder(nn.Module):
    def __init__(self, channel):
        super(FeatureAggregationDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class RAP(nn.Module):
    def __init__(self, channel):
        super(RAP, self).__init__()
        self.channel = channel
        self.conv1 = BasicConv2d(self.channel, self.channel, kernel_size=3, padding=1)
        self.conv1_1 = BasicConv2d(self.channel, 1, kernel_size=1, padding=0)
        self.conv2 = BasicConv2d(self.channel, self.channel, kernel_size=3, padding=1)
        self.conv2_1 = BasicConv2d(self.channel, 1, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, cfeat, x):
        x1 = cfeat.expand(-1, self.channel, -1, -1).mul(x)
        x2 = torch.abs(x - x1)

        out1 = self.relu(self.conv1(x1)) + x
        out2 = self.relu(self.conv2(x2)) + x
        out = self.conv1_1(out1) + self.conv2_1(out2)
        return out + cfeat


class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(Network, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        # ---- Partial Decoder ----
        self.FAD = FeatureAggregationDecoder(channel)

        # # ---- reverse stage ----
        self.RAP5 = RAP(32)
        self.RAP4 = RAP(32)
        self.RAP3 = RAP(32)

    def forward(self, x):
        # Feature Extraction
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11

        # Receptive Field Block (enhanced)
        x2_rfb = self.rfb2_1(x2)  # channel -> 32
        x3_rfb = self.rfb3_1(x3)  # channel -> 32
        x4_rfb = self.rfb4_1(x4)  # channel -> 32

        # Neighbourhood Connected Decoder
        S_g = self.FAD(x4_rfb, x3_rfb, x2_rfb)
        S_g_pred = F.interpolate(S_g, scale_factor=8, mode='bilinear')  # Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        #
        # # ---- reverse stage 5 ----
        guidance_g = F.interpolate(S_g, scale_factor=0.25, mode='bilinear')
        ra4_feat = self.RAP5(guidance_g, x4_rfb)
        S_5 = ra4_feat + guidance_g
        S_5_pred = F.interpolate(S_5, scale_factor=32, mode='bilinear')  # Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse stage 4 ----
        guidance_5 = F.interpolate(S_5, scale_factor=2, mode='bilinear')
        ra3_feat = self.RAP4(guidance_5, x3_rfb)
        S_4 = ra3_feat + guidance_5
        S_4_pred = F.interpolate(S_4, scale_factor=16, mode='bilinear')

        # ---- reverse stage 3 ----
        guidance_4 = F.interpolate(S_4, scale_factor=2, mode='bilinear')
        ra2_feat = self.RAP3(guidance_4, x2_rfb)
        S_3 = ra2_feat + guidance_4
        S_3_pred = F.interpolate(S_3, scale_factor=8, mode='bilinear')

        return S_g_pred, S_5_pred, S_4_pred, S_3_pred


if __name__ == '__main__':
    net = Network()
    net.eval()

    dump_x = torch.randn(1, 3, 352, 352)
    y = net(dump_x)
    print(y[0].shape, y[1].shape, y[2].shape, y[3].shape)
