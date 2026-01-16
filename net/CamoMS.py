import torch
import torch.nn as nn
import torch.nn.functional as F

from net.smt import smt_b


# Conv+Norm+Relu
class ConvBNR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0, dilation=1, groups=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# Preliminary Mask Generation(PMG)
class MaskGeneration(nn.Module):
    def __init__(self, in_channel, hid_channel):
        super(MaskGeneration, self).__init__()
        self.res_1_1 = ConvBNR(in_channel, hid_channel, 1, 1)
        self.Convblock = nn.Sequential(
            ConvBNR(in_channel, hid_channel, 1, 1),
            ConvBNR(hid_channel, hid_channel, 3, 1, padding=1)
        )
        self.proj_1_1 = nn.Conv2d(hid_channel, 1, (1, 1), (1, 1))

    def forward(self, x1, x2):
        x2 = F.interpolate(x2, x1.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x1, x2), dim=1)

        mx = self.res_1_1(x) + self.Convblock(x)
        mask = torch.sigmoid(self.proj_1_1(mx))

        return mask


# Foreground Encoder
class ForegroundEncoder(nn.Module):
    def __init__(self, channel):
        super(ForegroundEncoder, self).__init__()

        self.proj = nn.Identity()

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // 2, (1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // 2, channel, (1, 1), bias=False)
        )

        # Spatial Attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=(3, 3), padding=1)
        )

    def forward(self, x):
        fg = self.proj(x)

        avgp = self.mlp(self.avg_pool(fg))
        maxp = self.mlp(self.max_pool(fg))
        ca = torch.sigmoid(avgp+maxp)
        fg = fg * ca

        sa = torch.sigmoid(self.spatial_att(fg))
        fg = fg * sa

        return fg


# Background Encoder
class BackgroundEncoder(nn.Module):
    def __init__(self, channel):
        super(BackgroundEncoder, self).__init__()

        self.proj = nn.Identity()

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // 2, (1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // 2, channel, (1, 1), bias=False)
        )

        # Spatial Attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=(3, 3), padding=1)
        )

    def forward(self, x):
        bg = self.proj(x)

        avgp = self.mlp(self.avg_pool(bg))
        maxp = self.mlp(self.max_pool(bg))
        ca = torch.sigmoid(avgp+maxp)
        bg = bg * ca

        sa = torch.sigmoid(self.spatial_att(bg))
        bg = bg * sa

        return bg


# Mask Guided Separation Learning(MGSL)
class FeatureDecoupling(nn.Module):
    def __init__(self, channel):
        super(FeatureDecoupling, self).__init__()
        self.fgEnc = ForegroundEncoder(channel)
        self.bgEnc = BackgroundEncoder(channel)
        self.proj = ConvBNR(channel, channel, 1, 1)

    def forward(self, x, mask):
        if x.size()[2:] != mask.size()[2:]:
            mask = F.interpolate(mask, x.size()[2:], mode='bilinear', align_corners=False)

        fg = self.fgEnc(x*mask)
        bg = self.bgEnc(x*(1-mask))
        return self.proj(fg+bg)


# Selective Channel Fusion with Attention(SCFA)
class FeatureFusion(nn.Module):
    def __init__(self, channel, hw, k):
        super(FeatureFusion, self).__init__()
        self.Qurey = nn.Linear(hw, 256)
        self.Key = nn.Linear(hw, 256)

        self.topk = k
        self.scale = torch.sqrt(torch.FloatTensor([256]))

        self.proj_conv = ConvBNR(channel, 256, 1, 1)

    def forward(self, x1, x2):
        x2 = F.interpolate(x2, x1.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x1, x2), dim=1)
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1)

        device = x.device
        scale = self.scale.to(device)
        query = self.Qurey(x)
        key = self.Key(x)
        att = torch.matmul(query, key.permute(0, 2, 1)) / scale

        diagonal_values = att.diagonal(dim1=-2, dim2=-1)
        att_modified = att.clone()
        att_modified.diagonal(dim1=-2, dim2=-1).fill_(-1e9)
        topk_indices = torch.topk(att_modified, k=self.topk, dim=-1).indices
        topk_values = att_modified.gather(dim=-1, index=topk_indices)
        top_att = torch.full_like(att, fill_value=-1e9)
        top_att.scatter_(-1, topk_indices, topk_values)
        top_att.diagonal(dim1=-2, dim2=-1).copy_(diagonal_values)

        sigmoid_att = torch.sigmoid(top_att)

        attx = torch.matmul(sigmoid_att, x)

        outx = (attx + x).reshape(b, c, h, w)

        return self.proj_conv(outx)


# Local Feature Refinement(LFR)
class FeatureRefinement(nn.Module):
    def __init__(self):
        super(FeatureRefinement, self).__init__()
        self.conv1 = ConvBNR(512, 256, 1, 1)
        self.bottleneck1 = ConvBNR(64, 64, 3, 1, padding=1, dilation=1)
        self.bottleneck2 = ConvBNR(64, 64, 3, 1, padding=2, dilation=2)
        self.bottleneck3 = ConvBNR(64, 64, 3, 1, padding=3, dilation=3)
        self.bottleneck4 = ConvBNR(64, 64, 3, 1, padding=4, dilation=4)
        self.conv2 = ConvBNR(256, 256, 1, 1)
        self.conv3 = ConvBNR(256, 256, 3, 1, 1)

    def forward(self, x, rx):
        rx = F.interpolate(rx, x.size()[2:], mode='bilinear', align_corners=False)

        xm = self.conv1(torch.cat((x, rx), dim=1))

        s = torch.chunk(xm, 4, dim=1)

        s0 = self.bottleneck1(s[0]+s[3])
        s1 = self.bottleneck2(s[1]+s0)
        s2 = self.bottleneck3(s[2]+s1)
        s3 = self.bottleneck4(s[3]+s2)

        xs = self.conv2(torch.cat((s0, s1, s2, s3), dim=1))

        xx = self.conv3(xs + x)

        return xx


# Prediction
class MapPrediction(nn.Module):
    def __init__(self):
        super(MapPrediction, self).__init__()
        self.conv = nn.Conv2d(256, 1, (1, 1), (1, 1))

    def forward(self, x):
        return self.conv(x)


class CamoMS(nn.Module):
    def __init__(self,
                 dims=[64, 128, 256, 512],
                 pdims=[64, 128, 128, 256, 256, 512],
                 hw=[10816, 2704, 676],
                 k=[5, 7, 9],
                 adim=[64+128, 128+256, 256+512]):
        super(CamoMS, self).__init__()

        # Backbone
        self.encoder = smt_b()

        # PMG
        self.MaskGenerationLayers = nn.ModuleList()
        for i in range(3):
            MaskGenerationLayer = MaskGeneration(in_channel=dims[i]+dims[i+1], hid_channel=dims[i])
            self.MaskGenerationLayers.append(MaskGenerationLayer)

        # MGSL
        self.FeatureDecouplingLayers = nn.ModuleList()
        for i in range(6):
            FeatureDecouplingLayer = FeatureDecoupling(pdims[i])
            self.FeatureDecouplingLayers.append(FeatureDecouplingLayer)

        # SCFA
        self.conv_xa4 = ConvBNR(512, 256, 1, 1)
        self.FeatureFusionLayers = nn.ModuleList()
        for i in range(3):
            FeatureFusionLayer = FeatureFusion(adim[i], hw[i], k[i])
            self.FeatureFusionLayers.append(FeatureFusionLayer)

        # LFR
        self.FeatureRefinementLayers = nn.ModuleList()
        for i in range(3):
            FeatureRefinementLayer = FeatureRefinement()
            self.FeatureRefinementLayers.append(FeatureRefinementLayer)

        # Prediction
        self.MapPredictionLayers = nn.ModuleList()
        for i in range(3):
            MapPredictionLayer = MapPrediction()
            self.MapPredictionLayers.append(MapPredictionLayer)

    def load_pre(self, pre_model):
        self.encoder.load_state_dict(torch.load(pre_model)['model'])
        print(f"loading pre_model ${pre_model}")

    def forward(self, x):
        features = self.encoder(x)
        f1 = features[0]
        f2 = features[1]
        f3 = features[2]
        f4 = features[3]

        mask1 = self.MaskGenerationLayers[0](f1, f2)
        mask2 = self.MaskGenerationLayers[1](f2, f3)
        mask3 = self.MaskGenerationLayers[2](f3, f4)

        f1m1 = self.FeatureDecouplingLayers[0](f1, mask1)
        f2m1 = self.FeatureDecouplingLayers[1](f2, mask1)
        f2m2 = self.FeatureDecouplingLayers[2](f2, mask2)
        f3m2 = self.FeatureDecouplingLayers[3](f3, mask2)
        f3m3 = self.FeatureDecouplingLayers[4](f3, mask3)
        f4m3 = self.FeatureDecouplingLayers[5](f4, mask3)

        fusion1 = self.FeatureFusionLayers[0](f1m1, f2m1)
        fusion2 = self.FeatureFusionLayers[1](f2m2, f3m2)
        fusion3 = self.FeatureFusionLayers[2](f3m3, f4m3)

        xa4 = self.conv_xa4(f4)
        xa3 = self.FeatureRefinementLayers[2](fusion3, xa4)
        map3 = self.MapPredictionLayers[2](xa3)
        xa2 = self.FeatureRefinementLayers[1](fusion2, xa3)
        map2 = self.MapPredictionLayers[1](xa2)
        xa1 = self.FeatureRefinementLayers[0](fusion1, xa2)
        map1 = self.MapPredictionLayers[0](xa1)

        map3 = F.interpolate(map3, scale_factor=16, mode='bilinear', align_corners=False)
        map2 = F.interpolate(map2, scale_factor=8, mode='bilinear', align_corners=False)
        map1 = F.interpolate(map1, scale_factor=4, mode='bilinear', align_corners=False)

        mask3 = F.interpolate(mask3, scale_factor=16, mode='bilinear', align_corners=False)
        mask2 = F.interpolate(mask2, scale_factor=8, mode='bilinear', align_corners=False)
        mask1 = F.interpolate(mask1, scale_factor=4, mode='bilinear', align_corners=False)

        return [map1, map2, map3], [mask1, mask2, mask3]


if __name__ == '__main__':
    bs = torch.rand(1, 3, 416, 416).cuda()

    net = CamoMS().cuda()
    map_list, mask_list = net(bs)

    print(len(map_list), len(mask_list))

