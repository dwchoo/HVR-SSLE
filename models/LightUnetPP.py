import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlockGroupNorm(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=16):
        super(ConvBlockGroupNorm, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class LightUNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, z_channels=64, out_channels=64, init_features=32, num_groups=16):
        super(LightUNetPlusPlus, self).__init__()
        features = init_features

        self.padding_multiply_factor = 4

        # Encoder
        input_dim = in_channels + z_channels
        #self.conv0_0 = ConvBlock(input_dim, features)
        #self.conv1_0 = ConvBlock(features, features * 2)
        #self.conv2_0 = ConvBlock(features * 2, features * 4)
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ## Upsampling
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        ## Nested Skip Pathways
        #self.conv0_1 = ConvBlock(features + features * 2, features)
        #self.conv1_1 = ConvBlock(features * 2 + features * 4, features * 2)
        #self.conv0_2 = ConvBlock(features + features + features * 2, features)

        self.conv0_0 = ConvBlockGroupNorm(input_dim, features, num_groups=num_groups)
        self.conv1_0 = ConvBlockGroupNorm(features, features * 2, num_groups=num_groups)
        self.conv2_0 = ConvBlockGroupNorm(features * 2, features * 4, num_groups=num_groups)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Nested Skip Pathways
        self.conv0_1 = ConvBlockGroupNorm(features + features * 2, features, num_groups=num_groups)
        self.conv1_1 = ConvBlockGroupNorm(features * 2 + features * 4, features * 2, num_groups=num_groups)
        self.conv0_2 = ConvBlockGroupNorm(features + features + features * 2, features, num_groups=num_groups)


        # Final Output
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
        

    def forward(self, x, z):
        concat_x_z = torch.cat((x,z), dim=1)
        x0_0 = self.conv0_0(concat_x_z)
        #x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))

        # No padding here - assume input is properly sized
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        return self.final_conv(x0_2)