import torch
import torch.nn as nn
import math


# Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# Squeeze and Excitation block
class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SEBlock, self).__init__()
        se_channels = int(in_channels * se_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, se_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(se_channels, in_channels, kernel_size=1)
        self.swish = Swish()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.swish(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


# MBConv block (Mobile Inverted Bottleneck Conv)
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25,
                 drop_connect_rate=0.2):
        super(MBConv, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = (in_channels == out_channels and stride == 1)

        expand_channels = in_channels * expand_ratio

        # Expansion phase (pointwise convolution)
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expand_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_channels),
            Swish()
        ) if expand_ratio != 1 else nn.Identity()

        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(expand_channels, expand_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=expand_channels, bias=False),
            nn.BatchNorm2d(expand_channels),
            Swish()
        )

        # Squeeze and Excitation
        self.se = SEBlock(expand_channels, se_ratio)

        # Pointwise convolution (projection phase)
        self.project = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def _drop_connect(self, x):
        if not self.training or self.drop_connect_rate == 0:
            return x
        keep_prob = 1 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob + torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x * binary_tensor / keep_prob

    def forward(self, x):
        identity = x

        # Expansion
        x = self.expand(x)

        # Depthwise convolution
        x = self.depthwise(x)

        # Squeeze and Excitation
        x = self.se(x)

        # Projection
        x = self.project(x)

        # Skip connection with drop connect
        if self.use_residual:
            x = self._drop_connect(x)
            x = x + identity

        return x


# EfficientNet model
class EfficientNet(nn.Module):
    def __init__(self, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2, num_classes=1000):
        super(EfficientNet, self).__init__()

        # Base parameters
        base_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        base_depths = [1, 2, 2, 3, 3, 4, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        strides = [1, 2, 2, 2, 1, 2, 1]
        expand_ratios = [1, 6, 6, 6, 6, 6, 6]

        # Scale channels and depths
        channels = [int(math.ceil(ch * width_coefficient)) for ch in base_channels]
        depths = [int(math.ceil(depth * depth_coefficient)) for depth in base_depths]

        # Initial convolutional layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            Swish()
        )

        # MBConv blocks
        self.blocks = nn.Sequential()
        in_channels = channels[0]

        block_idx = 0
        total_blocks = sum(depths)

        for stage_idx in range(len(depths)):
            stage_out_channels = channels[stage_idx + 1]
            repeats = depths[stage_idx]
            kernel_size = kernel_sizes[stage_idx]
            stride = strides[stage_idx]
            expand_ratio = expand_ratios[stage_idx]

            for i in range(repeats):
                # Only use stride on the first block of each stage
                block_stride = stride if i == 0 else 1

                # Adjust drop connect rate based on block position
                drop_rate = self.drop_connect_rate * block_idx / total_blocks

                self.blocks.add_module(f'block_{block_idx}', MBConv(
                    in_channels=in_channels,
                    out_channels=stage_out_channels,
                    kernel_size=kernel_size,
                    stride=block_stride,
                    expand_ratio=expand_ratio,
                    drop_connect_rate=drop_rate
                ))

                in_channels = stage_out_channels
                block_idx += 1

        # Final stage
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, channels[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(channels[-1]),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes)
        )

        # Initialize weights
        self._initialize_weights()

        # Set drop connect rate
        self.drop_connect_rate = dropout_rate

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


def EfficientNetB0(num_classes=1000):
    return EfficientNet(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        dropout_rate=0.2,
        num_classes=num_classes
    )


def EfficientNetB1(num_classes=1000):
    return EfficientNet(
        width_coefficient=1.0,
        depth_coefficient=1.1,
        dropout_rate=0.2,
        num_classes=num_classes
    )


def EfficientNetB2(num_classes=1000):
    return EfficientNet(
        width_coefficient=1.1,
        depth_coefficient=1.2,
        dropout_rate=0.3,
        num_classes=num_classes
    )