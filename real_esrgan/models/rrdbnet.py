# Copyright Lornatang. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
from torch import Tensor, nn
from torch.nn import functional as F_torch

from real_esrgan.layers.blocks import ResidualResidualDenseBlock

__all__ = [
    "RRDBNet",
]


class RRDBNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            growth_channels: int = 32,
            num_rrdb: int = 23,
    ) -> None:
        super(RRDBNet, self).__init__()
        # The first layer of convolutional layer
        self.conv_1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network
        trunk = []
        for _ in range(num_rrdb):
            trunk.append(ResidualResidualDenseBlock(channels, growth_channels))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks
        self.conv_2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        # Up-sampling convolutional layer
        self.up_sampling_1 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
        )

        self.up_sampling_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
        )

        # Output layer
        self.conv_3 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
        )

        # Output layer
        self.conv_4 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        out1 = self.conv_1(x)
        out = self.trunk(out1)
        out2 = self.conv_2(out)
        out = torch.add(out1, out2)

        out = self.up_sampling_1(F_torch.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.up_sampling_2(F_torch.interpolate(out, scale_factor=2, mode="nearest"))

        out = self.conv_3(out)
        out = self.conv_4(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out
