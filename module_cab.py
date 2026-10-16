import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        self.channel = num_feat

        if self.channel == 31:
            self.channel = 32
        elif self.channel == 62:
            self.channel = 64
        elif self.channel == 124:
            pass
        else:
            raise ValueError("Unexpected channel size: {}".format(self.channel))

        num_feat = self.channel

        if is_light_sr: # we use dilated-conv & DWConv for lightSR for a large ERF
            compress_ratio = 2 
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 1, 1, 0),
                nn.Conv2d(num_feat//compress_ratio, num_feat // compress_ratio, 3, 1, 1,groups=num_feat//compress_ratio),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 1, 1, 0),
                nn.Conv2d(num_feat, num_feat, 3,1,padding=2,groups=num_feat,dilation=2),
                ChannelAttention(num_feat, squeeze_factor)
            )
        else:
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
                ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        channels = x.size(1)
        if channels == 31:
            last_x = x[:,-1:, : , :]
            x = torch.cat([x, last_x], dim=1)
        elif channels == 62:
            last_x = x[:, -2:, :, :]
            x = torch.cat([x, last_x], dim=1)
        elif channels == 124:
            pass
        else:
            raise ValueError("Unexpected Input channel size: {}".format(channels))
        result = self.cab(x)
        if channels == 31:
            result = result[:, :31, :, :]
        elif channels == 62:
            result = result[:, :62, :, :]
        elif channels == 124:
            pass
        else:
            raise ValueError("Unexpected Output channel size: {}".format(channels))
        return result



if __name__ == "__main__":
    device = 8
    torch.cuda.set_device(device)
    input_shapes = [
        (1, 31, 64, 64),
        (1, 62, 32, 32),
        (1, 124, 16, 16),
    ]
    for input_shape in input_shapes:
        for light_sr in [True, False]:

            dim = input_shape[1]
            if dim == 31:
                dim = 32
            elif dim == 62:
                dim = 64
            elif dim == 124:
                pass
            else:
                raise ValueError("Unexpected dim: {}".format(dim))
            config_dict = {
                "num_feat": dim,
                "is_light_sr": light_sr,
                "compress_ratio": 4,
            }
            config_dict["squeeze_factor"] = input_shape[1] // config_dict["compress_ratio"]

            model = CAB(**config_dict).to(device)
            x = torch.randn(*input_shape).to(device)
            y = model(x)
            # print(y.shape)  # Expected output: torch.Size([1, 31, 64, 64])
            assert y.shape == x.shape, "For input shape {}, light_sr={}, expected output shape {}, but got {}".format(input_shape, light_sr, x.shape, y.shape)
    