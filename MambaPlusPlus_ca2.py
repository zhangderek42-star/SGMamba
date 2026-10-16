import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from mambair1_seq3 import SS2D_seq
from module_cab import CAB

from ca_module2 import ca_module


ss2d_config_dict = { 
    "d_model": 124,
    "d_state": 10,
    "d_conv": 3,
    "expand": 1.2,
    "dt_rank": "auto",
    "dt_min": 0.001,
    "dt_max": 0.1,
    "dt_init": "random",
    "dt_scale": 1.0,
    "dt_init_floor": 1e-4,
    "dropout": 0.0,
    "conv_bias": True,
    "bias": False,
    "device": None,
    "dtype": None,
    "input_resolution": (64,64)
}



def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]


class MS_MSA_ori(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out




class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            stage_id=0,
    ):
        super().__init__()
        self.stage_id = stage_id
        self.ss2d_model_dict = ss2d_config_dict.copy()
        self.ss2d_model_dict["d_model"] = dim
        self.ss2d_model_dict["stage_id"] = stage_id + "_ss"

        msa_stage_id = stage_id + "_ss"
        self.ss2d = SS2D_seq(**self.ss2d_model_dict)
        
        self.ms_msa = MS_MSA_ori(dim=dim, dim_head=dim_head, heads=heads)

        self.ca_msa = ca_module(stage_id=stage_id, seq_length=32)

        # print(f'MSA: dim={dim}, dim_head={dim_head}, heads={heads}')

    def forward(self, x_in, index_dict=None):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        
        mamba_out = self.ss2d(x_in, index_dict)
        msa_out = self.ms_msa(x_in)
        ca_out = self.ca_msa(x_in, index_dict)

        return mamba_out + msa_out + ca_out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.config={
            "dim": dim,
            "mult": mult,
        }
        self.cab_config = {
            "num_feat": dim,
            "is_light_sr": True,
            "compress_ratio": 4,
        }
        self.cab_config["squeeze_factor"] = dim // self.cab_config["compress_ratio"]
        self.net = CAB(**self.cab_config)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        result = out.permute(0, 2, 3, 1)
        return result

class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
            stage_id=0,
    ):
        super().__init__()
        self.stage_id = stage_id
        self.blocks = nn.ModuleList([])
        for nb in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads, stage_id=str(stage_id)+"_block_"+str(nb)),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, index_dict=None):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, index_dict) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class MST(nn.Module):
    def __init__(self, in_dim=31, out_dim=31, dim=31, stage=2, num_blocks=[2,4,4], stage_id=0):
        super(MST, self).__init__()
        self.dim = dim
        self.stage = stage
        self.stage_id = stage_id

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim, stage_id=str(stage_id)+"_enc_"+str(i)),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = MSAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1], stage_id=str(stage_id)+"_bottleneck")

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim, stage_id=str(stage_id)+"_dec_"+str(i)),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_data):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        x, index_dict = input_data

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for (MSAB, FeaDownSample) in self.encoder_layers:
            fea = MSAB(fea, index_dict)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea, index_dict)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = LeWinBlcok(fea, index_dict)

        # Mapping
        out = self.mapping(fea) + x

        return out, index_dict

class Mamba_Plus_Plus_ca(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, n_feat=31, stage=1):
        super(Mamba_Plus_Plus_ca, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        modules_body = [MST(dim=31, stage=2, num_blocks=[1,1,1], stage_id=stage_id) for stage_id in range(stage)]
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)

        self.create_id()


    def forward(self, R, x, index_dict):
        return self.ORIG_forward(x, index_dict)

    def ORIG_forward(self, x, index_dict):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        # index_dict = self.cal_index(x)

        x = self.conv_in(x)
        h, _ = self.body((x, index_dict))
        h = self.conv_out(h)
        h += x
        return h[:, :, :h_inp, :w_inp]
    
    def create_id(self):
        if hasattr(self, 'id'):
            return
        self.id = "rMamba++dc"
    
    def cal_index(self, x):
        return {}
    


if __name__ == "__main__":
    
    device = 0
    torch.cuda.set_device(device)
    model = Mamba_Plus_Plus_ca().cuda()

    x = torch.randn(1, 3, 64, 64).cuda()

    index_dict = {
        0: torch.arange(64*64).reshape(1, 64, 64).cuda(),
        1: torch.arange(32*32).reshape(1, 32, 32).cuda(),
        2: torch.arange(16*16).reshape(1, 16, 16).cuda(),
    }

    y = model.forward(None, x, index_dict)

    print(y.shape)  # Expected output: torch.Size([1, 31, 64, 64])












