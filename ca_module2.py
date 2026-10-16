import torch
from ca_transform3 import ca_transform
from mambair1_ca3 import SS2D_ca
from stage_id2resolution import stage_id2resolution

class ca_module(torch.nn.Module):
    def __init__(self, stage_id=None, seq_length=64):
        super().__init__()

        self.stage_id = stage_id
        self.seq_length = seq_length
        
        self.ca_transform = ca_transform(stage_id=stage_id, seq_length=seq_length)

        ss2d_config_dict = { 
            "d_model": 124,
            "d_state": 10,
            "d_conv": 3,
            "expand": 1,
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
        }
        
        ss2d_config_dict["d_model"] = self.seq_length
        self.ss2d_ca = SS2D_ca(**ss2d_config_dict)
        self.index_resolution = stage_id2resolution(self.stage_id)

    def forward(self, x, index_dict):
        """
        Input: x with shape (B, H, W, C)
        Output: shape (B, H, W, C)
        
        Process:
        1. Fold: (B, H, W, C) -> ((B*H*W)//seq_length, C, seq_length)
        2. Permute: ((B*H*W)//seq_length, C, seq_length) -> ((B*H*W)//seq_length, seq_length, C)
        3. SS2D_ca process
        4. Permute back: ((B*H*W)//seq_length, seq_length, C) -> ((B*H*W)//seq_length, C, seq_length)
        5. Unfold: ((B*H*W)//seq_length, C, seq_length) -> (B, H, W, C)
        """
        B, H, W, C = x.shape

        
        # Fold operation: (B, H, W, C) -> ((B*H*W)//seq_length, C, seq_length)

        image_index = index_dict[self.index_resolution]

        folded_x = self.ca_transform.fold(x, image_index=image_index)
        
        ss2d_output = self.ss2d_ca(folded_x)
        
        output = self.ca_transform.unfold(ss2d_output, original_B=B, original_H=H, original_W=W, image_index=image_index)
        
        return output


def test1():
    stage_id = "0_enc_0_block_0_ss"
    seq_length = 64
    ca_module_instance = ca_module(stage_id=stage_id, seq_length=seq_length)

    ca_module_instance = ca_module_instance.cuda()

    x = torch.randn(4, 64, 64, 31).cuda()

    index_dict = {
        0: torch.arange(64*64).reshape(1, 64, 64).repeat(4, 1, 1).cuda(), # 64x64 resolution index
    }


    print("Input x shape:", x.shape) # (4, 64, 64, 31)

    output = ca_module_instance(x, index_dict)
    print("Output shape:", output.shape) # (4, 64, 64, 31)

    x2 = x[:2]

    index_dict2 = {
        0: index_dict[0][:2].cuda(), # 64x64 resolution index for smaller batch
    }

    output2 = ca_module_instance(x2, index_dict2)
    print("Output shape for smaller batch:", output2.shape) # (2, 64, 64, 31)
    print("Output sim:", torch.allclose(output2, output[:2]))





if __name__ == "__main__":
    test1()
