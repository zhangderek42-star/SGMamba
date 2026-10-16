import torch
from kmeans_index_class import KMeansIndex

class ca_transform(torch.nn.Module):
    def __init__(self, stage_id=None, seq_length=64):
        super().__init__()

        self.seq_length = seq_length
        self.stage_id = stage_id
        self.kmeans_index_model = KMeansIndex(n_clusters=5)

        self.mapping = {
            64: [8, 8],
            32: [4, 8],
            16: [4, 4],
            128: [16, 8],
        }

        self.h = self.mapping[seq_length][0]
        self.w = self.mapping[seq_length][1]


    def fold(self, x, image_index=None):
        B, H, W, C = x.shape
        x1 = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x1_seq = self.kmeans_index_model.unfold(x1, image_index=image_index)  # (B, C, H*W)
        x1_seq2 = x1_seq.permute(0, 2, 1).contiguous()  # (B, H*W, C)

        x2 = x1_seq2.reshape(B * H * W // self.seq_length, self.seq_length, C)
        x3 = x2.permute(0, 2, 1).contiguous()
        return x3
    
    def unfold(self, x, original_B, original_H, original_W, image_index=None):
        B_prime, C, L = x.shape
        
        # Step 1: permute(0, 2, 1) to reverse the last permute in fold
        x1 = x.permute(0, 2, 1).contiguous()  # ((B*H*W)//seq_length, seq_length, C)
        
        # Step 2: reshape to (B, H*W, C)
        HW = original_H * original_W
        x2 = x1.reshape(original_B, HW, C)  # (B, H*W, C)
        
        x3 = x2.permute(0, 2, 1).contiguous()  # (B, C, H*W)
        x3_seq = self.kmeans_index_model.fold(x3, image_index=image_index) # B, C, H, W
        x3_seq2 = x3_seq.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        
        return x3_seq2
        
    

if __name__ == "__main__":

    """
    这里fold的逻辑是:
    (b,h,w,c) -kmeans unfold-> (b, h*w, c) -> ((b*h*w)/d, d, c) -> ((b*h*w)/d, c, d)
    """

    seq_length = 64
    transform = ca_transform(stage_id="enc0", seq_length=seq_length)


    x = torch.randn(4, 64, 64, 31)

    image_index = torch.arange(64*64).reshape(1, 64, 64).repeat(4, 1, 1) # 64x64 resolution index


    original_B, original_H, original_W, C = x.shape
    print("x shape:", x.shape) # (4, 64, 64, 31)

    folded_x = transform.fold(x, image_index)
    print("folded_x shape:", folded_x.shape) # (256, 31, 64)

    unfolded_x = transform.unfold(folded_x, original_B=original_B, original_H=original_H, original_W=original_W, image_index=image_index)
    print("unfolded_x shape:", unfolded_x.shape) # (4, 64, 64, 31)

    print(torch.allclose(x, unfolded_x))


    u = x[:2]
    folded_u = transform.fold(u, image_index=image_index[:2])
    print("folded_u shape:", folded_u.shape) # (128, 31, 64)

    reduced_batch_size = int(folded_u.shape[0])

    print(torch.allclose(folded_u, folded_x[:reduced_batch_size]))
