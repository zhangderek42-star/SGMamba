import torch
from torch_kmeans import KMeans
from tqdm import trange, tqdm
import torch.nn as nn

"""
这个实现还需要有的特性:
● [b, c, h, w] -> ([b, c, h*w], [h*w]), 正向操作, 将图片转换为序列
● ([b, c, h*w], [h*w]) -> [b, c, h, w], 逆向操作, 将序列复原
其中每一个b上的数据不能串
"""

class KMeansIndexer(nn.Module):
    def __init__(self, n_clusters=5):
        super(KMeansIndexer, self).__init__()
        self.n_clusters = n_clusters
        self.model = KMeans(init_method="k-means++",n_clusters=self.n_clusters, mode='euclidean', verbose=0, num_init=1)
        self.cache = None
        


    def unfold(self, image_tensor, index_type="default"):
        """
        输入: [b, c, h, w]
        返回: ([b, c, h*w], [b, h*w])
        type: "default" or "distance"
        """
        image_tensor = torch.as_tensor(image_tensor, dtype=torch.float32)
        b, c, h, w = image_tensor.shape
        pixels = image_tensor.view(b, c, -1)

        input_pixels = pixels.permute(0, 2, 1).contiguous()
        kmeans_result = self.model(input_pixels)

        """
        result.labels: [b, h*w]
        result.centers: [b, n_clusters, c]
        """
        labels = kmeans_result.labels  # [b, h*w]
        centers = kmeans_result.centers  # [b, n_clusters, c]
        inertia = kmeans_result.inertia[1]  # [b, h*w]
        if index_type == "distance":

            index_inner_center = torch.argsort(inertia, dim=1)
            labels_inner_center = torch.gather(labels, 1, index_inner_center)

            index_outer_center = torch.argsort(labels_inner_center, dim=1, stable=True)

            index_gather = torch.gather(index_inner_center, 1, index_outer_center)
        
        elif index_type == "default":

            index_gather = torch.argsort(labels, dim=1)
        
        else:
            raise NotImplementedError(f"index_type {index_type} not implemented")

        result_pixels = torch.gather(input_pixels, 1, index_gather.unsqueeze(-1).expand(-1, -1, c))

        return result_pixels.permute(0, 2, 1).contiguous(), index_gather
    

    
    def unfold_custom_index(self, image_tensor, index_gather):
        """
        输入: [b, c, h, w], [b, h*w]
        返回: ([b, c, h*w])

        """
        image_tensor = torch.as_tensor(image_tensor, dtype=torch.float32)
        
        b, c, h, w = image_tensor.shape
        pixels = image_tensor.view(b, c, -1)

        input_pixels = pixels.permute(0, 2, 1).contiguous()

        result_pixels = torch.gather(input_pixels, 1, index_gather.unsqueeze(-1).expand(-1, -1, c))

        return result_pixels.permute(0, 2, 1).contiguous(), index_gather
    
    
    def fold(self, scanned_tensor, index_tensor):
        """
        输入: ([b, c, h*w], [b, h*w])
        返回: [b, c, h, w]
        """
        b, c, l = scanned_tensor.shape

        index_reverse = torch.argsort(index_tensor, dim=1).unsqueeze(-1)
        restored_pixels = torch.gather(scanned_tensor.permute(0, 2, 1).contiguous(), 1, index_reverse.expand(-1, -1, c))
        h_w = l
        h = int(h_w**0.5)
        w = h_w // h
        return restored_pixels.permute(0, 2, 1).contiguous().view(b, c, h, w)



class SequenceIndexer(nn.Module):
    def __init__(self):
        super(SequenceIndexer, self).__init__()
    
    def unfold(self, target_sequence, index_sequence):
        """
        输入: target_sequence: [b, l], index_sequence: [b, l]
        """
        target_sequence = torch.as_tensor(target_sequence, dtype=torch.float32)
        index_sequence = torch.as_tensor(index_sequence, dtype=torch.long)
        b, l = target_sequence.shape
        result_sequence = torch.gather(target_sequence, 1, index_sequence)
        return result_sequence
    def fold(self, scanned_sequence, index_sequence):
        """
        输入: scanned_sequence: [b, l], index_sequence: [b, l]
        """
        scanned_sequence = torch.as_tensor(scanned_sequence, dtype=torch.float32)
        index_sequence = torch.as_tensor(index_sequence, dtype=torch.long)
        b, l = scanned_sequence.shape
        index_reverse = torch.argsort(index_sequence, dim=1)
        result_sequence = torch.gather(scanned_sequence, 1, index_reverse)
        return result_sequence
    
    def transform_rescale(self, index_sequence, input_resolution=None, output_resolution=None):
        """
        输入: target_sequence: [b, l], index_sequence: [b, l], resolution: (n)
        输出: result_sequence: [b, output_resolution*output_resolution]
        """

        index_sequence = torch.as_tensor(index_sequence, dtype=torch.long)

        assert input_resolution is not None and output_resolution is not None, "Please provide input_resolution and output_resolution"
        b, l = index_sequence.shape
        assert l == input_resolution*input_resolution, "Input sequence length does not match input resolution"

        input_sequence = torch.arange(0, input_resolution*input_resolution).reshape(1, input_resolution, input_resolution).repeat(b,1,1).to(index_sequence.device)

        folded_sequence = self.fold(input_sequence.reshape(b, -1), index_sequence).reshape(b, input_resolution, input_resolution)

        natural_image_resized = nn.functional.interpolate(folded_sequence.unsqueeze(1).float(), size=(output_resolution, output_resolution), mode='nearest-exact',)

        natural_image_flatten = natural_image_resized.reshape(b, -1).long()

        output_sequence = torch.arange(0, output_resolution*output_resolution).reshape(1, output_resolution*output_resolution).repeat(b,1).to(index_sequence.device)
        result_sequence_index = self.fold(output_sequence, natural_image_flatten)

        result_sequence_index = result_sequence_index.long()

        return result_sequence_index
    

    def transform_slice(self, index_sequence, x_start=None, y_start=None, x_end=None, y_end=None, h=None, w=None):
        """
        输入: index_sequence: [b, l], x_start, y_start, x_end, y_end
        输出: result_sequence: [b, x_width*y_width]
        """
        x_width = x_end - x_start
        y_width = y_end - y_start
        index_sequence = torch.as_tensor(index_sequence, dtype=torch.long)

        assert x_start < x_end and y_start < y_end, "Please provide valid slice coordinates"
        b, l = index_sequence.shape
        if h is not None and w is not None:
            assert l == h*w, "Input sequence length does not match provided resolution"
            input_sequence = torch.arange(0, h*w).reshape(1, h, w).repeat(b,1,1).to(index_sequence.device)
            folded_sequence = self.fold(input_sequence.reshape(b, -1), index_sequence).reshape(b, h, w)
        else:
            input_resolution = int(l**0.5)

            input_sequence = torch.arange(0, input_resolution*input_resolution).reshape(1, input_resolution, input_resolution).repeat(b,1,1).to(index_sequence.device)

            folded_sequence = self.fold(input_sequence.reshape(b, -1), index_sequence).reshape(b, input_resolution, input_resolution)

        # natural_image_resized = nn.functional.interpolate(folded_sequence.unsqueeze(1).float(), size=(output_resolution, output_resolution), mode='nearest-exact',)
        natural_image_sliced = folded_sequence[:, x_start:x_end, y_start:y_end]

        natural_image_flatten = natural_image_sliced.reshape(b, -1).long()

        output_sequence = torch.arange(0, x_width*y_width).reshape(1, x_width*y_width).repeat(b,1).to(index_sequence.device)
        result_sequence_index = self.fold(output_sequence, natural_image_flatten)

        result_sequence_index = result_sequence_index.long()

        return result_sequence_index
    




if __name__ == "__main__":
    # 测试代码
    b, c, h, w = 2, 3, 64, 64
    image_tensor = torch.rand(b, c, h, w)

    kmeans_indexer = KMeansIndexer(n_clusters=5)
    scanned_tensor, index_tensor = kmeans_indexer.unfold(image_tensor)
    restored_tensor = kmeans_indexer.fold(scanned_tensor, index_tensor)

    print("原始图像形状:", image_tensor.shape)
    print("扫描后形状:", scanned_tensor.shape, index_tensor.shape)
    print("还原后形状:", restored_tensor.shape)

    print("验证还原是否正确:", torch.allclose(image_tensor, restored_tensor))



