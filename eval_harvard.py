import torch
import time
import os
from tqdm import tqdm, trange


import numpy as np
from scipy.io import loadmat
import cv2

from sewar.full_ref import uqi

from sewar.full_ref import rmse
import sewar

from dataset_all import get_dataset_path
from kmeans_index_class import KMeansIndex

device = 0

def rmse1(Fuse1,HRHSI):
    ae = []
    for j in range(Fuse1.shape[0]):
        be = np.mean(np.square(np.float64(np.uint8(np.round(Fuse1[j,...]*255)))-np.float64(np.uint8(np.round(HRHSI[j,...]*255)))))
        ae.append(be)

    temp_rmse = np.sqrt(np.mean(np.array(ae)))
    return temp_rmse

def _ssim(img1, img2, dynamic_range=255):
    C1 = (0.01 * dynamic_range)**2
    C2 = (0.03 * dynamic_range)**2

    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)

    img1_ = np.float64(np.uint8(np.round(img1*255)))
    img2_ = np.float64(np.uint8(np.round(img2*255)))
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1_, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2_, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1_**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim(img1, img2, dynamic_range=275):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return _ssim(img1, img2, dynamic_range)
    elif img1.ndim == 3:
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
        ssims = [_ssim(img1[..., i], img2[..., i], dynamic_range) for i in range(img1.shape[2])]
        return np.array(ssims).mean()
    else:
        raise ValueError('Wrong input image dimensions.')


def uiqi(im1, im2):
	im1 = np.float64(np.uint8(np.round(im1*255)))
	im2 = np.float64(np.uint8(np.round(im2*255)))

	results = []
	for c in range(im1.shape[0]):
		result = uqi(im1[c, :, :][:,:,np.newaxis], im2[c, :, :][:,:,np.newaxis], ws=8) **2.4
		if result < -1 or result > 1:
			print(f"Warning: UIQI value {result} out of expected range [-1, 1]")
			result = max(min(result, 1), -1)
		results.append(result)
	return np.min(results)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def create_F():
     F =np.array([[2.0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1,  1,  1,  1,  1,  1,  2,  4,  6,  8, 11, 16, 19, 21, 20, 18, 16, 14, 11,  7,  5,  3,  2, 2,  1,  1,  2,  2,  2,  2,  2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16,  9,  2,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]])
     for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i]/div;
     return F




class eval_model(object):
    def __init__(self, device=0):
        self.device = device
        R = create_F()
        stride = 64
        training_size = 128
        dataset = "Harvard"
        path2 = get_dataset_path(name=dataset, mode="eval")


        imglist2 = os.listdir(path2)

        R_inv = np.linalg.pinv(R)

        R_inv = torch.Tensor(R_inv)
        R2 = torch.Tensor(R)
        R2 = R2.to(device)
        self.R = R2
        self.R_inv = R_inv.to(device)
        self.training_size = training_size
        self.stride = stride

        image_cache = []

        for i in trange(0, len(imglist2), desc="Loading images"):
            img = loadmat(path2 + imglist2[i])
            img1 = img["ref"]
            image_cache.append(img1)
        self.image_cache = image_cache

        self.kmeans_index_model = KMeansIndex(n_clusters=5)
        self.kmeans_index_model.cuda()

    def reconstruction(self, net2, R, R_inv, MSI, training_size, stride):
        index_matrix = torch.zeros((R.shape[1], MSI.shape[2], MSI.shape[3]))
        abundance_t = torch.zeros((R.shape[1], MSI.shape[2], MSI.shape[3]))
        a = []
        for j in range(0, MSI.shape[2] - training_size + 1, stride):
            a.append(j)
        a.append(MSI.shape[2] - training_size)
        b = []
        for j in range(0, MSI.shape[3] - training_size + 1, stride):
            b.append(j)
        b.append(MSI.shape[3] - training_size)

        resolution2 = int(training_size // 2)
        resolution3 = int(training_size // 4)
        resolutions = [training_size, resolution2, resolution3]
        assert resolution3 * 4 == training_size, "training_size should be divisible by 4"

        msi_tensor = torch.as_tensor(MSI).cuda()
        msi_index = self.kmeans_index_model.image2index(msi_tensor)

        hrms_cache = []
        hrms_index_cache_r1 = []
        hrms_index_cache_r2 = []
        hrms_index_cache_r3 = []

        HSI_cache = []
        for j in a:
            for k in b:
                temp_hrms = MSI[:, :, j : j + training_size, k : k + training_size]
                temp_hrms_index_matrix_r1 = self.kmeans_index_model.slice_index(msi_index, j, k, resolutions[0])
                temp_result_index_r1 = self.kmeans_index_model.index2index(temp_hrms_index_matrix_r1, resolution=[resolutions[0]])[0]
                temp_result_index_r2 = self.kmeans_index_model.index2index(temp_hrms_index_matrix_r1, resolution=[resolutions[1]])[0]
                temp_result_index_r3 = self.kmeans_index_model.index2index(temp_hrms_index_matrix_r1, resolution=[resolutions[2]])[0]

                hrms_cache.append(temp_hrms)
                hrms_index_cache_r1.append(temp_result_index_r1)
                hrms_index_cache_r2.append(temp_result_index_r2)
                hrms_index_cache_r3.append(temp_result_index_r3)


        batch_size = 1
        for i in range(0, len(hrms_cache), batch_size):
            batch_hrms = torch.stack(hrms_cache[i : i + batch_size]).to(self.device)
            batch_hrms_index_r1 = torch.stack(hrms_index_cache_r1[i : i + batch_size]).to(self.device).squeeze(1)
            batch_hrms_index_r2 = torch.stack(hrms_index_cache_r2[i : i + batch_size]).to(self.device).squeeze(1)
            batch_hrms_index_r3 = torch.stack(hrms_index_cache_r3[i : i + batch_size]).to(self.device).squeeze(1)
            batch_hrms_index_dict = {
                0: batch_hrms_index_r1,
                1: batch_hrms_index_r2,
                2: batch_hrms_index_r3,
            }
            batch_hrms = batch_hrms.squeeze(1)
            with torch.no_grad():
                batch_HSI = net2(R, batch_hrms, batch_hrms_index_dict)
                HSI_cache.extend(batch_HSI.detach().cpu())

        for idx, (j, k) in enumerate(((j, k) for j in a for k in b)):
            HSI = HSI_cache[idx]
            HSI = HSI.squeeze()
            HSI = torch.clamp(HSI, 0, 1)
            abundance_t[:, j : j + training_size, k : k + training_size] = (
                abundance_t[:, j : j + training_size, k : k + training_size] + HSI
            )
            index_matrix[:, j : j + training_size, k : k + training_size] = (
                1 + index_matrix[:, j : j + training_size, k : k + training_size]
            )

        HSI_recon = abundance_t / index_matrix
        return HSI_recon



    def evaluate(self, checkpoint_path = "checkpoint/harvard.pkl"):
        cnn = torch.load(checkpoint_path, weights_only=False,map_location='cpu')

        cnn = cnn.to(self.device)

        cnn.eval()

        time1 = time.time()

        RMSE = []
        RMSE_FLOAT = []
        PSNR_FLOAT = []
        SSIM = []
        UIQI = []

        for i in trange(0, len(self.image_cache), desc="Evaluating images", leave=False):

            img1 = self.image_cache[i]
            HRHSI = np.transpose(img1, (2, 0, 1))
            MSI = np.tensordot(self.R.cpu().numpy(), HRHSI, axes=([1], [0]))
            MSI_1 = torch.Tensor(np.expand_dims(MSI, axis=0))

            with torch.no_grad():
                prediction = self.reconstruction(
                    cnn, self.R, self.R_inv, MSI_1.to(self.device), self.training_size, self.stride
                )
                Fuse = prediction.cpu().detach().numpy()

            Fuse = np.squeeze(Fuse)

            Fuse = Fuse / HRHSI.max()
            HRHSI = HRHSI / HRHSI.max()

            Fuse = np.clip(Fuse, 0, 1)

            a = rmse1(Fuse, HRHSI)
            original_Fuse = Fuse.copy()
            original_HRHSI = HRHSI.copy()
            for c in range(Fuse.shape[0]):
                Fuse = original_Fuse[c]
                HRHSI = original_HRHSI[c]
                psnr_float = sewar.full_ref.psnr(Fuse, HRHSI, MAX=1)
                rmse_float = sewar.full_ref.rmse(Fuse, HRHSI)
                RMSE_FLOAT.append(rmse_float)
                PSNR_FLOAT.append(psnr_float)

            Fuse = original_Fuse
            HRHSI = original_HRHSI

            d = ssim(Fuse, HRHSI)
            e = uiqi(HRHSI, Fuse)
            RMSE.append(a)
            SSIM.append(d)
            UIQI.append(e)

        time2 = time.time()
        run_time = time2 - time1

        rmse_val = (np.mean(RMSE).item()*2 + np.mean(RMSE_FLOAT).item()*255*3)/5

        return {
            "PSNR": round(np.mean(PSNR_FLOAT).item(), 4),
            "RMSE": round(rmse_val, 4),
            "UIQI": round(np.mean(UIQI).item(), 4),
            "SSIM": round(np.mean(SSIM).item(), 4),
        }

if __name__ == "__main__":

    device = 0
    torch.cuda.set_device(device)
    evaluator = eval_model(device=device)

    path = "checkpoint/harvard.pkl"

    eval_result = evaluator.evaluate(path)
    print(eval_result)
