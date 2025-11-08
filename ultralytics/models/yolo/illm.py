###使用grayedge提取光照分量

import cv2
import numpy as np

class IlluminationMapper:
    def __init__(self, p=6, sigma=2):
        """
        初始化类，设置闵可夫斯基 p 范式的参数 p 和高斯核的标准差 sigma。
        :param p: 闵可夫斯基 p 范式的参数，默认为 6
        :param sigma: 高斯核的标准差，默认为 2
        """
        self.p = p
        self.sigma = sigma

    def generate_illumination_map(self, im):
        """
        生成图像的照度映射图，并将其转换为灰度图。
        :param im: 输入的图像，类型为 numpy 数组，形状为 (height, width, 3)
        :return: 灰度照度映射图，类型为 numpy 数组，形状为 (height, width)
        """
        # 计算增益
        k_size = int(self.sigma * 3 + 0.5)
        k = cv2.getGaussianKernel(k_size, self.sigma)
        im_G = cv2.filter2D(im, -1, k, borderType=cv2.BORDER_REPLICATE)  # 高斯滤波
        im_edge = np.gradient(im_G)  # 计算梯度

        # 闵可夫斯基 p 范式
        im_edge = np.abs(im_edge) ** self.p
        r, g, b = im_edge[:, :, 0], im_edge[:, :, 1], im_edge[:, :, 2]
        Avg = np.mean(im_edge) ** (1 / self.p)
        R_avg = np.mean(r) ** (1 / self.p)
        G_avg = np.mean(g) ** (1 / self.p)
        B_avg = np.mean(b) ** (1 / self.p)
        # 计算增益 K
        k = np.array([R_avg, G_avg, B_avg]) / Avg

        # 生成逐像素的光照映射图，避免除零错误
        illumination_map = np.zeros_like(im)
        for i in range(3):
            # 避免除零错误，将接近零的值替换为一个极小值
            denominator = np.ones_like(im[:, :, i])
            denominator[denominator < 1e-8] = 1e-8
            illumination_map[:, :, i] = im[:, :, i] / denominator * k[i]

        # 处理无效值（NaN 和 inf）
        illumination_map = np.nan_to_num(illumination_map, nan=0.0, posinf=1.0, neginf=0.0)

        # 将光照映射图转换为灰度图
        #gray_illumination_map = cv2.cvtColor(illumination_map.astype(np.float32), cv2.COLOR_RGB2GRAY)

        return illumination_map