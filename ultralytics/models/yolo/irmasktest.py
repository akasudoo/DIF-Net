import os
import numpy as np
import cv2
from skimage import filters, morphology
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve
import sys  # 导入 sys 模块以处理异常


# ==============================================================================
# --- 以下是核心函数部分 (从之前的代码中整合而来) ---
# ==============================================================================

def read_gray(path):
    """读取灰度图像"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"错误：无法读取图像文件，请检查路径是否正确: {path}")
        sys.exit(1)  # 退出脚本
    return img.astype(np.float32)


def normalize01(x):
    """将数组归一化到 [0, 1] 区间"""
    x = x.astype(np.float64)
    min_val, max_val = x.min(), x.max()
    if max_val - min_val < 1e-12:
        return np.zeros_like(x)
    return (x - min_val) / (max_val - min_val)


def gaussian_smooth(img, ksz=7, sigma=0):
    """高斯平滑"""
    return cv2.GaussianBlur(img, (ksz, ksz), sigma)


def find_all_hot_points(img, percentile=95):
    """根据百分位阈值找到所有高温点"""
    I = normalize01(img)
    thr = np.percentile(I, percentile)
    hot_mask = (I >= thr).astype(np.uint8)
    hot_mask = morphology.opening(hot_mask, morphology.disk(1))
    return hot_mask


def build_weighted_laplacian_from_k(k):
    """根据扩散系数场 k(x) 构建加权拉普拉斯矩阵"""
    H, W = k.shape
    N = H * W
    inds = np.arange(N).reshape(H, W)
    rows, cols, vals = [], [], []
    for dy, dx in [(0, 1), (1, 0)]:
        if dx == 1:
            i1, i2 = inds[:, :-1], inds[:, 1:]
            k1, k2 = k[:, :-1], k[:, 1:]
        else:
            i1, i2 = inds[:-1, :], inds[1:, :]
            k1, k2 = k[:-1, :], k[1:, :]
        kij = 0.5 * (k1 + k2)
        i1f, i2f, kijf = i1.ravel(), i2.ravel(), kij.ravel()
        rows.extend(i1f.tolist());
        cols.extend(i2f.tolist());
        vals.extend((-kijf).tolist())
        rows.extend(i2f.tolist());
        cols.extend(i1f.tolist());
        vals.extend((-kijf).tolist())
    Wmat = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    diag = -Wmat.sum(axis=1).A1
    return Wmat + sparse.diags(diag)


def solve_steady_state(T, hot_mask, K=0.06, lam=1e-4, solver_tol=1e-6, use_cg=True):
    """求解热扩散方程的稳态解"""
    Tn = normalize01(T)
    H, W = Tn.shape
    grad = np.hypot(filters.sobel_h(Tn), filters.sobel_v(Tn))
    gnorm = normalize01(grad)
    k = np.exp(-(gnorm / (K + 1e-12)) ** 2)
    L = build_weighted_laplacian_from_k(k)
    N = H * W
    b = (hot_mask.ravel() * Tn.ravel()).astype(np.float64)
    b[b > 0] += 1e-9
    A = L + lam * sparse.identity(N)
    try:
        if use_cg:
            #u_flat, info = cg(A, b, tol=solver_tol, maxiter=2000)
            u_flat, info = cg(A, b, rtol=solver_tol, maxiter=2000)
            if info != 0:
                print("警告: CG 求解器未收敛，回退到直接求解器。")
                u_flat = spsolve(A.tocsr(), b)
        else:
            u_flat = spsolve(A.tocsr(), b)
    except Exception as e:
        print(f"错误: 求解器失败 ({e})，回退到直接求解器。")
        u_flat = spsolve(A.tocsr(), b)
    return normalize01(u_flat.reshape(H, W))


# ==============================================================================
# --- 主执行逻辑 ---
# ==============================================================================

if __name__ == "__main__":
    # --- 1. 设置文件路径 ---
    # 使用 r"..." 字符串格式来防止 Windows 系统中的反斜杠 `\` 被误解
    input_image_path = r"D:\BaiduNetdiskDownload\LLVIP\infrared\train\080224.jpg"
    output_dir = r"D:\BaiduNetdiskDownload\LLVIP\infrared\irmask"

    # --- 2. 设置算法参数 (可以根据需要调整) ---
    PERCENTILE = 95.0  # 用于确定高温点的亮度百分位阈值
    K_COEFF = 0.06  # Perona-Malik 扩散系数 (值越小，对边缘的阻挡效应越强)
    LAMBDA = 1e-4  # 正则化项系数
    GAUSS_KSZ = 7  # 预处理高斯平滑的核大小

    # --- 3. 执行处理流程 ---
    print(f"正在处理图像: {input_image_path}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 从输入路径生成输出文件名
    base_filename = os.path.basename(input_image_path)
    filename_no_ext, _ = os.path.splitext(base_filename)
    output_mask_filename = f"{filename_no_ext}_mask.png"
    output_mask_path = os.path.join(output_dir, output_mask_filename)

    # 读取和预处理图像
    img = read_gray(input_image_path)
    img_s = gaussian_smooth(img, ksz=GAUSS_KSZ)

    # 寻找所有高温点作为热源
    hot_mask = find_all_hot_points(img_s, percentile=PERCENTILE)
    num_hot_pixels = np.sum(hot_mask)

    if num_hot_pixels == 0:
        print(f"警告: 在百分位 {PERCENTILE}% 下未找到任何热源点。掩码将是全黑的。")
        # 创建一个全黑的图像作为结果
        final_mask = np.zeros_like(img_s)
    else:
        print(f"找到 {num_hot_pixels} 个高温像素点作为热源。")
        # 以所有高温点为热源，求解 PDE 稳态解
        final_mask = solve_steady_state(img_s, hot_mask, K=K_COEFF, lam=LAMBDA)

    # 保存最终的伪掩码
    # 将掩码从 [0, 1] 的浮点数范围转换到 [0, 255] 的 uint8 整数范围
    mask_to_save = (final_mask * 255).astype(np.uint8)
    cv2.imwrite(output_mask_path, mask_to_save)

    print("-" * 30)
    print("处理完成！")
    print(f"生成的伪掩码已保存至: {output_mask_path}")
    print("-" * 30)