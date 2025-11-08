

import os
import argparse
import numpy as np
import cv2
from skimage import filters, morphology, measure, img_as_float
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve
import matplotlib.pyplot as plt


# ---------- 工具函数 ----------
def read_gray(path):
    """读取灰度图像"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32)


def normalize01(x):
    """将数组归一化到 [0, 1] 区间"""
    x = x.astype(np.float64)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def gaussian_smooth(img, ksz=7, sigma=0):
    """高斯平滑"""
    return cv2.GaussianBlur(img, (ksz, ksz), sigma)


# ---------- 高温点提取 (新功能) ----------
def find_all_hot_points(img, percentile=95):
    """
    根据百分位阈值找到所有高温点。

    参数:
    img: 灰度图像 (uint8 或 float)
    percentile: 用于二值化的阈值百分位 (例如, 95)

    返回:
    hot_mask: 一个二值掩码，其中所有高温点的值为1，其余为0。
    """
    # 归一化以便稳定使用百分位
    I = normalize01(img)
    thr = np.percentile(I, percentile)
    hot_mask = (I >= thr).astype(np.uint8)

    # 可选：使用形态学开运算去除小的噪声点
    hot_mask = morphology.opening(hot_mask, morphology.disk(1))

    return hot_mask


# ---------- 构建加权拉普拉斯矩阵 ----------
def build_weighted_laplacian_from_k(k):
    """根据扩散系数场 k(x) 构建加权拉普拉斯矩阵"""
    H, W = k.shape
    N = H * W
    inds = np.arange(N).reshape(H, W)
    rows = []
    cols = []
    vals = []
    # 连接右侧和下方的邻居，然后对称化
    for dy, dx in [(0, 1), (1, 0)]:
        if dx == 1:
            i1 = inds[:, :-1];
            i2 = inds[:, 1:]
            k1 = k[:, :-1];
            k2 = k[:, 1:]
        else:
            i1 = inds[:-1, :];
            i2 = inds[1:, :]
            k1 = k[:-1, :];
            k2 = k[1:, :]
        kij = 0.5 * (k1 + k2)
        i1f = i1.ravel();
        i2f = i2.ravel();
        kijf = kij.ravel()
        rows.extend(i1f.tolist());
        cols.extend(i2f.tolist());
        vals.extend((-kijf).tolist())
        rows.extend(i2f.tolist());
        cols.extend(i1f.tolist());
        vals.extend((-kijf).tolist())
    Wmat = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    diag = -Wmat.sum(axis=1).A1
    L = Wmat + sparse.diags(diag)
    return L


# ---------- PDE 稳态求解器 (已修改) ----------
def solve_steady_state(T, hot_mask, K=0.06, lam=1e-4, solver_tol=1e-6, use_cg=True):
    """
    求解热扩散方程的稳态解。

    参数:
    T: 原始灰度图像 (float)
    hot_mask: (H,W) 的二值掩码，表示热源位置

    返回:
    u: 归一化的结果掩码 (H,W)
    k: 扩散系数图 k(x)
    """
    Tn = normalize01(T)
    H, W = Tn.shape
    gx = filters.sobel_h(Tn)
    gy = filters.sobel_v(Tn)
    grad = np.hypot(gx, gy)
    gnorm = normalize01(grad)
    k = np.exp(- (gnorm / (K + 1e-12)) ** 2)  # Perona-Malik 风格扩散系数

    L = build_weighted_laplacian_from_k(k)
    N = H * W

    # 热源 b 现在是 hot_mask 中的所有点
    # 热源的强度可以与该点的原始温度成正比，以获得更自然的效果
    b = (hot_mask.ravel() * Tn.ravel()).astype(np.float64)
    b[b > 0] += 1e-9  # 确保热源不为零

    A = L + lam * sparse.identity(N)

    # 尝试使用共轭梯度法 (CG)，失败则回退到直接求解法
    try:
        if use_cg:
            u_flat, info = cg(A, b, tol=solver_tol, maxiter=2000)
            if info != 0:
                print("CG solver did not converge, falling back to direct solver.")
                u_flat = spsolve(A.tocsr(), b)
        else:
            u_flat = spsolve(A.tocsr(), b)
    except Exception as e:
        print(f"Solver failed with error: {e}. Falling back to direct solver.")
        u_flat = spsolve(A.tocsr(), b)

    u = u_flat.reshape(H, W)
    u = normalize01(u)
    return u, k


# ---------- 径向高斯基线 ----------
def radial_gaussian_mask(T, seed=None, sigma_pixels=None, temp_power=1.0):
    """生成一个径向高斯掩码作为对比基线"""
    Tn = normalize01(T)
    H, W = Tn.shape
    if seed is None:
        # 如果未提供种子，则使用图像中最亮的点
        idx = np.unravel_index(np.argmax(Tn), Tn.shape)
        y0, x0 = idx
    else:
        y0, x0 = seed
    if sigma_pixels is None:
        sigma_pixels = max(1.0, min(H, W) / 6.0)
    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]
    d2 = (ys - y0) ** 2 + (xs - x0) ** 2
    mask = np.exp(-d2 / (2.0 * (sigma_pixels ** 2)))
    if temp_power != 0:
        mask *= (Tn ** temp_power)
    mask = normalize01(mask)
    return mask


# ---------- 可视化辅助函数 ----------
def save_overlay_gray_with_colormap(gray, mask, outpath, alpha=0.6, cmap='jet'):
    """将掩码以彩色图谱叠加到灰度图上并保存"""
    bg = normalize01(gray)
    bg_rgb = np.stack([bg, bg, bg], axis=-1)
    cmap_m = plt.get_cmap(cmap)
    mask_rgba = cmap_m(mask)
    mask_rgb = mask_rgba[..., :3]
    overlay = (1 - alpha) * bg_rgb + alpha * mask_rgb
    overlay = (overlay * 255).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(outpath, overlay_bgr)


# ---------- 主流程 (已修改) ----------
def main(args):
    img = read_gray(args.infile)
    if args.max_dim is not None:
        # 可选：调整图像大小以加快处理速度
        h, w = img.shape
        scale = min(args.max_dim / max(h, w), 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 预处理：高斯平滑
    img_s = gaussian_smooth(img, ksz=args.gauss_ksz, sigma=0)

    # 1. 寻找所有高温点，生成一个二值掩码
    hot_mask = find_all_hot_points(img_s, percentile=args.percentile)
    num_hot_pixels = np.sum(hot_mask)
    print(f"已找到 {num_hot_pixels} 个高温像素点作为热源。")

    if num_hot_pixels == 0:
        print("在当前百分位阈值下未找到高温点，请尝试降低 --percentile 的值。")
        return

    # 2. 以所有高温点为热源，求解PDE稳态解
    u, kmap = solve_steady_state(img_s, hot_mask, K=args.K, lam=args.lam, solver_tol=args.solver_tol,
                                 use_cg=not args.force_direct)

    os.makedirs(args.out_dir, exist_ok=True)

    # 保存扩散后的掩码
    mask_path = os.path.join(args.out_dir, "pde_diffusion_mask.png")
    cv2.imwrite(mask_path, (u * 255).astype(np.uint8))
    print(f"已将扩散掩码保存至: {mask_path}")

    # 保存叠加后的图像
    overlay_path = os.path.join(args.out_dir, "pde_diffusion_overlay.png")
    save_overlay_gray_with_colormap(img_s, u, overlay_path, alpha=args.alpha, cmap='jet')
    print(f"已将叠加图像保存至: {overlay_path}")

    # 生成并保存径向高斯基线作为对比
    # 使用图像中最亮的一个点作为高斯中心
    y0, x0 = np.unravel_index(np.argmax(img_s), img_s.shape)
    radial = radial_gaussian_mask(img_s, seed=(y0, x0), sigma_pixels=args.radial_sigma,
                                  temp_power=args.radial_temp_power)
    radial_path = os.path.join(args.out_dir, "radial_baseline_overlay.png")
    save_overlay_gray_with_colormap(img_s, radial, radial_path, alpha=args.alpha, cmap='jet')
    print(f"已将径向基线图像保存至: {radial_path}")

    # 可选：保存扩散系数 k 图的可视化结果
    kvis = normalize01(kmap)
    cv2.imwrite(os.path.join(args.out_dir, "k_map.png"), (kvis * 255).astype(np.uint8))

    print("\n--- 使用建议 ---")
    print("如果掩码扩散范围太小，请尝试增大 --K 的值或降低 --percentile。")
    print("如果掩码扩散范围太大或“泄露”到背景，请尝试减小 --K 的值或提高 --percentile。")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="从红外图像生成热扩散掩码")
    p.add_argument("--infile", type=str, required=True, help="输入的灰度红外图像路径")
    p.add_argument("--out_dir", type=str, default="./out", help="输出结果的文件夹路径")
    p.add_argument("--percentile", type=float, default=95.0, help="用于确定高温点的亮度百分位阈值")
    p.add_argument("--K", type=float, default=0.06, help="Perona-Malik 扩散系数K (值越小，对边缘的阻挡效应越强)")
    p.add_argument("--lam", type=float, default=1e-4, help="正则化项 A = L + lam*I 的系数")
    p.add_argument("--solver_tol", type=float, default=1e-6, help="求解器的容忍度")
    p.add_argument("--gauss_ksz", type=int, default=7, help="预处理高斯平滑的核大小")
    p.add_argument("--alpha", type=float, default=0.6, help="叠加图像的透明度")
    p.add_argument("--max_dim", type=int, default=512, help="处理图像的最大尺寸 (若大于此值则会缩放)")
    p.add_argument("--radial_sigma", type=float, default=None, help="径向高斯基线的 sigma (像素)")
    p.add_argument("--radial_temp_power", type=float, default=1.0, help="径向高斯基线中温度项的幂")
    p.add_argument("--force_direct", action='store_true', help="强制使用直接求解器 (spsolve) 而不是共轭梯度法 (CG)")

    args = p.parse_args()
    main(args)