#!/usr/bin/env python3
# irmasktest2.py
# 测试脚本：GPU (CuPy) 优先求解稳态扩散生成热掩码，并用 k-map 的 Canny 边缘增强 mask。
# 若 GPU 不可用，则自动回退到 CPU。

import os
import numpy as np
import cv2
from skimage import filters, morphology
from scipy import sparse as sp_sparse
from scipy.sparse.linalg import cg as cg_cpu, spsolve as spsolve_cpu
import matplotlib.pyplot as plt

# 尝试导入 CuPy（GPU 加速）
USE_CUPY = False
try:
    import cupy as cp
    import cupyx
    import cupyx.scipy.sparse as cusparse
    import cupyx.scipy.sparse.linalg as cusparse_linalg
    _ = cp.cuda.runtime.getDeviceCount()
    USE_CUPY = True
    print("[INFO] CuPy available - GPU will be attempted.")
except Exception as e:
    print("[INFO] CuPy/GPU not available - using CPU. Error:", e)
    USE_CUPY = False

# ---------------- 工具函数 ----------------
def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img.astype(np.float32)

def normalize01_np(x):
    x = x.astype(np.float64)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def gaussian_smooth(img, ksz=7, sigma=0):
    return cv2.GaussianBlur(img, (ksz, ksz), sigma)

def find_all_hot_points(img, percentile=95):
    I = normalize01_np(img)
    thr = np.percentile(I, percentile)
    hot_mask = (I >= thr).astype(np.uint8)
    hot_mask = morphology.opening(hot_mask, morphology.disk(1))
    return hot_mask

def save_overlay_gray_with_colormap(gray, mask, outpath, alpha=0.6, cmap='jet'):
    bg = normalize01_np(gray)
    bg_rgb = np.stack([bg, bg, bg], axis=-1)
    cmap_m = plt.get_cmap(cmap)
    mask_rgba = cmap_m(mask)
    mask_rgb = mask_rgba[..., :3]
    overlay = (1 - alpha) * bg_rgb + alpha * mask_rgb
    overlay = (overlay * 255).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(outpath, overlay_bgr)

# ---------------- CPU Laplacian 构建 ----------------
def build_weighted_laplacian_from_k_cpu(k):
    H, W = k.shape
    N = H * W
    inds = np.arange(N).reshape(H, W)

    i1 = inds[:, :-1].ravel()
    i2 = inds[:, 1:].ravel()
    k_right = 0.5 * (k[:, :-1] + k[:, 1:]).ravel()

    j1 = inds[:-1, :].ravel()
    j2 = inds[1:, :].ravel()
    k_down = 0.5 * (k[:-1, :] + k[1:, :]).ravel()

    rows = np.concatenate([i1, i2, j1, j2])
    cols = np.concatenate([i2, i1, j2, j1])
    vals = np.concatenate([-k_right, -k_right, -k_down, -k_down])

    Wmat = sp_sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    diag = -Wmat.sum(axis=1).A1
    L = Wmat + sp_sparse.diags(diag)
    return L

# ---------------- GPU Laplacian 构建 ----------------
def build_weighted_laplacian_from_k_gpu(k_cpu):
    k = cp.asarray(k_cpu)
    H, W = k.shape
    N = H * W
    inds = cp.arange(N, dtype=cp.int32).reshape(H, W)

    i1 = inds[:, :-1].ravel()
    i2 = inds[:, 1:].ravel()
    k_right = 0.5 * (k[:, :-1] + k[:, 1:]).ravel()

    j1 = inds[:-1, :].ravel()
    j2 = inds[1:, :].ravel()
    k_down = 0.5 * (k[:-1, :] + k[1:, :]).ravel()

    rows = cp.concatenate([i1, i2, j1, j2]).astype(cp.int32)
    cols = cp.concatenate([i2, i1, j2, j1]).astype(cp.int32)
    vals = cp.concatenate([-k_right, -k_right, -k_down, -k_down]).astype(cp.float64)

    Wmat = cusparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    diag_np = -Wmat.sum(axis=1).get().ravel()
    diag_gpu = cp.asarray(diag_np)
    L = Wmat + cusparse.diags(diag_gpu)
    return L

# ---------------- 稳态扩散求解 ----------------
def solve_steady_state(T, hot_mask, K=0.06, lam=1e-4, solver_tol=1e-6, try_gpu=USE_CUPY):
    Tn = normalize01_np(T)
    H, W = Tn.shape
    gx = filters.sobel_h(Tn)
    gy = filters.sobel_v(Tn)
    grad = np.hypot(gx, gy)
    gnorm = normalize01_np(grad)
    k = np.exp(- (gnorm / (K + 1e-12)) ** 2)

    N = H * W
    b = (hot_mask.ravel() * Tn.ravel()).astype(np.float64)
    b[b > 0] += 1e-12

    if try_gpu:
        try:
            L_gpu = build_weighted_laplacian_from_k_gpu(k)
            I_gpu = cusparse.identity(N, dtype=cp.float64, format='csr')
            A_gpu = L_gpu + lam * I_gpu
            b_gpu = cp.asarray(b)
            u_gpu, info = cusparse_linalg.cg(A_gpu, b_gpu, tol=solver_tol, maxiter=2000)
            if info != 0:
                raise RuntimeError(f"GPU CG not converged (info={info})")
            u_flat = cp.asnumpy(u_gpu)
            u = u_flat.reshape(H, W)
            return normalize01_np(u), k
        except Exception as e:
            print("[WARN] GPU failed, fallback to CPU:", e)

    L_cpu = build_weighted_laplacian_from_k_cpu(k)
    A_cpu = L_cpu + lam * sp_sparse.identity(N)
    try:
        u_flat, info = cg_cpu(A_cpu, b, tol=solver_tol, maxiter=2000)
        if info != 0:
            u_flat = spsolve_cpu(A_cpu.tocsr(), b)
    except Exception:
        u_flat = spsolve_cpu(A_cpu.tocsr(), b)
    u = u_flat.reshape(H, W)
    return normalize01_np(u), k

# ---------------- Canny边缘增强 ----------------
def enhance_mask_with_k_canny(mask, kmap,
                              canny_thresh1=30, canny_thresh2=120,
                              dilate_radius=2,
                              edge_strength=0.8,
                              normalize_result=True):
    """
    使用 Canny 算子提取 k-map 边缘，并增强 mask。
    """
    kmap = normalize01_np(kmap)
    kmap_uint8 = (kmap * 255).astype(np.uint8)

    # 1) Canny 检测
    edges = cv2.Canny(kmap_uint8, canny_thresh1, canny_thresh2)
    edges = (edges > 0).astype(np.uint8)

    # 2) 膨胀
    if dilate_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_radius+1, 2*dilate_radius+1))
        edges = cv2.dilate(edges, kernel)

    # 3) 平滑并归一化
    edges_f = cv2.GaussianBlur(edges.astype(np.float32), (5,5), 1.0)
    if edges_f.max() > 0:
        edges_f /= edges_f.max()

    # 4) mask增强
    m = normalize01_np(mask)
    enhanced = m + edge_strength * edges_f

    if normalize_result:
        enhanced = normalize01_np(enhanced)
    return enhanced

# ---------------- 主流程 ----------------
if __name__ == "__main__":
    # 输入与输出路径（可自行修改）
    input_image_path = r"D:\BaiduNetdiskDownload\LLVIP\infrared\train\080224.jpg"
    output_dir = r"D:\BaiduNetdiskDownload\LLVIP\infrared\irmask_gpu_test2"

    os.makedirs(output_dir, exist_ok=True)

    # 参数设定
    PERCENTILE = 95.0
    K_COEFF = 0.06
    LAMBDA = 1e-4
    GAUSS_KSZ = 7
    ALPHA = 0.6
    EDGE_STRENGTH = 0.8

    print(f"[INFO] Processing {input_image_path}")
    img = read_gray(input_image_path)
    h, w = img.shape
    max_dim = 512
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        print(f"[INFO] Resized to {img.shape}")

    img_s = gaussian_smooth(img, ksz=GAUSS_KSZ)
    hot_mask = find_all_hot_points(img_s, percentile=PERCENTILE)
    print(f"[INFO] Hot pixels: {np.sum(hot_mask)}")

    if np.sum(hot_mask) == 0:
        final_mask = np.zeros_like(img_s)
        kmap = np.zeros_like(img_s)
    else:
        final_mask, kmap = solve_steady_state(img_s, hot_mask, K=K_COEFF, lam=LAMBDA, try_gpu=USE_CUPY)

    base = os.path.basename(input_image_path)
    name, _ = os.path.splitext(base)
    out_mask = os.path.join(output_dir, f"{name}_mask.png")
    out_kmap = os.path.join(output_dir, f"{name}_k_map.png")

    cv2.imwrite(out_mask, (final_mask * 255).astype(np.uint8))
    cv2.imwrite(out_kmap, (normalize01_np(kmap) * 255).astype(np.uint8))

    out_overlay = os.path.join(output_dir, f"{name}_overlay.png")
    save_overlay_gray_with_colormap(img_s, final_mask, out_overlay, alpha=ALPHA)
    print(f"[INFO] Saved mask & overlay.")

    # 进行Canny边缘增强
    mask_enh = enhance_mask_with_k_canny(final_mask, kmap,
                                         canny_thresh1=30, canny_thresh2=120,
                                         dilate_radius=2,
                                         edge_strength=EDGE_STRENGTH)

    out_mask_enh = os.path.join(output_dir, f"{name}_mask_enh.png")
    cv2.imwrite(out_mask_enh, (mask_enh * 255).astype(np.uint8))
    out_overlay_enh = os.path.join(output_dir, f"{name}_overlay_enh.png")
    save_overlay_gray_with_colormap(img_s, mask_enh, out_overlay_enh, alpha=ALPHA)
    print(f"[INFO] Saved enhanced mask & overlay.")

    print("[DONE] All outputs saved to:", output_dir)
