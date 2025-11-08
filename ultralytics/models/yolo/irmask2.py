

import os
import argparse
import numpy as np
import cv2
from skimage import filters, morphology
from scipy import sparse as sp_sparse
from scipy.sparse.linalg import cg as cg_cpu, spsolve as spsolve_cpu
import matplotlib.pyplot as plt

# Try import CuPy
USE_CUPY = False
try:
    import cupy as cp
    import cupyx
    import cupyx.scipy.sparse as cusparse
    import cupyx.scipy.sparse.linalg as cusparse_linalg
    # check GPU availability
    _ = cp.cuda.runtime.getDeviceCount()
    USE_CUPY = True
    print("[INFO] CuPy found: will attempt GPU acceleration.")
except Exception as e:
    print("[INFO] CuPy/GPU not available or import failed; will use CPU path. Error:", e)
    USE_CUPY = False

# -------------------- Utilities --------------------
def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img.astype(np.float32)

def normalize01_np(x):
    x = x.astype(np.float64)
    mn = x.min()
    mx = x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mn) / (mx - mn)

def gaussian_smooth(img, ksz=7, sigma=0):
    return cv2.GaussianBlur(img, (ksz, ksz), sigma)

def find_all_hot_points(img, percentile=95):
    I = normalize01_np(img)
    thr = np.percentile(I, percentile)
    hot_mask = (I >= thr).astype(np.uint8)
    hot_mask = morphology.opening(hot_mask, morphology.disk(1))
    return hot_mask

def radial_gaussian_mask(T, seed=None, sigma_pixels=None, temp_power=1.0):
    Tn = normalize01_np(T)
    H, W = Tn.shape
    if seed is None:
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
    mask = normalize01_np(mask)
    return mask

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

# -------------------- k-map -> weighted Laplacian (CPU fast) --------------------
def build_weighted_laplacian_from_k_fast_cpu(k):
    H, W = k.shape
    N = H * W
    inds = np.arange(N).reshape(H, W)

    # right neighbors
    i1 = inds[:, :-1].ravel()
    i2 = inds[:, 1:].ravel()
    k_right = 0.5 * (k[:, :-1] + k[:, 1:])
    k_right = k_right.ravel()

    # down neighbors
    j1 = inds[:-1, :].ravel()
    j2 = inds[1:, :].ravel()
    k_down = 0.5 * (k[:-1, :] + k[1:, :])
    k_down = k_down.ravel()

    rows = np.concatenate([i1, i2, j1, j2, i2, i1, j2, j1])
    cols = np.concatenate([i2, i1, j2, j1, i1, i2, j1, j2])
    vals = np.concatenate([-k_right, -k_right, -k_down, -k_down,
                           -k_right, -k_right, -k_down, -k_down])

    Wmat = sp_sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    diag = -Wmat.sum(axis=1).A1
    L = Wmat + sp_sparse.diags(diag)
    return L

# -------------------- k-map -> weighted Laplacian (GPU fast) --------------------
def build_weighted_laplacian_from_k_fast_gpu(k_cpu):
    """
    Input: k_cpu (numpy float64 array)
    Return: L_gpu (cupyx csr_matrix)
    """
    k = cp.asarray(k_cpu)
    H, W = k.shape
    N = H * W
    inds = cp.arange(N, dtype=cp.int32).reshape(H, W)

    # right neighbors
    i1 = inds[:, :-1].ravel()
    i2 = inds[:, 1:].ravel()
    k_right = 0.5 * (k[:, :-1] + k[:, 1:])
    k_right = k_right.ravel()

    # down neighbors
    j1 = inds[:-1, :].ravel()
    j2 = inds[1:, :].ravel()
    k_down = 0.5 * (k[:-1, :] + k[1:, :])
    k_down = k_down.ravel()

    rows = cp.concatenate([i1, i2, j1, j2, i2, i1, j2, j1]).astype(cp.int32)
    cols = cp.concatenate([i2, i1, j2, j1, i1, i2, j1, j2]).astype(cp.int32)
    vals = cp.concatenate([-k_right, -k_right, -k_down, -k_down,
                           -k_right, -k_right, -k_down, -k_down]).astype(cp.float64)

    # build sparse COO -> CSR on GPU
    Wmat = cusparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()

    # compute diag (note: Wmat.sum returns a cupy matrix/array; use .get() to convert to numpy then cp.asarray)
    diag_np = -Wmat.sum(axis=1).get().ravel()
    diag_gpu = cp.asarray(diag_np)
    L = Wmat + cusparse.diags(diag_gpu)
    return L

# -------------------- solve (GPU preferred) --------------------
def solve_steady_state(T, hot_mask, K=0.06, lam=1e-4, solver_tol=1e-6, try_gpu=USE_CUPY, force_cpu=False):
    """
    Returns (u (numpy float array normalized [0,1]), k (numpy float array))
    """
    if force_cpu:
        try_gpu = False

    Tn = normalize01_np(T)
    H, W = Tn.shape
    gx = filters.sobel_h(Tn)
    gy = filters.sobel_v(Tn)
    grad = np.hypot(gx, gy)
    gnorm = normalize01_np(grad)
    k = np.exp(- (gnorm / (K + 1e-12)) ** 2)  # numpy array (float64)

    N = H * W
    b_cpu = (hot_mask.ravel() * Tn.ravel()).astype(np.float64)
    b_cpu[b_cpu > 0] += 1e-12

    # GPU path
    if try_gpu:
        try:
            L_gpu = build_weighted_laplacian_from_k_fast_gpu(k)
            I_gpu = cusparse.identity(N, dtype=cp.float64, format='csr')
            A_gpu = L_gpu + lam * I_gpu
            b_gpu = cp.asarray(b_cpu)

            # Use cupyx's cg if available
            if hasattr(cusparse_linalg, 'cg'):
                u_gpu, info = cusparse_linalg.cg(A_gpu, b_gpu, tol=solver_tol, maxiter=2000)
                if info != 0:
                    raise RuntimeError(f"GPU CG did not converge (info={info})")
                u_flat = cp.asnumpy(u_gpu)
                u = u_flat.reshape(H, W)
                u = normalize01_np(u)
                return u, k
            else:
                # If cg not available on this cupy build, raise to fallback
                raise RuntimeError("cupyx.sparse.linalg.cg not available in this cupy build")
        except Exception as e:
            print("[WARN] GPU path failed or not usable; falling back to CPU. Error:", e)
            # continue to CPU path

    # CPU path
    L_cpu = build_weighted_laplacian_from_k_fast_cpu(k)
    A_cpu = L_cpu + lam * sp_sparse.identity(N)
    try:
        u_flat, info = cg_cpu(A_cpu, b_cpu, tol=solver_tol, maxiter=2000)
        if info != 0:
            print(f"[WARN] CPU CG did not converge (info={info}); using direct solver.")
            u_flat = spsolve_cpu(A_cpu.tocsr(), b_cpu)
    except Exception as e:
        print("[WARN] CPU CG failed; using direct solver. Error:", e)
        u_flat = spsolve_cpu(A_cpu.tocsr(), b_cpu)

    u = u_flat.reshape(H, W)
    u = normalize01_np(u)
    return u, k

# -------------------- enhance mask with k edges --------------------
def enhance_mask_with_k_edges(mask, kmap,
                              method='canny',
                              canny_thresh1=30, canny_thresh2=120,
                              sobel_thresh=0.05,
                              dilate_radius=1,
                              edge_strength=0.6,
                              normalize_result=True):
    """
    Detect edges on kmap and add/enhance them into mask.

    mask: numpy 2D float in [0,1]
    kmap: numpy 2D float
    method: 'canny' or 'sobel'
    returns enhanced mask (float 2D in [0,1] if normalize_result True)
    """
    # normalize kmap to uint8 for canny
    k_norm = kmap.astype(np.float64)
    kmin, kmax = k_norm.min(), k_norm.max()
    if kmax - kmin < 1e-12:
        k_scaled = np.zeros_like(k_norm, dtype=np.uint8)
    else:
        k_scaled = ((k_norm - kmin) / (kmax - kmin) * 255.0).astype(np.uint8)

    if method == 'canny':
        edges = cv2.Canny(k_scaled, int(canny_thresh1), int(canny_thresh2))
        edges = (edges > 0).astype(np.uint8)
    elif method == 'sobel':
        gx = cv2.Sobel(k_norm, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(k_norm, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.hypot(gx, gy)
        gmin, gmax = grad.min(), grad.max()
        if gmax - gmin < 1e-12:
            grad_n = np.zeros_like(grad)
        else:
            grad_n = (grad - gmin) / (gmax - gmin)
        edges = (grad_n >= sobel_thresh).astype(np.uint8)
    else:
        raise ValueError("method must be 'canny' or 'sobel'")

    if dilate_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_radius+1, 2*dilate_radius+1))
        edges = cv2.dilate(edges.astype(np.uint8), kernel)

    edges_f = edges.astype(np.float32)
    if dilate_radius > 0:
        sigma = max(0.5, dilate_radius * 0.6)
        ksz = int(max(3, 2 * int(3*sigma) + 1))
        edges_f = cv2.GaussianBlur(edges_f, (ksz, ksz), sigma)

    if edges_f.max() > 0:
        edges_f = edges_f / edges_f.max()
    else:
        edges_f = edges_f

    m = mask.astype(np.float32)
    if m.max() > 0:
        m = m / (m.max() if m.max() > 0 else 1.0)
    enhanced = m + edge_strength * edges_f

    if normalize_result:
        mn, mx = enhanced.min(), enhanced.max()
        if mx - mn < 1e-12:
            enhanced_n = np.zeros_like(enhanced)
        else:
            enhanced_n = (enhanced - mn) / (mx - mn)
        return enhanced_n
    else:
        return enhanced

# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser(description="IR mask generator with GPU acceleration (CuPy).")
    parser.add_argument("--infile", type=str, required=True, help="Path to input grayscale infrared image")
    parser.add_argument("--out_dir", type=str, default="./out", help="Output directory")
    parser.add_argument("--percentile", type=float, default=95.0, help="percentile threshold for hot points")
    parser.add_argument("--K", type=float, default=0.06, help="edge sensitivity for diffusion coefficient")
    parser.add_argument("--lam", type=float, default=1e-4, help="regularization lambda")
    parser.add_argument("--solver_tol", type=float, default=1e-6, help="solver tolerance for CG")
    parser.add_argument("--gauss_ksz", type=int, default=7, help="gaussian blur kernel size")
    parser.add_argument("--alpha", type=float, default=0.6, help="overlay alpha")
    parser.add_argument("--max_dim", type=int, default=512, help="max image dimension (will downscale larger images)")
    parser.add_argument("--radial_sigma", type=float, default=None, help="radial baseline sigma in pixels")
    parser.add_argument("--radial_temp_power", type=float, default=1.0, help="multiply radial baseline by temp^power")
    parser.add_argument("--force_cpu", action="store_true", help="force CPU mode even if CuPy available")
    parser.add_argument("--edge_method", type=str, default="canny", choices=["canny","sobel"], help="method to detect k-map edges")
    parser.add_argument("--edge_dilate", type=int, default=2, help="dilate radius for edge band")
    parser.add_argument("--edge_strength", type=float, default=0.8, help="strength to add k edges into mask")
    args = parser.parse_args()

    infile = args.infile
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Reading image {infile}")
    img = read_gray(infile)

    # Resize if necessary
    if args.max_dim is not None:
        h, w = img.shape
        scale = min(args.max_dim / max(h, w), 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            print(f"[INFO] Resized image to {img.shape}")

    img_s = gaussian_smooth(img, ksz=args.gauss_ksz, sigma=0)
    hot_mask = find_all_hot_points(img_s, percentile=args.percentile)
    num_hot = np.sum(hot_mask)
    print(f"[INFO] Found {num_hot} hot pixels (percentile={args.percentile})")
    if num_hot == 0:
        print("[WARN] No hot pixels found; exiting.")
        return

    print(f"[INFO] Solving steady-state diffusion (K={args.K}, lam={args.lam}, gpu_try={USE_CUPY and not args.force_cpu})")
    u, kmap = solve_steady_state(img_s, hot_mask, K=args.K, lam=args.lam, solver_tol=args.solver_tol,
                                try_gpu=USE_CUPY and not args.force_cpu, force_cpu=args.force_cpu)

    # Save basic outputs
    base = os.path.basename(infile)
    name, _ = os.path.splitext(base)
    mask_path = os.path.join(out_dir, f"{name}_pde_mask.png")
    cv2.imwrite(mask_path, (u * 255).astype(np.uint8))
    overlay_path = os.path.join(out_dir, f"{name}_pde_overlay.png")
    save_overlay_gray_with_colormap(img_s, u, overlay_path, alpha=args.alpha, cmap='jet')
    print(f"[INFO] Saved mask -> {mask_path}")
    print(f"[INFO] Saved overlay -> {overlay_path}")

    # radial baseline
    y0, x0 = np.unravel_index(np.argmax(img_s), img_s.shape)
    radial = radial_gaussian_mask(img_s, seed=(y0, x0), sigma_pixels=args.radial_sigma, temp_power=args.radial_temp_power)
    radial_path = os.path.join(out_dir, f"{name}_radial_overlay.png")
    save_overlay_gray_with_colormap(img_s, radial, radial_path, alpha=args.alpha, cmap='jet')
    print(f"[INFO] Saved radial baseline overlay -> {radial_path}")

    # Save k-map visualization
    kvis = normalize01_np(kmap)
    k_path = os.path.join(out_dir, f"{name}_k_map.png")
    cv2.imwrite(k_path, (kvis * 255).astype(np.uint8))
    print(f"[INFO] Saved k-map -> {k_path}")

    # Enhance mask using k-map edges
    enhanced_mask = enhance_mask_with_k_edges(u, kmap,
                                              method=args.edge_method,
                                              canny_thresh1=30, canny_thresh2=120,
                                              sobel_thresh=0.05,
                                              dilate_radius=args.edge_dilate,
                                              edge_strength=args.edge_strength,
                                              normalize_result=True)
    enh_mask_path = os.path.join(out_dir, f"{name}_mask_enh.png")
    cv2.imwrite(enh_mask_path, (enhanced_mask * 255).astype(np.uint8))
    enh_overlay_path = os.path.join(out_dir, f"{name}_overlay_enh.png")
    save_overlay_gray_with_colormap(img_s, enhanced_mask, enh_overlay_path, alpha=args.alpha, cmap='jet')
    print(f"[INFO] Saved enhanced mask -> {enh_mask_path}")
    print(f"[INFO] Saved enhanced overlay -> {enh_overlay_path}")

    print("[DONE] All outputs saved to", out_dir)

if __name__ == "__main__":
    main()
