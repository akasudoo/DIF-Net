# DIF-Net: Enhancing Low-Light Object Detection with Zero-Shot Dual-Branch Illumination-Invariant Network

Official PyTorch implementation of:

**Enhancing Low-Light Object Detection with Zero-Shot Dual-Branch Illumination-Invariant Network**  
ACM Multimedia Asia 2025  
Chengpeng Li, Aiwen Jiang, Jiatian Miao

[[Paper]](https://doi.org/10.1145/3743093.3771051)

---

<img width="685" height="507" alt="屏幕截图 2026-04-25 162623" src="https://github.com/user-attachments/assets/76f5fae9-9fe3-4f24-9d7e-784ad378ab3f" />


## Introduction

Object detection in low-light environments remains challenging due to severe image degradation, uneven illumination, color distortion, and the lack of large-scale annotated low-light datasets. Existing low-light enhancement methods are usually optimized for human visual perception, which may introduce artifacts and semantic inconsistency, leading to degraded detection performance.

To address this problem, we propose **DIF-Net**, a lightweight zero-shot dual-branch illumination-invariant enhancement network designed specifically for low-light object detection. Instead of performing pixel-level image restoration, DIF-Net learns detection-oriented illumination-invariant representations at the feature level.

DIF-Net contains a spatial-domain branch based on Lambertian reflectance and Cross-Chromatic Ratio modeling, and a frequency-domain branch based on complex-valued convolution and Fourier-domain illumination-invariant feature extraction. The extracted features are adaptively fused and combined with a Gray-Edge based illumination estimator to improve object localization under challenging lighting conditions.

---

## Highlights

- **Zero-shot low-light enhancement for detection**  
  DIF-Net does not require real low-light images during the enhancement stage. Synthetic low-light image pairs are generated from normal-light images for illumination-invariant feature learning.

- **Detection-oriented feature enhancement**  
  Unlike traditional low-light enhancement methods, DIF-Net operates at the feature level and is optimized for object detection rather than visual quality.

- **Dual-branch illumination-invariant representation**  
  The spatial branch extracts non-illumination features using Lambertian reflectance and Cross-Chromatic Ratio modeling, while the frequency branch captures global structural information using complex-valued convolution.

- **Lightweight and detector-agnostic design**  
  DIF-Net introduces only approximately **0.055M** additional parameters and can be integrated into different detectors such as YOLOv8n, YOLOv10n, YOLOv11n, and Faster R-CNN.

- **Strong performance on low-light benchmarks**  
  DIF-Net achieves significant improvements on ExDark and DarkFace, demonstrating strong generalization ability.

---

## Framework

The overall framework consists of the following components:

1. **Spatial-domain Non-Illumination Module**

   The spatial branch is inspired by the Lambertian reflectance model. It extracts illumination-invariant features by computing Cross-Chromatic Ratio features among RGB channels.

2. **Frequency-domain Non-Illumination Module**

   The frequency branch applies Fourier transform and complex-valued convolution to model both amplitude and phase information. This branch helps preserve structural cues such as object contours, textures, and boundaries under low illumination.

3. **Adaptive Feature Fusion Module**

   Spatial-domain and frequency-domain features are adaptively fused through learnable attention weights. A channel attention mechanism is further introduced to suppress noise and artifacts caused by feature fusion.

4. **Consistency Loss**

   A consistency loss is used to align illumination-invariant features extracted from paired normal-light and synthetic low-light images.

5. **Gray-Edge based Illumination Estimator**

   A local illumination map is estimated using a Gray-Edge prior and concatenated with illumination-invariant features to improve boundary localization and structural awareness.

---

## Method Overview

```text
Input Image
    │
    ├── Spatial-domain Non-Illumination Module
    │       └── Lambertian Reflectance + Cross-Chromatic Ratio
    │
    ├── Frequency-domain Non-Illumination Module
    │       └── Fourier Transform + Complex Convolution + Frequency CCR
    │
    ├── Adaptive Feature Fusion
    │       └── Spatial/Frequency Feature Fusion + Channel Attention
    │
    ├── Gray-Edge Illumination Estimator
    │
    └── Enhanced Detection-oriented Representation
            └── Object Detector


## Citation

If you find this work useful for your research, please cite:
@inproceedings{li2025difnet,
  title     = {Enhancing Low-Light Object Detection with Zero-Shot Dual-Branch Illumination-Invariant Network},
  author    = {Li, Chengpeng and Jiang, Aiwen and Miao, Jiatian},
  booktitle = {Proceedings of the 7th ACM International Conference on Multimedia in Asia},
  year      = {2025},
  pages     = {1--7},
  doi       = {10.1145/3743093.3771051}
}

