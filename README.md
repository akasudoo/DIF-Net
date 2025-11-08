# DIF-Net
Enhancing Low-Light Object Detection with Zero-Shot  Dual-Branch Illumination-Invariant Network

Thrilled to share our work was Accepted by MMAsia25!
<img width="981" height="729" alt="image" src="https://github.com/user-attachments/assets/d4c12b7d-a6c7-4ef1-8486-f1a2079d5de2" />

Abstract

 Object detection in low-light environments remains challenging due to severe image degradation, limited annotations, and the high cost of manuallabeling. While traditional enhancement methods improve visual appearance, they often hinder detection performance. In this work, we propose a lightweight zero-shot enhancement network designed specifically for object detection, enabling effective illumination correction without requiring real low-light data. Unlike pixel-level restoration approaches, our method operates at the feature level, leveraging a physics-inspired model that extracts illumination-invariant representations via Lambertian reflectance and cross-channel chrominance ratios. To enhance global feature perception with minimal computational overhead, we introduce a frequency-domain branch based on complex convolutions, and fuse it adaptively with spatial-domain features through a dual-branch architecture. The proposed module is compact and detector-agnostic, and can be seamlessly integrated into existing frameworks. Extensive experiments show that our approach significantly improves detection performance under low-light conditions, achieving a 4.6% mAP gain on ExDark and a 2.7% improvement on DarkFace.

 
