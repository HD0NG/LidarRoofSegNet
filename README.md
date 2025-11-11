# LidarRoofSegNet
Roof face segmentation from airborne LiDAR using Point-U-Net/PointNet++ and other networks.

- **Preprocessing**: PDAL + laspy, tiling & normalization
- **Training**: PyTorch Lightning + Hydra configs
- **Reproducibility**: DVC for data/experiments, GitHub Actions CI
- **Eval**: per-face IoU, F1, confusion, visualizations via Open3D
