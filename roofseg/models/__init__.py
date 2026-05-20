"""Model wrappers.

The PointUNet / DGCNN encoder lives in ``trainModel.py`` (legacy path) and
depends on PyTorch Geometric. To keep ``roofseg`` importable on machines
where PyG / torch_scatter aren't healthy, model imports here are lazy.
"""

from roofseg.models.loader import load_pointunet, build_pointunet, default_config

__all__ = ["load_pointunet", "build_pointunet", "default_config"]
