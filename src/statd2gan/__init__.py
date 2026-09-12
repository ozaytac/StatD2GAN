"""StatD2GAN: multi-discriminator GAN for synthetic multivariate weather sequences.

See the top-level README for the paper context, the held-out evaluation
protocol (v2), and which findings are locked vs. retracted.
"""
from .config import C, Constants, ExperimentConfig, LocationPaths, make_paths, set_seed

__all__ = ["C", "Constants", "ExperimentConfig", "LocationPaths", "make_paths", "set_seed"]

__version__ = "2.0.0"
