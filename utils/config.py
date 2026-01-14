"""
Hydra config utilities.
"""
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

def resolve_config(cfg: DictConfig) -> DictConfig:
    """Resolve interpolations like ${project_root}."""
    OmegaConf.resolve(cfg)
    return cfg

def print_config(cfg: DictConfig, resolve: bool = True, save: bool = False) -> None:
    """Pretty-print and optionally save config."""
    if resolve:
        cfg = resolve_config(cfg)
    print(OmegaConf.to_yaml(cfg))
    
    if save:
        OmegaConf.save(cfg, Path(hydra.utils.get_original_cwd()) / "reports/config.yaml")

def get_project_root() -> Path:
    """Return absolute project root."""
    return Path(hydra.utils.get_original_cwd())
