from __future__ import annotations

import yaml

from ..config import SimulationConfig


def save_preset(path: str, cfg: SimulationConfig) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=True)


def load_preset(path: str, device: str | None = None) -> SimulationConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = SimulationConfig(**data)
    if device is not None:
        cfg.to_device(device)
    return cfg
