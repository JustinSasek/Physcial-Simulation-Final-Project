from dataclasses import dataclass, field

import torch


@dataclass
class SimulationConfig:
    _TENSOR_FIELDS = {
        "time_step",
        "gravity_g",
        "spring_stiffness",
        "damping_stiffness",
        "floor_bounce",
    }

    time_step: torch.Tensor = field(
        default_factory=lambda: torch.tensor(0.0002, dtype=torch.float32)
    )
    gravity_g: torch.Tensor = field(
        default_factory=lambda: torch.tensor(-9.8, dtype=torch.float32)
    )
    particle_mass: float = 1.0
    spring_stiffness: torch.Tensor = field(
        default_factory=lambda: torch.tensor(100.0, dtype=torch.float32)
    )
    damping_stiffness: torch.Tensor = field(
        default_factory=lambda: torch.tensor(1.0, dtype=torch.float32)
    )
    max_spring_dist: float = 0.35
    floor_y: float = -0.5
    floor_bounce: torch.Tensor = field(
        default_factory=lambda: torch.tensor(1.0, dtype=torch.float32)
    )
    floor_enabled: bool = True
    gravity_enabled: bool = True
    springs_enabled: bool = True
    damping_enabled: bool = True

    def __post_init__(self) -> None:
        for name in self._TENSOR_FIELDS:
            value = getattr(self, name)
            object.__setattr__(self, name, self._coerce_tensor(value))

    def _coerce_tensor(
        self,
        value: object,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if device is not None or dtype is not None:
                return value.to(
                    device=device or value.device, dtype=dtype or value.dtype
                )
            return value
        return torch.tensor(
            float(value),
            dtype=dtype or torch.float32,
            device=device,
        )

    def to_device(self, device: str | torch.device) -> None:
        d = torch.device(device)
        for name in self._TENSOR_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(float(value), dtype=torch.float32, device=d)
            elif value.device != d:
                requires_grad = bool(getattr(value, "requires_grad", False))
                value = value.detach().to(device=d)
                value.requires_grad_(requires_grad)
            object.__setattr__(self, name, value)

    def set_value(self, name: str, value: float | torch.Tensor | bool) -> None:
        if name in self._TENSOR_FIELDS:
            current = getattr(self, name)
            if not isinstance(current, torch.Tensor):
                object.__setattr__(self, name, self._coerce_tensor(value))
                return

            with torch.no_grad():
                current.copy_(self._coerce_tensor(value, current.device, current.dtype))
            return

        object.__setattr__(self, name, value)

    def copy_from(self, other: "SimulationConfig") -> None:
        for name in self._TENSOR_FIELDS:
            self.set_value(name, getattr(other, name))
        for name in (
            "particle_mass",
            "max_spring_dist",
            "floor_y",
            "floor_enabled",
            "gravity_enabled",
            "springs_enabled",
            "damping_enabled",
        ):
            object.__setattr__(self, name, getattr(other, name))

    def to_dict(self) -> dict[str, float | bool]:
        payload: dict[str, float | bool] = {
            "particle_mass": self.particle_mass,
            "max_spring_dist": self.max_spring_dist,
            "floor_y": self.floor_y,
            "floor_enabled": self.floor_enabled,
            "gravity_enabled": self.gravity_enabled,
            "springs_enabled": self.springs_enabled,
            "damping_enabled": self.damping_enabled,
        }
        for name in self._TENSOR_FIELDS:
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                payload[name] = float(value.detach().item())
            else:
                payload[name] = float(value)
        return payload


@dataclass
class RuntimeConfig:
    width: int = 1240
    height: int = 800
    panel_width: int = 420
    fps: int = 60
    substeps: int = 16
    world_x_min: float = -2.0
    world_x_max: float = 2.0
    world_y_min: float = -1.0
    world_y_max: float = 1.0
    device: str = "cpu"
    use_compile_in_run_mode: bool = True
    render_pyramid_levels: int = 4
    render_particle_sigma_world: float = 0.04
    render_pyramid_kernel_size: int = 5
