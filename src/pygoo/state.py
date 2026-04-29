from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import SimulationConfig


@dataclass
class SimulationState:
    pos: torch.Tensor
    vel: torch.Tensor
    mass: torch.Tensor
    fixed: torch.Tensor
    edges: torch.Tensor
    rest_len: torch.Tensor
    edge_k: torch.Tensor

    @classmethod
    def empty(cls, device: str = "cpu") -> "SimulationState":
        d = torch.device(device)
        return cls(
            pos=torch.empty((0, 2), dtype=torch.float32, device=d),
            vel=torch.empty((0, 2), dtype=torch.float32, device=d),
            mass=torch.empty((0,), dtype=torch.float32, device=d),
            fixed=torch.empty((0,), dtype=torch.bool, device=d),
            edges=torch.empty((0, 2), dtype=torch.int64, device=d),
            rest_len=torch.empty((0,), dtype=torch.float32, device=d),
            edge_k=torch.empty((0,), dtype=torch.float32, device=d),
        )

    @property
    def device(self) -> torch.device:
        return self.pos.device

    def clone(self) -> "SimulationState":
        return SimulationState(
            pos=self.pos.clone(),
            vel=self.vel.clone(),
            mass=self.mass.clone(),
            fixed=self.fixed.clone(),
            edges=self.edges.clone(),
            rest_len=self.rest_len.clone(),
            edge_k=self.edge_k.clone(),
        )

    def add_particle(
        self,
        xy: tuple[float, float],
        cfg: SimulationConfig,
        fixed: bool = False,
        mass: float | None = None,
        spring_stiffness: float | None = None,
    ) -> None:
        x = torch.tensor([[xy[0], xy[1]]], dtype=torch.float32, device=self.device)
        self.pos = torch.cat([self.pos, x], dim=0)
        self.vel = torch.cat(
            [self.vel, torch.zeros((1, 2), dtype=torch.float32, device=self.device)],
            dim=0,
        )
        particle_mass = cfg.particle_mass if mass is None else mass
        if fixed:
            particle_mass = max(particle_mass, 1e6)
        self.mass = torch.cat(
            [
                self.mass,
                torch.tensor([particle_mass], dtype=torch.float32, device=self.device),
            ],
            dim=0,
        )
        self.fixed = torch.cat(
            [self.fixed, torch.tensor([fixed], dtype=torch.bool, device=self.device)],
            dim=0,
        )
        new_idx = self.pos.shape[0] - 1
        if new_idx == 0:
            return

        other_idx = torch.arange(new_idx, device=self.device, dtype=torch.int64)
        d = self.pos[other_idx] - self.pos[new_idx]
        dist = torch.linalg.norm(d, dim=1)
        valid = (dist < cfg.max_spring_dist) & (dist > 1e-6)
        if not torch.any(valid):
            return

        src = other_idx[valid]
        dst = torch.full_like(src, new_idx)
        new_edges = torch.stack([src, dst], dim=1)
        new_rest = dist[valid]
        k_base = cfg.spring_stiffness if spring_stiffness is None else spring_stiffness
        new_k = k_base / torch.clamp(new_rest, min=1e-6)

        self.edges = torch.cat([self.edges, new_edges], dim=0)
        self.rest_len = torch.cat([self.rest_len, new_rest], dim=0)
        self.edge_k = torch.cat([self.edge_k, new_k], dim=0)

    def set_particle_mass(self, idx: int, mass: float) -> bool:
        if idx < 0 or idx >= self.mass.shape[0]:
            return False
        if self.fixed[idx].item():
            self.mass[idx] = max(mass, 1e6)
        else:
            self.mass[idx] = max(mass, 1e-6)
        return True

    def set_edge_stiffness(self, idx: int, stiffness: float) -> bool:
        if idx < 0 or idx >= self.edge_k.shape[0]:
            return False
        self.edge_k[idx] = max(stiffness, 0.0)
        return True
