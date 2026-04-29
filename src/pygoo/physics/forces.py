from __future__ import annotations

import torch


def compute_spring_forces(
    pos: torch.Tensor,
    vel: torch.Tensor,
    edges: torch.Tensor,
    rest_len: torch.Tensor,
    edge_k: torch.Tensor,
    damping: float | torch.Tensor,
    springs_enabled: bool,
    damping_enabled: bool,
) -> torch.Tensor:
    n = pos.shape[0]
    forces = torch.zeros_like(pos)
    if n == 0 or edges.numel() == 0:
        return forces

    i = edges[:, 0]
    j = edges[:, 1]
    dv = vel[j] - vel[i]
    d = pos[j] - pos[i]
    length = torch.linalg.norm(d, dim=1)
    inv = torch.rsqrt(torch.clamp(length * length, min=1e-12))
    direction = d * inv[:, None]

    if springs_enabled:
        ext = length - rest_len
        fs = edge_k * ext
        edge_force = fs[:, None] * direction
        forces.index_add_(0, i, edge_force)
        forces.index_add_(0, j, -edge_force)

    if damping_enabled:
        if isinstance(damping, torch.Tensor):
            if damping.numel() == 1 and float(damping.detach().item()) == 0.0:
                return forces
            fd = damping * dv
        else:
            if damping == 0.0:
                return forces
            fd = damping * dv
        forces.index_add_(0, i, fd)
        forces.index_add_(0, j, -fd)

    return forces
