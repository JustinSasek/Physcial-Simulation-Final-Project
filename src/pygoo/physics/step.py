from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ..config import SimulationConfig
from ..state import SimulationState
from .forces import compute_spring_forces


@dataclass
class Stepper:
    cfg: SimulationConfig
    use_compile: bool = True

    def __post_init__(self) -> None:
        self._compiled: Callable | None = None
        self._shape_sig: tuple[int, int, str] | None = None
        self._compiled_rollouts: dict[tuple[int, int, int, str], Callable] = {}

    def _kernel(self, pos, vel, mass, fixed, edges, rest_len, edge_k):
        # Match original loop: predict position with current velocity, then update velocity from forces at predicted position.
        pos_pred = pos + self.cfg.time_step * vel
        spring_forces = compute_spring_forces(
            pos_pred,
            vel,
            edges,
            rest_len,
            edge_k,
            self.cfg.damping_stiffness,
            self.cfg.springs_enabled,
            self.cfg.damping_enabled,
        )
        total_forces = spring_forces
        if self.cfg.gravity_enabled:
            total_forces = total_forces.clone()
            total_forces[:, 1] = total_forces[:, 1] + mass * self.cfg.gravity_g

        inv_mass = 1.0 / torch.clamp(mass, min=1e-3)
        acc = total_forces * inv_mass[:, None]
        vel_new = vel + self.cfg.time_step * acc
        pos_new = pos_pred

        if self.cfg.floor_enabled:
            floor_mask = (pos_new[:, 1] < self.cfg.floor_y) & (vel_new[:, 1] < 0)
            pos_y = torch.where(
                floor_mask,
                torch.full_like(pos_new[:, 1], self.cfg.floor_y),
                pos_new[:, 1],
            )
            vel_y = torch.where(
                floor_mask,
                -self.cfg.floor_bounce * vel_new[:, 1],
                vel_new[:, 1],
            )
            pos_new = torch.stack((pos_new[:, 0], pos_y), dim=1)
            vel_new = torch.stack((vel_new[:, 0], vel_y), dim=1)

        fixed_expand = fixed[:, None]
        pos_new = torch.where(fixed_expand, pos, pos_new)
        vel_new = torch.where(fixed_expand, torch.zeros_like(vel_new), vel_new)
        return pos_new, vel_new

    def _make_rollout_kernel(self, num_steps: int) -> Callable:
        def _rollout(pos, vel, mass, fixed, edges, rest_len, edge_k):
            p = pos
            v = vel
            for _ in range(num_steps):
                p, v = self._kernel(p, v, mass, fixed, edges, rest_len, edge_k)
            return p, v

        return _rollout

    def _build_compiled(self):
        return torch.compile(self._kernel, mode="default", backend="inductor")

    def _build_compiled_rollout(self, num_steps: int) -> Callable:
        return torch.compile(
            self._make_rollout_kernel(num_steps), mode="default", backend="inductor"
        )

    def step(self, state: SimulationState) -> None:
        if state.pos.numel() == 0:
            return

        sig = (state.pos.shape[0], state.edges.shape[0], str(state.device))
        fn: Callable | None = None
        if self.use_compile:
            if self._compiled is None or self._shape_sig != sig:
                self._compiled = self._build_compiled()
                self._shape_sig = sig
            fn = self._compiled
        else:
            fn = self._kernel

        if fn is None:
            fn = self._kernel

        pos, vel = fn(
            state.pos,
            state.vel,
            state.mass,
            state.fixed,
            state.edges,
            state.rest_len,
            state.edge_k,
        )
        state.pos = pos
        state.vel = vel
        return

    def rollout(self, state: SimulationState, num_steps: int) -> None:
        if state.pos.numel() == 0 or num_steps <= 0:
            return

        fn: Callable
        if not self.use_compile:
            fn = self._make_rollout_kernel(num_steps)
        else:
            sig = (
                num_steps,
                state.pos.shape[0],
                state.edges.shape[0],
                str(state.device),
            )
            compiled = self._compiled_rollouts.get(sig)
            if compiled is None:
                compiled = self._build_compiled_rollout(num_steps)
                self._compiled_rollouts[sig] = compiled
            fn = compiled

        pos, vel = fn(
            state.pos,
            state.vel,
            state.mass,
            state.fixed,
            state.edges,
            state.rest_len,
            state.edge_k,
        )
        state.pos = pos
        state.vel = vel
