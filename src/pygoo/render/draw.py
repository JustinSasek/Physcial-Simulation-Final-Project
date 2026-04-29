from __future__ import annotations

import pygame

from ..state import SimulationState
from .viewport import Viewport

BG = (236, 236, 236)
FLOOR = (122, 188, 132)
SPRING = (90, 105, 235)
PARTICLE = (88, 88, 88)
FIXED = (220, 55, 55)
SELECTED = (244, 187, 68)


def draw_scene(
    surface: pygame.Surface,
    state: SimulationState,
    vp: Viewport,
    floor_y: float,
    floor_enabled: bool,
    selected_particle_idx: int = -1,
    selected_spring_idx: int = -1,
) -> None:
    surface.fill(BG)

    if floor_enabled:
        _, y0 = vp.world_to_screen(vp.x_min, floor_y)
        rect = pygame.Rect(vp.screen_x, y0, vp.screen_w, vp.screen_y + vp.screen_h - y0)
        pygame.draw.rect(surface, FLOOR, rect)

    if state.edges.numel() > 0:
        edges = state.edges.detach().cpu().numpy()
        pos = state.pos.detach().cpu().numpy()
        for i, (a, b) in enumerate(edges):
            ax, ay = vp.world_to_screen(float(pos[a, 0]), float(pos[a, 1]))
            bx, by = vp.world_to_screen(float(pos[b, 0]), float(pos[b, 1]))
            color = SELECTED if i == selected_spring_idx else SPRING
            width = 4 if i == selected_spring_idx else 2
            pygame.draw.line(surface, color, (ax, ay), (bx, by), width=width)

    if state.pos.numel() > 0:
        pos = state.pos.detach().cpu().numpy()
        fixed = state.fixed.detach().cpu().numpy()
        for i in range(pos.shape[0]):
            cx, cy = vp.world_to_screen(float(pos[i, 0]), float(pos[i, 1]))
            color = FIXED if bool(fixed[i]) else PARTICLE
            pygame.draw.circle(surface, color, (cx, cy), 10)
            if i == selected_particle_idx:
                pygame.draw.circle(surface, SELECTED, (cx, cy), 14, width=2)
