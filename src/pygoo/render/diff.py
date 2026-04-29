from __future__ import annotations

from math import comb

import numpy as np
import pygame
import torch
import torch.nn.functional as F

from ..state import SimulationState
from .viewport import Viewport

_BG_RGB = (236.0 / 255.0, 236.0 / 255.0, 236.0 / 255.0)
_FLOOR_RGB = (122.0 / 255.0, 188.0 / 255.0, 132.0 / 255.0)
_SPRING_RGB = (90.0 / 255.0, 105.0 / 255.0, 235.0 / 255.0)
_PARTICLE_RGB = (88.0 / 255.0, 88.0 / 255.0, 88.0 / 255.0)
_FIXED_RGB = (220.0 / 255.0, 55.0 / 255.0, 55.0 / 255.0)
_PARTICLE_RADIUS_PX = 10.0
_SPRING_HALF_WIDTH_PX = 0.55
_MASK_EDGE_SOFTNESS_PX = 0.75
_GRID_CACHE: dict[tuple[torch.device, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
_KERNEL_CACHE: dict[tuple[torch.device, int, int], torch.Tensor] = {}
_COLOR_CACHE: dict[
    torch.device,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def _world_grid(
    height: int, width: int, vp: Viewport, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device, height, width)
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached

    xx = torch.arange(width, device=device, dtype=torch.float32)[None, :]
    yy = torch.arange(height, device=device, dtype=torch.float32)[:, None]
    _GRID_CACHE[key] = (xx, yy)
    return xx, yy


def _colors(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    c = _COLOR_CACHE.get(device)
    if c is not None:
        return c
    bg = torch.tensor(_BG_RGB, device=device, dtype=torch.float32)
    floor = torch.tensor(_FLOOR_RGB, device=device, dtype=torch.float32)
    spring = torch.tensor(_SPRING_RGB, device=device, dtype=torch.float32)
    particle = torch.tensor(_PARTICLE_RGB, device=device, dtype=torch.float32)
    fixed = torch.tensor(_FIXED_RGB, device=device, dtype=torch.float32)
    out = (bg, floor, spring, particle, fixed)
    _COLOR_CACHE[device] = out
    return out


def render_soft_particles(
    state: SimulationState,
    vp: Viewport,
    height: int,
    width: int,
    sigma_world: float = 0.04,
    floor_y: float | None = None,
    floor_enabled: bool = False,
) -> torch.Tensor:
    device = state.pos.device
    bg_col, floor_col, spring_col, particle_col, fixed_col = _colors(device)
    img = bg_col[None, None, :].expand(height, width, 3).clone()
    img = _overlay_floor_base(img, vp, floor_y, floor_enabled, floor_col)
    xx, yy = _world_grid(height, width, vp, device)

    world_w = max(vp.x_max - vp.x_min, 1e-6)
    world_h = max(vp.y_max - vp.y_min, 1e-6)
    sx_scale = max((width - 1) / world_w, 1e-6)
    sy_scale = max((height - 1) / world_h, 1e-6)
    sigma_px_world = float(sigma_world) * min(sx_scale, sy_scale)
    mass_sigma_px = min(max(0.08 * sigma_px_world, 0.6), 2.0)

    spring_sigma_px = min(max(0.75 * mass_sigma_px, 0.5), 1.6)
    img = _render_soft_springs_exact(
        img=img,
        pos=state.pos,
        edges=state.edges,
        vp=vp,
        width=width,
        height=height,
        sigma_px=spring_sigma_px,
        spring_col=spring_col,
        xx=xx,
        yy=yy,
    )

    if state.pos.numel() == 0:
        return img

    sx = (state.pos[:, 0] - vp.x_min) * sx_scale
    sy = (vp.y_max - state.pos[:, 1]) * sy_scale

    return _render_soft_particles_exact(
        sx=sx,
        sy=sy,
        fixed=state.fixed,
        height=height,
        width=width,
        sigma_px=mass_sigma_px,
        particle_col=particle_col,
        fixed_col=fixed_col,
        xx=xx,
        yy=yy,
        img=img,
    )


def _overlay_floor_base(
    image_hwc: torch.Tensor,
    vp: Viewport,
    floor_y: float | None,
    floor_enabled: bool,
    floor_col: torch.Tensor,
) -> torch.Tensor:
    if not floor_enabled or floor_y is None:
        return image_hwc

    h = int(image_hwc.shape[0])
    if h <= 0:
        return image_hwc

    world_h = max(vp.y_max - vp.y_min, 1e-6)
    floor_row = int(round((vp.y_max - floor_y) * (h - 1) / world_h))
    floor_row = max(0, min(h, floor_row))
    if floor_row >= h:
        return image_hwc

    out = image_hwc.clone()
    out[floor_row:, :, :] = floor_col
    return out


def _render_soft_springs_exact(
    img: torch.Tensor,
    pos: torch.Tensor,
    edges: torch.Tensor,
    vp: Viewport,
    width: int,
    height: int,
    sigma_px: float,
    spring_col: torch.Tensor,
    xx: torch.Tensor,
    yy: torch.Tensor,
) -> torch.Tensor:
    if edges.numel() == 0 or pos.numel() == 0:
        return img

    world_w = max(vp.x_max - vp.x_min, 1e-6)
    world_h = max(vp.y_max - vp.y_min, 1e-6)
    sx_scale = max((width - 1) / world_w, 1e-6)
    sy_scale = max((height - 1) / world_h, 1e-6)

    seg = pos.index_select(0, edges.view(-1)).view(-1, 2, 2)
    ax = (seg[:, 0, 0] - vp.x_min) * sx_scale
    ay = (vp.y_max - seg[:, 0, 1]) * sy_scale
    bx = (seg[:, 1, 0] - vp.x_min) * sx_scale
    by = (vp.y_max - seg[:, 1, 1]) * sy_scale

    margin = 3.0 * sigma_px
    min_x = torch.minimum(ax, bx)
    max_x = torch.maximum(ax, bx)
    min_y = torch.minimum(ay, by)
    max_y = torch.maximum(ay, by)
    in_view = (
        (max_x >= -margin)
        & (min_x <= (width - 1) + margin)
        & (max_y >= -margin)
        & (min_y <= (height - 1) + margin)
    )
    if not bool(torch.any(in_view)):
        return img

    ax = ax[in_view]
    ay = ay[in_view]
    bx = bx[in_view]
    by = by[in_view]

    px = xx.expand(height, width)[None, :, :]
    py = yy.expand(height, width)[None, :, :]

    bax = (bx - ax)[:, None, None]
    bay = (by - ay)[:, None, None]
    pax = px - ax[:, None, None]
    pay = py - ay[:, None, None]
    denom = torch.clamp(bax * bax + bay * bay, min=1e-8)
    t = torch.clamp((pax * bax + pay * bay) / denom, min=0.0, max=1.0)

    cx = ax[:, None, None] + t * bax
    cy = ay[:, None, None] + t * bay
    dist = torch.sqrt((px - cx).pow(2) + (py - cy).pow(2) + 1e-8)

    line_mask = torch.sigmoid((_SPRING_HALF_WIDTH_PX - dist) / _MASK_EDGE_SOFTNESS_PX)
    alpha = torch.clamp(line_mask.sum(dim=0), min=0.0, max=1.0)
    alpha = _blur_alpha(alpha, sigma_px)

    out = img * (1.0 - alpha[..., None]) + alpha[..., None] * spring_col
    return torch.clamp(out, min=0.0, max=1.0)


def _blur_alpha(alpha: torch.Tensor, sigma_px: float) -> torch.Tensor:
    sigma = float(sigma_px)
    if sigma <= 0.01:
        return torch.clamp(alpha, min=0.0, max=1.0)

    size = max(3, int(2 * round(2.5 * sigma) + 1))
    if size % 2 == 0:
        size += 1
    kernel = _gaussian_kernel(alpha.device, channels=1, size=size)
    pad = size // 2

    x = alpha[None, None, :, :]
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    x = F.conv2d(x, kernel, groups=1)
    return torch.clamp(x[0, 0], min=0.0, max=1.0)


def _render_soft_particles_exact(
    sx: torch.Tensor,
    sy: torch.Tensor,
    fixed: torch.Tensor,
    height: int,
    width: int,
    sigma_px: float,
    particle_col: torch.Tensor,
    fixed_col: torch.Tensor,
    xx: torch.Tensor,
    yy: torch.Tensor,
    img: torch.Tensor,
) -> torch.Tensor:

    inv_two_sigma2 = 1.0 / (2.0 * sigma_px * sigma_px)
    margin = 3.0 * sigma_px
    in_view = (
        (sx >= -margin)
        & (sx <= (width - 1) + margin)
        & (sy >= -margin)
        & (sy <= (height - 1) + margin)
    )
    if not bool(torch.any(in_view)):
        return img

    sx = sx[in_view]
    sy = sy[in_view]
    fixed = fixed[in_view]

    px = xx[None, :, :]
    py = yy[None, :, :]
    dist = torch.sqrt(
        (px - sx[:, None, None]).pow(2) + (py - sy[:, None, None]).pow(2) + 1e-8
    )
    disk_mask = torch.sigmoid((_PARTICLE_RADIUS_PX - dist) / _MASK_EDGE_SOFTNESS_PX)

    fixed_w = fixed[:, None, None]
    free_w = (~fixed)[:, None, None]
    fixed_alpha = torch.clamp(
        disk_mask.masked_fill(~fixed_w, 0.0).sum(dim=0), min=0.0, max=1.0
    )
    free_alpha = torch.clamp(
        disk_mask.masked_fill(~free_w, 0.0).sum(dim=0), min=0.0, max=1.0
    )
    free_alpha = _blur_alpha(free_alpha, sigma_px)
    fixed_alpha = _blur_alpha(fixed_alpha, sigma_px)

    out = img * (1.0 - free_alpha[..., None]) + free_alpha[..., None] * particle_col
    out = out * (1.0 - fixed_alpha[..., None]) + fixed_alpha[..., None] * fixed_col
    return torch.clamp(out, min=0.0, max=1.0)


def _gaussian_kernel(device: torch.device, channels: int, size: int) -> torch.Tensor:
    size = max(3, int(size))
    if size % 2 == 0:
        size += 1

    key = (device, channels, size)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    n = size - 1
    k1 = torch.tensor(
        [float(comb(n, i)) for i in range(size)], device=device, dtype=torch.float32
    )
    k2 = torch.outer(k1, k1)
    kernel = (k2 / k2.sum())[None, None, :, :].repeat(channels, 1, 1, 1)
    _KERNEL_CACHE[key] = kernel
    return kernel


def build_gaussian_pyramid(
    image: torch.Tensor, num_levels: int, kernel_size: int = 5
) -> list[torch.Tensor]:
    if num_levels <= 0:
        return []

    x = image.permute(2, 0, 1).unsqueeze(0)
    levels: list[torch.Tensor] = [image]
    kernel = _gaussian_kernel(image.device, channels=x.shape[1], size=kernel_size)
    pad = kernel.shape[-1] // 2

    for _ in range(1, num_levels):
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        x = F.conv2d(x, kernel, stride=2, groups=x.shape[1])

        levels.append(x[0].permute(1, 2, 0))

    return levels


def render_gaussian_pyramid(
    state: SimulationState,
    vp: Viewport,
    height: int,
    width: int,
    num_levels: int,
    sigma_world: float = 0.04,
    kernel_size: int = 5,
    floor_y: float | None = None,
    floor_enabled: bool = False,
) -> list[torch.Tensor]:
    base = render_soft_particles(
        state,
        vp,
        height,
        width,
        sigma_world=sigma_world,
        floor_y=floor_y,
        floor_enabled=floor_enabled,
    )

    levels = build_gaussian_pyramid(base, num_levels, kernel_size=kernel_size)

    return levels


def blit_tensor_image(
    surface: pygame.Surface,
    image_hwc: torch.Tensor,
    x: int,
    y: int,
    target_width: int,
    target_height: int,
    smooth: bool = False,
) -> None:
    np_img = (255.0 * image_hwc.detach().clamp(0.0, 1.0).cpu().numpy()).astype(np.uint8)
    srf = pygame.surfarray.make_surface(np.transpose(np_img, (1, 0, 2)))
    if srf.get_width() != target_width or srf.get_height() != target_height:
        if smooth:
            srf = pygame.transform.smoothscale(srf, (target_width, target_height))
        else:
            srf = pygame.transform.scale(srf, (target_width, target_height))
    surface.blit(srf, (x, y))
