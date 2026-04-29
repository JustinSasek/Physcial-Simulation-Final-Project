from __future__ import annotations

import json
import math
import random

import torch

from ..config import SimulationConfig
from ..state import SimulationState


def randomize_config(cfg: SimulationConfig, rng: random.Random) -> None:
    cfg.set_value("time_step", rng.uniform(0.0001, 0.001))
    cfg.set_value("gravity_g", rng.uniform(-18.0, -0.5))
    cfg.particle_mass = 10 ** rng.uniform(math.log10(0.08), math.log10(12.0))
    cfg.set_value(
        "spring_stiffness", 10 ** rng.uniform(math.log10(25.0), math.log10(850.0))
    )
    cfg.set_value("damping_stiffness", rng.uniform(0.0, 12.0))
    cfg.set_value("floor_bounce", rng.uniform(0.05, 1.0))


def generate_random_state(
    cfg: SimulationConfig,
    device: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    rng: random.Random,
) -> SimulationState:
    state = SimulationState.empty(device)
    n = rng.randint(8, 28)
    x_margin = max((x_max - x_min) * 0.025, 0.08)
    y_margin = max((y_max - y_min) * 0.03, 0.06)
    y_lo = max(cfg.floor_y + y_margin, y_min + y_margin)
    y_hi = y_max - y_margin
    x_lo = x_min + x_margin
    x_hi = x_max - x_margin
    if x_lo >= x_hi:
        xm = 0.5 * (x_min + x_max)
        x_lo = xm - 1e-3
        x_hi = xm + 1e-3
    if y_lo >= y_hi:
        ym = 0.5 * (max(cfg.floor_y, y_min) + y_max)
        y_lo = ym - 1e-3
        y_hi = ym + 1e-3
    min_sep2 = 0.045 * 0.045

    pts: list[tuple[float, float]] = []
    for _ in range(n):
        chosen: tuple[float, float] | None = None
        for _ in range(120):
            p = (rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi))
            if all(
                (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1])
                >= min_sep2
                for q in pts
            ):
                chosen = p
                break
        pts.append(chosen if chosen is not None else p)

    masses: list[float] = []
    fixed: list[bool] = []
    for _ in range(n):
        masses.append(10 ** rng.uniform(math.log10(0.08), math.log10(25.0)))
        fixed.append(rng.random() < 0.18)
    if all(fixed):
        fixed[rng.randrange(n)] = False

    edges: set[tuple[int, int]] = set()
    for i in range(1, n):
        dists = sorted(
            (
                ((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2, j)
                for j in range(i)
            ),
            key=lambda t: t[0],
        )
        candidates = [j for _, j in dists[: min(3, len(dists))]]
        j = rng.choice(candidates)
        a, b = sorted((i, j))
        edges.add((a, b))

    connect_dist = min(max(cfg.max_spring_dist * 1.7, 0.12), 0.75)
    connect_dist2 = connect_dist * connect_dist
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            if dx * dx + dy * dy <= connect_dist2 and rng.random() < 0.24:
                edges.add((i, j))

    edge_list = sorted(edges)
    rest_len: list[float] = []
    edge_k: list[float] = []
    for i, j in edge_list:
        dx = pts[j][0] - pts[i][0]
        dy = pts[j][1] - pts[i][1]
        rest = math.sqrt(dx * dx + dy * dy)
        k = 10 ** rng.uniform(math.log10(20.0), math.log10(900.0))
        rest_len.append(max(rest, 1e-6))
        edge_k.append(k / max(rest, 1e-6))

    d = torch.device(device)
    state.pos = torch.tensor(pts, dtype=torch.float32, device=d)
    state.vel = torch.zeros((n, 2), dtype=torch.float32, device=d)
    state.mass = torch.tensor(masses, dtype=torch.float32, device=d)
    state.fixed = torch.tensor(fixed, dtype=torch.bool, device=d)
    if edge_list:
        state.edges = torch.tensor(edge_list, dtype=torch.int64, device=d)
        state.rest_len = torch.tensor(rest_len, dtype=torch.float32, device=d)
        state.edge_k = torch.tensor(edge_k, dtype=torch.float32, device=d)
    return state


def save_scene(path: str, cfg: SimulationConfig, state: SimulationState) -> None:
    payload = {
        "version": 1,
        "config": cfg.to_dict(),
        "state": {
            "pos": state.pos.detach().cpu().tolist(),
            "vel": state.vel.detach().cpu().tolist(),
            "mass": state.mass.detach().cpu().tolist(),
            "fixed": state.fixed.detach().cpu().tolist(),
            "edges": state.edges.detach().cpu().tolist(),
            "rest_len": state.rest_len.detach().cpu().tolist(),
            "edge_k": state.edge_k.detach().cpu().tolist(),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_scene(path: str, device: str) -> tuple[SimulationConfig, SimulationState]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cfg = SimulationConfig(**payload["config"])
    cfg.to_device(device)
    s = payload["state"]
    d = torch.device(device)
    state = SimulationState(
        pos=torch.tensor(s["pos"], dtype=torch.float32, device=d),
        vel=torch.tensor(s["vel"], dtype=torch.float32, device=d),
        mass=torch.tensor(s["mass"], dtype=torch.float32, device=d),
        fixed=torch.tensor(s["fixed"], dtype=torch.bool, device=d),
        edges=torch.tensor(s["edges"], dtype=torch.int64, device=d),
        rest_len=torch.tensor(s["rest_len"], dtype=torch.float32, device=d),
        edge_k=torch.tensor(s["edge_k"], dtype=torch.float32, device=d),
    )
    return cfg, state


def load_observable_state(
    path: str,
    device: str,
    cfg: SimulationConfig,
) -> SimulationState:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    state_payload = payload.get("state") if isinstance(payload, dict) else None
    data = state_payload if isinstance(state_payload, dict) else payload

    if not isinstance(data, dict):
        raise ValueError("Invalid observable scene JSON")

    particles = data.get("particles")
    if isinstance(particles, list):
        pos_list = [[float(p["x"]), float(p["y"])] for p in particles]
        mass_list = [float(p.get("mass", 1.0)) for p in particles]
        fixed_list = [bool(p.get("fixed", False)) for p in particles]
    else:
        pos_raw = data.get("pos")
        mass_raw = data.get("mass")
        if not isinstance(pos_raw, list) or not isinstance(mass_raw, list):
            raise ValueError("Observable scene requires pos/mass or particles list")
        pos_list = [[float(p[0]), float(p[1])] for p in pos_raw]
        mass_list = [float(m) for m in mass_raw]
        fixed_raw = data.get("fixed")
        if fixed_raw is None:
            fixed_list = [False] * len(pos_list)
        elif isinstance(fixed_raw, list):
            fixed_list = [bool(v) for v in fixed_raw]
        else:
            raise ValueError("Observable scene fixed must be a list when present")

    if len(pos_list) != len(mass_list) or len(pos_list) != len(fixed_list):
        raise ValueError("Observable scene has mismatched pos and mass lengths")

    springs_raw = data.get("springs", data.get("edges", []))
    if not isinstance(springs_raw, list):
        raise ValueError("Observable scene springs/edges must be a list")

    edges: list[tuple[int, int]] = []
    for s in springs_raw:
        if isinstance(s, dict):
            a = int(s["a"])
            b = int(s["b"])
        else:
            a = int(s[0])
            b = int(s[1])
        if a == b:
            continue
        lo, hi = (a, b) if a < b else (b, a)
        if lo < 0 or hi >= len(pos_list):
            raise ValueError("Observable spring index out of range")
        edges.append((lo, hi))

    edges = sorted(set(edges))

    d = torch.device(device)
    state = SimulationState.empty(device)
    n = len(pos_list)
    state.pos = torch.tensor(pos_list, dtype=torch.float32, device=d)
    state.vel = torch.zeros((n, 2), dtype=torch.float32, device=d)
    state.mass = torch.tensor(
        [
            max(cfg.particle_mass, 1e6) if fixed else cfg.particle_mass
            for fixed in fixed_list
        ],
        dtype=torch.float32,
        device=d,
    )
    state.fixed = torch.tensor(fixed_list, dtype=torch.bool, device=d)

    if edges:
        state.edges = torch.tensor(edges, dtype=torch.int64, device=d)
        src_idx = state.edges[:, 0]
        dst_idx = state.edges[:, 1]
        rest = torch.linalg.norm(state.pos[dst_idx] - state.pos[src_idx], dim=1)
        rest = torch.clamp(rest, min=1e-6)
        state.rest_len = rest
        state.edge_k = float(cfg.spring_stiffness) / rest

    return state


def load_scene_or_observable(
    path: str,
    device: str,
    default_spring_stiffness: float,
    cfg: SimulationConfig | None = None,
) -> tuple[SimulationConfig, SimulationState]:
    try:
        return load_scene(path, device)
    except Exception:
        fallback_cfg = cfg if cfg is not None else SimulationConfig()
        fallback_cfg.to_device(device)
        return (
            fallback_cfg,
            load_observable_state(path, device, fallback_cfg),
        )
