from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import cast

import imageio.v2 as imageio  # type: ignore[import-untyped]
import numpy as np
import pygame
import torch

from pygoo.render.diff import render_soft_particles

OPT_LR = 0.1  # Adam simple ex
# OPT_LR = 1.0 # Adam
OPT_LOSS_THRESHOLD = 0.000001 # overriden by CLI arg
OPT_TEMPORAL_DISCOUNT = 0.97
OPT_HORIZON_STEP = 4
OPT_MAX_HORIZON = 120
OPT_AUTO_ADVANCE_NO_IMPROVE = False
OPT_NO_IMPROVE_PATIENCE = 8
OPT_LOSS_IMPROVEMENT_EPSILON = 1e-6

from .config import RuntimeConfig, SimulationConfig
from .io.scene import (
    generate_random_state,
    load_observable_state,
    load_scene,
    load_scene_or_observable,
    randomize_config,
    save_scene,
)
from .optimize import DifferentiableFitter
from .physics import Stepper
from .render import Viewport, blit_tensor_image, draw_scene, render_gaussian_pyramid
from .state import SimulationState
from .trajectory import rollout_state_trajectory, save_trajectory_npz
from .ui import ControlPanel


def make_run_config(
    *,
    width: int,
    height: int,
    panel_width: int,
    substeps: int,
    device: str,
    use_compile_in_run_mode: bool = False,
) -> RuntimeConfig:
    return RuntimeConfig(
        width=width,
        height=height,
        panel_width=panel_width,
        substeps=substeps,
        device=device,
        use_compile_in_run_mode=use_compile_in_run_mode,
    )


def make_scene_viewport(run_cfg: RuntimeConfig) -> Viewport:
    scene_w = run_cfg.width - run_cfg.panel_width
    return Viewport(
        run_cfg.world_x_min,
        run_cfg.world_x_max,
        run_cfg.world_y_min,
        run_cfg.world_y_max,
        run_cfg.panel_width,
        0,
        scene_w,
        run_cfg.height,
    )


def scene_size(run_cfg: RuntimeConfig) -> tuple[int, int]:
    return run_cfg.width - run_cfg.panel_width, run_cfg.height


def seed_state(state: SimulationState, cfg: SimulationConfig) -> None:
    for x, y in [(-0.12, -0.04), (0.12, -0.05), (-0.02, 0.16), (0.18, 0.22)]:
        state.add_particle((x, y), cfg, fixed=False)


def _pick_particle_idx(
    mx: int, my: int, state: SimulationState, vp: Viewport, pixel_radius: float = 14.0
) -> int:
    if state.pos.numel() == 0:
        return -1
    pos = state.pos.detach().cpu().numpy()
    best_idx = -1
    best_d2 = pixel_radius * pixel_radius
    for i in range(pos.shape[0]):
        sx, sy = vp.world_to_screen(float(pos[i, 0]), float(pos[i, 1]))
        d2 = float((sx - mx) * (sx - mx) + (sy - my) * (sy - my))
        if d2 <= best_d2:
            best_d2 = d2
            best_idx = i
    return best_idx


def _point_segment_distance_sq(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-12:
        return apx * apx + apy * apy
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx = ax + t * abx
    cy = ay + t * aby
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy


def _pick_spring_idx(
    mx: int, my: int, state: SimulationState, vp: Viewport, pixel_radius: float = 9.0
) -> int:
    if state.edges.numel() == 0 or state.pos.numel() == 0:
        return -1
    edges = state.edges.detach().cpu().numpy()
    pos = state.pos.detach().cpu().numpy()
    best_idx = -1
    best_d2 = pixel_radius * pixel_radius
    for i, (a, b) in enumerate(edges):
        ax, ay = vp.world_to_screen(float(pos[a, 0]), float(pos[a, 1]))
        bx, by = vp.world_to_screen(float(pos[b, 0]), float(pos[b, 1]))
        d2 = _point_segment_distance_sq(float(mx), float(my), ax, ay, bx, by)
        if d2 <= best_d2:
            best_d2 = d2
            best_idx = i
    return best_idx


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="pygoo simulator")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument(
        "--frames", type=int, default=0, help="Stop after N frames (0 = run forever)"
    )
    p.add_argument("--headless", action="store_true")
    p.add_argument("--screenshot", default="")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--seed-demo", action="store_true")
    p.add_argument("--autorun", action="store_true")
    p.add_argument("--scene-json", default="")
    p.add_argument("--observable-json", default="")
    p.add_argument("--target-video", default="")
    p.add_argument("--log_dir", default="")
    p.add_argument("--optimize-start", action="store_true")
    p.add_argument("--demo-mass-only", action="store_true")
    p.add_argument("--iterations", type=int, default=0, help="Stop optimization after N iterations (0 = unlimited)")
    p.add_argument("--set", action="append", default=[], help="Override simulation parameter(s). Repeatable. KEY=VALUE dot-path syntax, e.g. particle.2.mass=5.0 or time_step=0.001 or floor.bounce=0.8")
    p.add_argument("--freeze-params", default="", help="Comma-separated list of parameter keys to freeze (disable gradients) during optimization, e.g. 'mass,edge_k' or 'time_step'")
    p.add_argument("--optimize-params", default="", help="Comma-separated list of parameter keys to optimize, e.g. 'mass,edge_k' or 'time_step'. If not specified, all optimizable parameters will be trained.")
    p.add_argument("--curriculum-advance-threshold", type=float, default=None, help="Loss threshold for curriculum advancement (default: uses loss_threshold)")

    # Add subcommands for rendering a scene from a JSON without running the GUI
    subparsers = p.add_subparsers(dest="command")
    render_p = subparsers.add_parser("render-video", help="Render a scene JSON to a preview video")
    render_p.add_argument("--json", required=True, help="Scene or observable JSON to load")
    render_p.add_argument("--output", required=True, help="Output video path")
    render_p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    return p


def _resolve_cli_device(device: str) -> str:
    d = str(device).strip().lower()
    if d == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested, but CUDA is not available")
    if d == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not bool(mps_backend.is_available()):
            raise ValueError("--device mps requested, but MPS is not available")
    return d


def _copy_cfg(dst: SimulationConfig, src: SimulationConfig) -> None:
    dst.copy_from(src)


def _export_preview_video(
    out_path: str,
    state: SimulationState,
    sim_cfg: SimulationConfig,
    run_cfg: RuntimeConfig,
    num_frames: int = 180,
    fps: int = 30,
    timeout_seconds: float = 30.0,
    sigma_world: float = 0.04,
    kernel_size: int = 5,
) -> None:
    scene_w, scene_h = scene_size(run_cfg)

    zoom_out = 1.30
    world_cx = 0.5 * (run_cfg.world_x_min + run_cfg.world_x_max)
    world_cy = 0.5 * (run_cfg.world_y_min + run_cfg.world_y_max)
    world_half_w = 0.5 * (run_cfg.world_x_max - run_cfg.world_x_min) * zoom_out
    world_half_h = 0.5 * (run_cfg.world_y_max - run_cfg.world_y_min) * zoom_out

    vp = Viewport(
        world_cx - world_half_w,
        world_cx + world_half_w,
        world_cy - world_half_h,
        world_cy + world_half_h,
        0,
        0,
        scene_w,
        scene_h,
    )

    sim_state = state.clone()
    stepper = Stepper(sim_cfg, use_compile=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Ensure container supports FFV1
    if not out_path.lower().endswith((".avi", ".mkv")):
        out_path = os.path.splitext(out_path)[0] + ".avi"

    t0 = time.perf_counter()

    with imageio.get_writer(
        out_path, fps=fps, format="ffmpeg", codec="ffv1", pixelformat="rgb24"
    ) as writer:
        for i in range(num_frames):
            if i > 0:
                if (
                    timeout_seconds > 0
                    and (time.perf_counter() - t0) >= timeout_seconds
                ):
                    print(
                        f"[export] preview timeout after {timeout_seconds:.1f}s; "
                        f"wrote {i} frames"
                    )
                    break
                stepper.rollout(sim_state, run_cfg.substeps)

            frame: torch.Tensor = render_soft_particles(
                sim_state,
                vp,
                scene_h,
                scene_w,
                sigma_world=sigma_world,
                floor_y=sim_cfg.floor_y,
                floor_enabled=sim_cfg.floor_enabled,
            )  # [H, W, 3] float32 in [0, 1]

            frame_np = (
                (frame.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            )

            writer.append_data(frame_np)


def _export_trajectory_npz(
    out_path: str,
    state: SimulationState,
    sim_cfg: SimulationConfig,
    run_cfg: RuntimeConfig,
    num_frames: int = 180,
    timeout_seconds: float = 30.0,
) -> None:
    if timeout_seconds > 0:
        max_frames = max(1, min(num_frames, int(timeout_seconds * 12)))
        traj = rollout_state_trajectory(state, sim_cfg, run_cfg.substeps, max_frames)
        if max_frames < num_frames:
            print(
                f"[export] trajectory timeout budget used; wrote {max_frames}/{num_frames} frames"
            )
    else:
        traj = rollout_state_trajectory(state, sim_cfg, run_cfg.substeps, num_frames)
    cfg = sim_cfg.to_dict()
    save_trajectory_npz(
        out_path,
        traj,
        substeps=run_cfg.substeps,
        metadata={
            "time_step": cfg["time_step"],
            "gravity_g": cfg["gravity_g"],
            "damping_stiffness": cfg["damping_stiffness"],
            "spring_stiffness": cfg["spring_stiffness"],
            "floor_y": cfg["floor_y"],
            "floor_bounce": cfg["floor_bounce"],
            "mass_mean": state.mass.mean().item() if state.mass.numel() > 0 else 0.0,
            "edge_k_mean": (
                state.edge_k.mean().item() if state.edge_k.numel() > 0 else 0.0
            ),
        },
    )


def _draw_pyramid_stack(
    screen: pygame.Surface,
    levels: list,
    panel_width: int,
    scene_width: int,
    scene_height: int,
) -> None:
    if not levels:
        return

    blit_tensor_image(
        screen,
        levels[0],
        panel_width,
        0,
        scene_width,
        scene_height,
        smooth=False,
    )

    parent_x, parent_y, parent_w, parent_h = panel_width, 0, scene_width, scene_height
    for level_img in levels[1:]:
        w = max(72, parent_w // 2)
        h = max(56, parent_h // 2)
        x = parent_x + parent_w - w - 22
        y = parent_y + parent_h - h - 22

        frame = pygame.Rect(x - 3, y - 3, w + 6, h + 6)
        pygame.draw.rect(screen, (238, 238, 238), frame)
        pygame.draw.rect(screen, (170, 170, 170), frame, 1)

        blit_tensor_image(screen, level_img, x, y, w, h, smooth=False)
        parent_x, parent_y, parent_w, parent_h = x, y, w, h


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        args.device = _resolve_cli_device(args.device)
    except ValueError as exc:
        print(exc)
        return
    if args.curriculum_advance_threshold:
        global OPT_LOSS_THRESHOLD
        OPT_LOSS_THRESHOLD = args.curriculum_advance_threshold
    # Handle simple subcommands (render-video) and exit
    if getattr(args, "command", None) == "render-video":
        # headless rendering of a scene JSON to a preview video and exit
        sim_cfg = SimulationConfig()
        try:
            loaded_cfg, loaded_state = load_scene_or_observable(
                args.json,
                args.device,
                default_spring_stiffness=float(sim_cfg.spring_stiffness),
                cfg=sim_cfg,
            )
            sim_cfg = loaded_cfg
            sim_cfg.to_device(args.device)
            _export_preview_video(args.output, loaded_state, sim_cfg, make_run_config(width=RuntimeConfig().width, height=RuntimeConfig().height, panel_width=RuntimeConfig().panel_width, substeps=RuntimeConfig().substeps, device=args.device))
            print(f"Rendered {args.json} -> {args.output}")
            return
        except Exception as exc:
            print(f"render-video failed: {exc}")
            return
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    sim_cfg = SimulationConfig()
    use_compile = args.compile and not args.no_compile
    run_cfg = make_run_config(
        width=RuntimeConfig().width,
        height=RuntimeConfig().height,
        panel_width=RuntimeConfig().panel_width,
        substeps=RuntimeConfig().substeps,
        device=args.device,
        use_compile_in_run_mode=use_compile,
    )
    sim_cfg.to_device(run_cfg.device)

    pygame.init()
    screen = pygame.display.set_mode((run_cfg.width, run_cfg.height))
    pygame.display.set_caption("pygoo")
    clock = pygame.time.Clock()

    panel = ControlPanel(
        (run_cfg.width, run_cfg.height),
        run_cfg.panel_width,
        sim_cfg,
        pyramid_levels=run_cfg.render_pyramid_levels,
    )

    # capture any scene/observable loaded at startup as ground-truth source
    startup_ground_truth_cfg = None
    startup_ground_truth_state = None

    # CSV logging handles
    _opt_csv_fp = None
    _opt_csv_writer = None
    _opt_csv_fieldnames: list[str] | None = None

    state = SimulationState.empty(run_cfg.device)
    if args.seed_demo:
        seed_state(state, sim_cfg)

    if args.scene_json:
        try:
            loaded_cfg, loaded_state = load_scene_or_observable(
                args.scene_json,
                run_cfg.device,
                default_spring_stiffness=float(sim_cfg.spring_stiffness),
                cfg=sim_cfg,
            )
            _copy_cfg(sim_cfg, loaded_cfg)
            sim_cfg.to_device(run_cfg.device)
            state = loaded_state
            panel.refresh_from_config()
            print(f"Loaded startup scene: {args.scene_json}")
            startup_ground_truth_cfg = loaded_cfg
            startup_ground_truth_state = loaded_state
        except Exception as exc:
            print(f"Startup scene load failed: {exc}")

    if args.observable_json:
        try:
            state = load_observable_state(
                args.observable_json,
                run_cfg.device,
                sim_cfg,
            )
            panel.refresh_from_config()
            print(f"Loaded startup observables: {args.observable_json}")
            startup_ground_truth_cfg = sim_cfg
            startup_ground_truth_state = state
        except Exception as exc:
            print(f"Startup observables load failed: {exc}")

    # Apply CLI --set overrides (user-specified parameter overrides)
    def _apply_cli_set(kv: str) -> None:
        if "=" not in kv:
            raise ValueError(f"Invalid --set entry (missing '='): {kv}")
        key, raw = kv.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        try:
            val = float(raw)
        except Exception:
            raise ValueError(f"Unsupported value for --set: {raw}")

        # particle.<i>.mass
        if key.startswith("particle.") and key.endswith(".mass"):
            parts = key.split(".")
            if len(parts) != 3:
                raise ValueError(f"Invalid particle key: {key}")
            idx = int(parts[1])
            if idx < 0 or idx >= state.mass.shape[0]:
                raise ValueError(f"Particle index out of range: {idx}")
            state.set_particle_mass(idx, float(val))
            return

        # edge.<i>.k
        if key.startswith("edge.") and key.endswith(".k"):
            parts = key.split(".")
            if len(parts) != 3:
                raise ValueError(f"Invalid edge key: {key}")
            idx = int(parts[1])
            if idx < 0 or idx >= state.edge_k.shape[0]:
                raise ValueError(f"Edge index out of range: {idx}")
            state.set_edge_stiffness(idx, float(val))
            return

        # normalize dotted keys to underscore for config fields
        canonical = key.replace(".", "_")
        # accept 'timestep' as alias for 'time_step'
        if canonical == "timestep":
            canonical = "time_step"

        # check config has attribute
        if hasattr(sim_cfg, canonical):
            try:
                sim_cfg.set_value(canonical, float(val))
            except Exception as exc:
                raise ValueError(f"Failed to set {canonical}: {exc}")
            return

        raise ValueError(f"Unrecognized --set key: {key}")
    
    for entry in args.set:
        try:
            _apply_cli_set(entry)
            print(f"Applied override: {entry}")
        except Exception as exc:
            print(f"Error applying --set {entry}: {exc}")
            raise

    # freeze and unfreeze parameters according to CLI args
    if args.freeze_params:
        freeze_keys = set(k.strip() for k in args.freeze_params.split(",") if k.strip())
        for k in freeze_keys:
            if k == "mass":
                panel.mass_cfg_locked = True
            elif k == "spring_k":
                panel.spring_k_locked = True
            else:
                panel.value_locked[k] = True
    if args.optimize_params:
        optimize_keys = set(k.strip() for k in args.optimize_params.split(",") if k.strip())
        for k in optimize_keys:
            if k == "mass":
                panel.mass_cfg_locked = False
            elif k == "spring_k":
                panel.spring_k_locked = False
            else:
                panel.value_locked[k] = False
    panel._update_toggle_colors()

    eager_stepper = Stepper(sim_cfg, use_compile=False)
    compiled_stepper = Stepper(sim_cfg, use_compile=run_cfg.use_compile_in_run_mode)

    running = args.autorun
    frame = 0
    done = False
    selected_particle_idx = -1
    selected_spring_idx = -1
    fitter = DifferentiableFitter(
        run_cfg,
        num_levels=run_cfg.render_pyramid_levels,
        sigma_world=run_cfg.render_particle_sigma_world,
        kernel_size=run_cfg.render_pyramid_kernel_size,
    )
    last_fit_loss: float | None = None
    last_fit_status = "idle"
    opt_steps = 0
    pending_opt_start = bool(args.optimize_start)

    def _stop_optimization() -> None:
        panel.set_optimization_running(False)
        nonlocal _opt_csv_fp, _opt_csv_writer
        try:
            if _opt_csv_fp is not None:
                _opt_csv_fp.close()
        except Exception:
            pass
        _opt_csv_fp = None
        _opt_csv_writer = None

    def _start_optimization() -> None:
        nonlocal opt_steps, last_fit_status
        if not fitter.has_target:
            last_fit_status = "no target video loaded"
            _stop_optimization()
            return
        ok, msg = fitter.build_optimizer(state, sim_cfg)
        if not ok:
            print(f"Failed to start optimization: {msg}")
            last_fit_status = msg or "no trainable variables selected"
            _stop_optimization()
            return
        opt_steps = 0
        last_fit_status = "running"
        panel.set_optimization_running(True)
        # Setup CSV logging for optimization
        nonlocal _opt_csv_fp, _opt_csv_writer, _opt_csv_fieldnames
        try:
            if args.scene_json:
                scene_name = Path(args.scene_json).parent.name
            elif args.observable_json:
                scene_name = Path(args.observable_json).parent.name
            else:
                scene_name = "scene"
            logs_dir = Path(args.log_dir)
            logs_dir.mkdir(parents=True, exist_ok=True)
            csv_path = logs_dir / f"optim_log.csv"

            # determine optimizable parameter keys
            keys: list[str] = []
            if fitter.train_mass and state.mass.numel() > 0:
                for i in range(state.mass.shape[0]):
                    keys.append(f"particle.{i}.mass")
            if fitter.train_edge_k and state.edge_k.numel() > 0:
                for i in range(state.edge_k.shape[0]):
                    keys.append(f"edge.{i}.k")
            opt_cfg = getattr(fitter, "optimize_cfg", {}) or {}
            for k, do_train in opt_cfg.items():
                if do_train:
                    keys.append(k)

            fieldnames = ["iteration", "loss"] + keys + [f"gt.{k}" for k in keys]
            import csv

            _opt_csv_fp = open(csv_path, "w", newline="", encoding="utf-8")
            _opt_csv_writer = csv.DictWriter(_opt_csv_fp, fieldnames=fieldnames)
            _opt_csv_writer.writeheader()
            _opt_csv_fieldnames = fieldnames
            # populate ground-truth values from startup-loaded scene/observable if available
            _opt_gt_values = {}
            if startup_ground_truth_state is not None:
                for k in keys:
                    if k.startswith("particle.") and k.endswith(".mass"):
                        idx = int(k.split(".")[1])
                        if 0 <= idx < startup_ground_truth_state.mass.shape[0]:
                            _opt_gt_values[f"gt.{k}"] = float(
                                startup_ground_truth_state.mass[idx].item()
                            )
                        else:
                            _opt_gt_values[f"gt.{k}"] = None
                    elif k.startswith("edge.") and k.endswith(".k"):
                        idx = int(k.split(".")[1])
                        if (
                            startup_ground_truth_state is not None
                            and 0 <= idx < startup_ground_truth_state.edge_k.shape[0]
                        ):
                            _opt_gt_values[f"gt.{k}"] = float(
                                startup_ground_truth_state.edge_k[idx].item()
                            )
                        else:
                            _opt_gt_values[f"gt.{k}"] = None
                    else:
                        # config values from startup_ground_truth_cfg
                        if startup_ground_truth_cfg is not None:
                            val = getattr(startup_ground_truth_cfg, k, None)
                            if isinstance(val, float) or isinstance(val, int):
                                _opt_gt_values[f"gt.{k}"] = float(val)
                            else:
                                try:
                                    _opt_gt_values[f"gt.{k}"] = float(
                                        getattr(startup_ground_truth_cfg, k)
                                    )
                                except Exception:
                                    _opt_gt_values[f"gt.{k}"] = None
                        else:
                            _opt_gt_values[f"gt.{k}"] = None
            else:
                for k in keys:
                    _opt_gt_values[f"gt.{k}"] = None
            # write ground-truth header row as first row with iteration=-1
            row = {n: None for n in fieldnames}
            row["iteration"] = -1
            for gt_k, v in _opt_gt_values.items():
                row[gt_k] = v
            _opt_csv_writer.writerow(row)
            # write current state too as iteration=0
            row = {n: None for n in fieldnames}
            row["iteration"] = 0
            for k in keys:
                if k.startswith("particle.") and k.endswith(".mass"):
                    idx = int(k.split(".")[1])
                    if 0 <= idx < state.mass.shape[0]:
                        row[k] = float(state.mass[idx].item())
                elif k.startswith("edge.") and k.endswith(".k"):
                    idx = int(k.split(".")[1])
                    if 0 <= idx < state.edge_k.shape[0]:
                        row[k] = float(state.edge_k[idx].item())
                else:
                    val = getattr(sim_cfg, k, None)
                    if isinstance(val, float) or isinstance(val, int):
                        row[k] = float(val)
                    else:
                        try:
                            row[k] = float(getattr(sim_cfg, k))
                        except Exception:
                            row[k] = None
            _opt_csv_writer.writerow(row)
        except Exception as exc:
            print(f"Failed to create optimization CSV log: {exc}")

    def _rebuild_steppers() -> tuple[Stepper, Stepper]:
        return (
            Stepper(sim_cfg, use_compile=False),
            Stepper(sim_cfg, use_compile=run_cfg.use_compile_in_run_mode),
        )

    if args.target_video:
        try:
            frame_count = fitter.load_target_video(
                args.target_video,
                device=run_cfg.device,
                scene_width=run_cfg.width - run_cfg.panel_width,
                scene_height=run_cfg.height,
            )
            last_fit_status = f"loaded {frame_count} frames"
            print(f"Loaded startup target video: {args.target_video}")
        except Exception as exc:
            last_fit_status = f"video load failed: {exc}"
            print(f"Startup target video load failed: {exc}")

    while not done:
        dt_ui = clock.tick(run_cfg.fps) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                continue

            running, reset, actions = panel.process(event, running)
            if reset or bool(actions.get("reset_all")):
                _copy_cfg(sim_cfg, SimulationConfig())
                sim_cfg.to_device(run_cfg.device)
                state = SimulationState.empty(run_cfg.device)
                selected_particle_idx = -1
                selected_spring_idx = -1
                fitter.reset_optimizer()
                _stop_optimization()
                if args.seed_demo:
                    seed_state(state, sim_cfg)
                panel.refresh_from_config()
                eager_stepper, compiled_stepper = _rebuild_steppers()

            if bool(actions.get("reset_masses")):
                state = SimulationState.empty(run_cfg.device)
                selected_particle_idx = -1
                selected_spring_idx = -1
                fitter.reset_optimizer()
                _stop_optimization()
                eager_stepper, compiled_stepper = _rebuild_steppers()

            if (
                actions["apply_particle_mass"] is not None
                and selected_particle_idx >= 0
            ):
                state.set_particle_mass(
                    selected_particle_idx, float(actions["apply_particle_mass"])
                )
            if actions["apply_spring_k"] is not None and selected_spring_idx >= 0:
                state.set_edge_stiffness(
                    selected_spring_idx, float(actions["apply_spring_k"])
                )

            if bool(actions["random_scene"]):
                rng = random.Random()
                # randomize_config(sim_cfg, rng)
                state = generate_random_state(
                    sim_cfg,
                    run_cfg.device,
                    run_cfg.world_x_min,
                    run_cfg.world_x_max,
                    run_cfg.world_y_min,
                    run_cfg.world_y_max,
                    rng,
                )
                selected_particle_idx = -1
                selected_spring_idx = -1
                fitter.reset_optimizer()
                _stop_optimization()
                panel.refresh_from_config()
                eager_stepper, compiled_stepper = _rebuild_steppers()

            import_path = actions.get("import_path")
            if isinstance(import_path, str) and import_path:
                path = import_path
                if path:
                    try:
                        loaded_cfg, loaded_state = load_scene(path, run_cfg.device)
                        _copy_cfg(sim_cfg, loaded_cfg)
                        sim_cfg.to_device(run_cfg.device)
                        state = loaded_state
                        selected_particle_idx = -1
                        selected_spring_idx = -1
                        fitter.reset_optimizer()
                        _stop_optimization()
                        panel.refresh_from_config()
                        eager_stepper, compiled_stepper = _rebuild_steppers()
                    except Exception as exc:
                        print(f"Import failed: {exc}")

            import_observable_path = actions.get("import_observable_path")
            if isinstance(import_observable_path, str) and import_observable_path:
                path = import_observable_path
                if path:
                    try:
                        state = load_observable_state(
                            path,
                            run_cfg.device,
                            sim_cfg,
                        )
                        selected_particle_idx = -1
                        selected_spring_idx = -1
                        _stop_optimization()
                        eager_stepper, compiled_stepper = _rebuild_steppers()
                        print(f"Imported observable scene: {path}")
                    except Exception as exc:
                        print(f"Observable import failed: {exc}")

            target_video_path = actions.get("target_video_path")
            if isinstance(target_video_path, str) and target_video_path:
                try:
                    frame_count = fitter.load_target_video(
                        target_video_path,
                        device=run_cfg.device,
                        scene_width=run_cfg.width - run_cfg.panel_width,
                        scene_height=run_cfg.height,
                    )
                    last_fit_loss = None
                    last_fit_status = f"loaded {frame_count} frames"
                    _stop_optimization()
                    print(
                        f"Loaded target video: {target_video_path} ({frame_count} frames)"
                    )
                except Exception as exc:
                    last_fit_status = f"video load failed: {exc}"
                    print(f"Target video load failed: {exc}")

            if bool(actions.get("reset_optimizer")):
                fitter.reset_optimizer()
                last_fit_status = "optimizer reset"

            if bool(actions.get("optimize_start")):
                pending_opt_start = True

            if bool(actions.get("optimize_stop")):
                fitter.reset_optimizer()
                _stop_optimization()
                last_fit_status = "stopped"

            export_path = actions.get("export_path")
            if isinstance(export_path, str) and export_path:
                path = export_path
                if path:
                    try:
                        save_scene(path, sim_cfg, state)
                        video_path = str(Path(path).with_suffix(".avi"))
                        trajectory_path = str(Path(path).with_suffix(".traj.npz"))
                        _export_preview_video(video_path, state, sim_cfg, run_cfg)
                        _export_trajectory_npz(trajectory_path, state, sim_cfg, run_cfg)
                        print(f"Exported scene: {path}")
                        print(f"Exported preview video: {video_path}")
                        print(f"Exported trajectory: {trajectory_path}")
                    except Exception as exc:
                        print(f"Export failed: {exc}")

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if panel.file_dialog_open():
                    continue
                mx, my = event.pos
                if mx >= run_cfg.panel_width:
                    vp = make_scene_viewport(run_cfg)
                    if panel.click_add_mode:
                        wx, wy = vp.screen_to_world(mx, my)
                        state.add_particle(
                            (wx, wy),
                            sim_cfg,
                            fixed=panel.new_particle_fixed,
                            mass=sim_cfg.particle_mass,
                            spring_stiffness=cast(
                                float, sim_cfg.spring_stiffness.item()
                            ),
                        )
                        eager_stepper = Stepper(sim_cfg, use_compile=False)
                        compiled_stepper = Stepper(
                            sim_cfg, use_compile=run_cfg.use_compile_in_run_mode
                        )
                        fitter.reset_optimizer()
                        _stop_optimization()
                    else:
                        selected_particle_idx = _pick_particle_idx(mx, my, state, vp)
                        if selected_particle_idx >= 0:
                            selected_spring_idx = -1
                        else:
                            selected_spring_idx = _pick_spring_idx(mx, my, state, vp)

        if running:
            stepper = (
                compiled_stepper if run_cfg.use_compile_in_run_mode else eager_stepper
            )
            stepper.rollout(state, run_cfg.substeps)

        opt_cfg = panel.optimization_settings()
        opt_cfg["lr"] = OPT_LR
        opt_cfg["loss_threshold"] = OPT_LOSS_THRESHOLD
        opt_cfg["temporal_discount"] = OPT_TEMPORAL_DISCOUNT
        opt_cfg["horizon_step"] = OPT_HORIZON_STEP
        opt_cfg["max_horizon"] = OPT_MAX_HORIZON
        opt_cfg["auto_advance_no_improve"] = OPT_AUTO_ADVANCE_NO_IMPROVE
        opt_cfg["no_improve_patience"] = OPT_NO_IMPROVE_PATIENCE
        opt_cfg["loss_improvement_epsilon"] = OPT_LOSS_IMPROVEMENT_EPSILON
        optimize_cfg_raw = opt_cfg.get("optimize_cfg")
        optimize_cfg = None
        if isinstance(optimize_cfg_raw, dict):
            optimize_cfg = {str(k): bool(v) for k, v in optimize_cfg_raw.items()}

        fitter.configure(
            train_mass=bool(opt_cfg["train_mass"]),
            train_edge_k=bool(opt_cfg["train_edge_k"]),
            lr=float(cast(float, opt_cfg["lr"])),
            loss_threshold=float(cast(float, opt_cfg["loss_threshold"])),
            temporal_discount=float(cast(float, opt_cfg["temporal_discount"])),
            horizon_step=int(cast(int, opt_cfg["horizon_step"])),
            max_horizon=int(cast(int, opt_cfg["max_horizon"])),
            optimize_cfg=optimize_cfg,
            auto_advance_no_improve=bool(opt_cfg["auto_advance_no_improve"]),
            no_improve_patience=int(cast(int, opt_cfg["no_improve_patience"])),
            loss_improvement_epsilon=float(
                cast(float, opt_cfg["loss_improvement_epsilon"])
            ),
        )

        vp = make_scene_viewport(run_cfg)

        if pending_opt_start:
            _start_optimization()
            pending_opt_start = False

        run_opt_step = panel.optimize_autorun
        if run_opt_step:
            fit_result = fitter.step(state, sim_cfg, vp, run_cfg.substeps)
            if fit_result.ok:
                last_fit_loss = fit_result.loss
                opt_steps += 1
                print(
                    "[opt] "
                    f"step={opt_steps} "
                    f"total_loss={fit_result.loss:.9f} "
                    f"curriculum={('advanced' if fit_result.schedule_advanced else ('eligible' if fit_result.eligible_to_advance else 'waiting'))} "
                    f"mass_grad={fit_result.mass_grad_norm:.3e} "
                    f"edge_k_grad={fit_result.edge_k_grad_norm:.3e} "
                )
                last_fit_status = "running"
                # CSV logging row
                try:
                    with torch.no_grad():
                        if _opt_csv_writer is not None and _opt_csv_fieldnames is not None:
                            row = {k: None for k in _opt_csv_fieldnames}
                            row["iteration"] = opt_steps
                            row["loss"] = float(fit_result.loss)
                            if fitter.train_mass and state.mass.numel() > 0:
                                for i in range(state.mass.shape[0]):
                                    row[f"particle.{i}.mass"] = float(state.mass[i].item())
                            if fitter.train_edge_k and state.edge_k.numel() > 0:
                                for i in range(state.edge_k.shape[0]):
                                    row[f"edge.{i}.k"] = float(state.edge_k[i].item())
                            opt_cfg_local = getattr(fitter, "optimize_cfg", {}) or {}
                            for k, do_train in opt_cfg_local.items():
                                if do_train:
                                    try:
                                        val = getattr(sim_cfg, k)
                                        row[k] = float(val)
                                    except Exception:
                                        row[k] = None
                            _opt_csv_writer.writerow(row)
                            try:
                                _opt_csv_fp.flush()
                            except Exception:
                                pass
                except Exception as exc:
                    print(f"CSV logging error: {exc}")
                # stop if iterations limit reached
                if args.iterations and args.iterations > 0 and opt_steps >= args.iterations:
                    fitter.reset_optimizer()
                    panel.set_optimization_running(False)
                    last_fit_status = f"reached iterations limit {args.iterations}"
                    exit(0)
            else:
                last_fit_status = fit_result.message or "step failed"
                _stop_optimization()

        panel.sync_config_values()
        selected_particle_mass = (
            float(state.mass[selected_particle_idx].item())
            if 0 <= selected_particle_idx < state.mass.shape[0]
            else None
        )
        selected_spring_k = (
            float(state.edge_k[selected_spring_idx].item())
            if 0 <= selected_spring_idx < state.edge_k.shape[0]
            else None
        )
        panel.set_selection_info(
            selected_particle_idx,
            selected_spring_idx,
            selected_particle_mass,
            selected_spring_k,
        )
        if panel.render_mode == "Normal":
            draw_scene(
                screen,
                state,
                vp,
                sim_cfg.floor_y,
                sim_cfg.floor_enabled,
                selected_particle_idx=selected_particle_idx,
                selected_spring_idx=selected_spring_idx,
            )
        else:
            levels = render_gaussian_pyramid(
                state,
                vp,
                run_cfg.height,
                run_cfg.width - run_cfg.panel_width,
                run_cfg.render_pyramid_levels,
                sigma_world=run_cfg.render_particle_sigma_world,
                kernel_size=run_cfg.render_pyramid_kernel_size,
                floor_y=sim_cfg.floor_y,
                floor_enabled=sim_cfg.floor_enabled,
            )
            _draw_pyramid_stack(
                screen,
                levels,
                run_cfg.panel_width,
                run_cfg.width - run_cfg.panel_width,
                run_cfg.height,
            )
        panel.update(dt_ui)
        panel.draw(screen)
        pygame.display.flip()

        frame += 1
        if args.frames and frame >= args.frames:
            done = True

    if args.screenshot:
        os.makedirs(os.path.dirname(args.screenshot) or ".", exist_ok=True)
        pygame.image.save(screen, args.screenshot)

    pygame.quit()


if __name__ == "__main__":
    main()
