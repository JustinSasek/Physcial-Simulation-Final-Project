from __future__ import annotations

from dataclasses import dataclass

import imageio.v2 as imageio  # type: ignore[import-untyped]
import torch
import torch.nn.functional as F

from .config import RuntimeConfig, SimulationConfig
from .physics import Stepper
from .render import Viewport, build_gaussian_pyramid, render_gaussian_pyramid
from .state import SimulationState

START_HORIZON = 5
STRIDE = 2


@dataclass
class FitStepResult:
    ok: bool
    loss: float
    horizon: int
    spatial_stage: int
    max_horizon: int
    max_spatial_stage: int
    mass_grad_norm: float = 0.0
    edge_k_grad_norm: float = 0.0
    mass_param_delta: float = 0.0
    edge_k_param_delta: float = 0.0
    cfg_param_name: str = ""
    cfg_param_delta: float = 0.0
    advance_threshold: float = 0.0
    loss_threshold: float = 0.0
    eligible_to_advance: bool = False
    schedule_advanced: bool = False
    schedule_advance_target: str = ""
    schedule_advance_reason: str = ""
    loss_improved: bool = False
    no_improve_steps: int = 0
    no_improve_patience: int = 0
    message: str = ""


@dataclass
class GradientSnapshot:
    loss: float
    mass_grad_norm: float
    edge_k_grad_norm: float


class DifferentiableFitter:
    _CFG_LR_MULT = {
        "time_step": 0.0005,
        "gravity_g": 3.0,
        "damping_stiffness": 20.0,
        "floor_bounce": 0.5,
        "masses": 1.0,
        "edge_k": 500.0,
    }
    _CFG_RANGES = {
        "time_step": (1e-5, None),
        "gravity_g": (None, -0.1),
        "damping_stiffness": (1e-4, None),
        "floor_bounce": (1e-4, None),
    }

    def __init__(
        self,
        run_cfg: RuntimeConfig,
        num_levels: int,
        sigma_world: float,
        kernel_size: int,
    ):
        self.run_cfg = run_cfg
        self.num_levels = max(1, int(num_levels))
        self.sigma_world = float(sigma_world)
        self.kernel_size = int(kernel_size)

        self.target_video_path = ""
        self._target_pyramids: list[list[torch.Tensor]] = []
        self._target_device = ""

        self.lr = 0.02
        self.loss_threshold = 0.1
        self.temporal_discount = 0.97
        self.horizon_step = 4
        self.max_horizon = 120
        self.auto_advance_no_improve = True
        self.no_improve_patience = 8
        self.loss_improvement_epsilon = 1e-6

        self.train_mass = True
        self.train_edge_k = True
        self.cfg_step_scale = 0.05

        self._horizon = START_HORIZON
        self._spatial_stage = 1
        self._best_loss_since_advance = float("inf")
        self._no_improve_steps = 0

        self._optimizer: torch.optim.Optimizer | None = None

        self._stepper = Stepper(SimulationConfig(), use_compile=False)

    @property
    def has_target(self) -> bool:
        return bool(self._target_pyramids)

    @property
    def target_frame_count(self) -> int:
        return len(self._target_pyramids)

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def spatial_stage(self) -> int:
        return self._spatial_stage

    def configure(
        self,
        *,
        train_mass: bool,
        train_edge_k: bool,
        lr: float,
        loss_threshold: float | None = None,
        min_advance_loss: float | None = None,
        temporal_discount: float,
        horizon_step: int,
        max_horizon: int,
        optimize_cfg: dict[str, bool] | None = None,
        cfg_step_scale: float = 0.05,
        auto_advance_no_improve: bool = True,
        no_improve_patience: int = 8,
        loss_improvement_epsilon: float = 1e-6,
    ) -> None:
        self.train_mass = bool(train_mass)
        self.train_edge_k = bool(train_edge_k)
        self.lr = max(float(lr), 1e-6)
        raw_loss_threshold = (
            min_advance_loss if loss_threshold is None else loss_threshold
        )
        if raw_loss_threshold is not None:
            self.loss_threshold = max(float(raw_loss_threshold), 0.0)
        self.temporal_discount = min(max(float(temporal_discount), 0.0), 1.0)
        self.horizon_step = max(int(horizon_step), 1)
        self.max_horizon = max(
            int(max_horizon),
            5 * max(1, int(self.run_cfg.substeps)),
        )
        self.optimize_cfg = optimize_cfg
        self.cfg_step_scale = max(float(cfg_step_scale), 1e-4)
        self.auto_advance_no_improve = bool(auto_advance_no_improve)
        self.no_improve_patience = max(int(no_improve_patience), 1)
        self.loss_improvement_epsilon = max(float(loss_improvement_epsilon), 0.0)

    def reset_optimizer(self) -> None:
        self._optimizer = None

    def build_optimizer(
        self, state: SimulationState, config: SimulationConfig
    ) -> tuple[bool, str]:
        self.reset_optimizer()
        return self._build_optimizer(state, config)

    def reset_curriculum(self) -> None:
        self._horizon = max(START_HORIZON, self.horizon_step)
        self._spatial_stage = 1
        self._best_loss_since_advance = float("inf")
        self._no_improve_steps = 0

    def load_target_video(
        self,
        path: str,
        device: str,
        scene_width: int,
        scene_height: int,
    ) -> int:
        pyramids: list[list[torch.Tensor]] = []
        d = torch.device(device)
        stride = STRIDE

        reader = imageio.get_reader(path)
        try:
            for i, frame in enumerate(reader.iter_data()):
                if i % stride != 0:
                    continue

                x = torch.tensor(frame, dtype=torch.float32, device=d) / 255.0
                if x.ndim == 2:
                    x = x[..., None].repeat(1, 1, 3)
                if x.shape[-1] > 3:
                    x = x[..., :3]
                if x.shape[-1] == 1:
                    x = x.repeat(1, 1, 3)
                if x.shape[0] != scene_height or x.shape[1] != scene_width:
                    x = F.interpolate(
                        x.permute(2, 0, 1).unsqueeze(0),
                        size=(scene_height, scene_width),
                        mode="bilinear",
                        align_corners=False,
                    )[0].permute(1, 2, 0)
                pyramids.append(
                    build_gaussian_pyramid(
                        x.contiguous(),
                        num_levels=self.num_levels,
                        kernel_size=self.kernel_size,
                    )
                )
        finally:
            reader.close()

        if not pyramids:
            raise ValueError("Video has no frames")

        self.target_video_path = path
        self._target_pyramids = pyramids
        self._target_device = str(d)
        self.reset_optimizer()
        self.reset_curriculum()
        return len(pyramids)

    def _build_optimizer(
        self, state: SimulationState, config: SimulationConfig
    ) -> tuple[bool, str]:
        param_groups = []
        opt_cfg = self.optimize_cfg if self.optimize_cfg is not None else {}

        # Groups for mass and edge_k. Use configured LR multipliers when available
        edge_params: list[torch.Tensor] = []
        mass_params: list[torch.Tensor] = []
        if self.train_edge_k and state.edge_k.numel() > 0:
            state.edge_k.requires_grad_(True)
            edge_params.append(state.edge_k)

        if self.train_mass and state.mass.numel() > 0:
            state.mass.requires_grad_(True)
            mass_params.append(state.mass)

        # Apply per-group learning rate multipliers from _CFG_LR_MULT when present
        if edge_params:
            edge_lr = self._CFG_LR_MULT.get("edge_k", 1.0) * self.lr
            param_groups.append({"params": edge_params, "lr": edge_lr})
        if mass_params:
            mass_lr = self._CFG_LR_MULT.get("masses", 1.0) * self.lr
            param_groups.append({"params": mass_params, "lr": mass_lr})

        # Ensure config tensors are on the same device as the state so the
        # computation graph connects optimizer params to the simulation.
        try:
            config.to_device(state.device)
        except Exception:
            pass

        # Param groups for config parameters with their specific learning rates
        for k, do_train in opt_cfg.items():
            if k in self._CFG_LR_MULT:
                param = getattr(config, k, None)
                if isinstance(param, torch.Tensor) and param.numel() == 1:
                    if do_train:
                        if param.device != state.device:
                            param = param.detach().to(state.device)
                        param.requires_grad_(True)
                        param_groups.append(
                            {
                                "params": [param],
                                "lr": self._CFG_LR_MULT[k] * self.lr,
                            }
                        )
                    else:
                        param.requires_grad_(False)

        if not param_groups:
            self._optimizer = None
            return False, "No trainable variables selected"

        self._optimizer = torch.optim.Adam(param_groups)
        return True, ""

    def rollout_trajectory(
        self,
        base_state: SimulationState,
        sim_cfg: SimulationConfig,
        substeps: int,
        frames: int,
    ) -> torch.Tensor:
        if frames <= 0:
            return torch.empty(
                (0, 0, 2), dtype=base_state.pos.dtype, device=base_state.device
            )

        self._stepper.cfg = sim_cfg

        traj: list[torch.Tensor] = []
        for t in range(frames):
            traj.append(base_state.pos)
            if t + 1 < frames:
                self._stepper.rollout(base_state, substeps)
        return torch.stack(traj, dim=0)

    def _curriculum_level_indices(self) -> list[int]:
        start = max(0, self.num_levels - self._spatial_stage)
        return list(range(self.num_levels - 1, start - 1, -1))

    def _loss(
        self,
        base_state: SimulationState,
        sim_cfg: SimulationConfig,
        vp: Viewport,
        substeps: int,
    ) -> torch.Tensor:

        frame_count = len(self._target_pyramids)
        horizon = min(self._horizon, frame_count)
        level_idx = self._curriculum_level_indices()

        sim_state = SimulationState(
            pos=base_state.pos.detach(),
            vel=base_state.vel.detach(),
            mass=base_state.mass,
            fixed=base_state.fixed,
            edges=base_state.edges,
            rest_len=base_state.rest_len,
            edge_k=base_state.edge_k,
        )
        traj = self.rollout_trajectory(sim_state, sim_cfg, substeps * STRIDE, horizon)
        render_state = SimulationState(
            pos=traj[0],
            vel=sim_state.vel,
            mass=sim_state.mass,
            fixed=sim_state.fixed,
            edges=sim_state.edges,
            rest_len=sim_state.rest_len,
            edge_k=sim_state.edge_k,
        )

        total = torch.tensor(0.0, dtype=base_state.pos.dtype, device=base_state.device)
        count = 0

        # collect frames
        pred_frames = []
        target_frames = []
        diff_frames = []

        for t in range(horizon):
            render_state.pos = traj[t]
            pred = render_gaussian_pyramid(
                render_state,
                vp,
                vp.screen_h,
                vp.screen_w,
                num_levels=self.num_levels,
                sigma_world=self.sigma_world,
                kernel_size=self.kernel_size,
                floor_y=sim_cfg.floor_y,
                floor_enabled=sim_cfg.floor_enabled,
            )
            target = self._target_pyramids[t]

            for level in level_idx:
                total += F.mse_loss(pred[level], target[level])
                count += 1

            p = pred[0].detach().cpu().numpy()
            tgt = target[0].detach().cpu().numpy()
            pred_frames.append(p)
            target_frames.append(tgt)
            diff_frames.append(((p - tgt + 1.0) / 2.0).clip(0, 1))

        import matplotlib.pyplot as plt

        n_display = 5
        display_idx = [
            int(i * (horizon - 1) / (n_display - 1)) for i in range(n_display)
        ]
        fig, axes = plt.subplots(3, n_display, figsize=(4 * n_display, 10))
        labels = "pred target diff".split()
        for col, t in enumerate(display_idx):
            for row, img in enumerate(
                [pred_frames[t], target_frames[t], diff_frames[t]]
            ):
                axes[row][col].imshow(img.clip(0, 1))
                axes[row][col].set_title(f"{labels[row]} t={t}")
                axes[row][col].axis("off")
        plt.tight_layout()
        plt.savefig("debug_all_frames.png", dpi=100)
        plt.close(fig)

        return total / count

    def step(
        self,
        state: SimulationState,
        sim_cfg: SimulationConfig,
        vp: Viewport,
        substeps: int,
    ) -> FitStepResult:
        # Scale viewport by 1.3 to match _export_preview_video zoom_out factor
        vp = vp.scale(1.3)

        def _fail(message: str) -> FitStepResult:
            return FitStepResult(
                ok=False,
                loss=0.0,
                horizon=self._horizon,
                spatial_stage=self._spatial_stage,
                max_horizon=len(self._target_pyramids),
                max_spatial_stage=self.num_levels,
                message=message,
            )

        if not self.has_target:
            return _fail("No target video loaded")

        if self._target_device != str(state.device):
            return _fail("Target device mismatch")

        if self._optimizer is None:
            return _fail("Optimizer not initialized")

        loss = self._loss(state, sim_cfg, vp, substeps)

        self._optimizer.zero_grad()
        loss.backward()

        mass_grad_norm = 0.0
        edge_k_grad_norm = 0.0
        if state.edge_k.numel() > 0 and state.edge_k.grad is not None:
            edge_k_grad_norm = float(torch.linalg.norm(state.edge_k.grad).item())

        if state.mass.numel() > 0 and state.mass.grad is not None:
            mass_grad_norm = float(torch.linalg.norm(state.mass.grad).item())

        # gradient clipping
        for param in self._optimizer.param_groups[0]["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-1.0, 1.0)

        self._optimizer.step()

        # Clamp masses and spring stiffness and cfg parameters to above zero
        # fine if out of reasonable range
        with torch.no_grad():
            if state.edge_k.numel() > 0:
                state.edge_k.clamp_(min=1e-4)

            if state.mass.numel() > 0:
                state.mass.clamp_(min=0.08)

            opt_cfg = self.optimize_cfg if self.optimize_cfg is not None else {}
            for k, v in opt_cfg.items():
                if v and k in self._CFG_RANGES:
                    param = getattr(self._stepper.cfg, k, None)
                    if isinstance(param, torch.Tensor) and param.numel() == 1:
                        min_val, max_val = self._CFG_RANGES[k]
                        if min_val is not None and max_val is not None:
                            param.clamp_(min=min_val, max=max_val)
                        elif min_val is not None:
                            param.clamp_(min=min_val)
                        elif max_val is not None:
                            param.clamp_(max=max_val)

        loss_value = float(loss.detach().item())
        loss_improved = (
            self._best_loss_since_advance - loss_value
        ) > self.loss_improvement_epsilon
        if loss_improved:
            self._best_loss_since_advance = loss_value
            self._no_improve_steps = 0
        else:
            if self._best_loss_since_advance == float("inf"):
                self._best_loss_since_advance = loss_value
            self._no_improve_steps += 1

        prev_horizon = self._horizon
        prev_spatial_stage = self._spatial_stage
        eligible_by_threshold = loss_value <= self.loss_threshold
        eligible_by_no_improve = self.auto_advance_no_improve and (
            self._no_improve_steps >= self.no_improve_patience
        )
        eligible_to_advance = eligible_by_threshold or eligible_by_no_improve
        advance_reason = "none"
        if eligible_by_threshold:
            advance_reason = "threshold"
        elif eligible_by_no_improve:
            advance_reason = "no-improve"

        schedule_advance_target = "none"
        max_horizon_frames = len(self._target_pyramids)
        if eligible_to_advance:
            if self._horizon < max_horizon_frames:
                self._horizon = min(
                    self._horizon + self.horizon_step,
                    max_horizon_frames,
                    len(self._target_pyramids),
                )
                schedule_advance_target = "horizon"
            elif self._spatial_stage < self.num_levels:
                self._spatial_stage += 1
                schedule_advance_target = "spatial"

        schedule_advanced = (self._horizon != prev_horizon) or (
            self._spatial_stage != prev_spatial_stage
        )
        if schedule_advanced:
            self._best_loss_since_advance = float("inf")
            self._no_improve_steps = 0

        return FitStepResult(
            ok=True,
            loss=loss_value,
            horizon=self._horizon,
            spatial_stage=self._spatial_stage,
            max_horizon=len(self._target_pyramids),
            max_spatial_stage=self.num_levels,
            mass_grad_norm=mass_grad_norm,
            edge_k_grad_norm=edge_k_grad_norm,
            advance_threshold=self.loss_threshold,
            loss_threshold=self.loss_threshold,
            eligible_to_advance=eligible_to_advance,
            schedule_advanced=schedule_advanced,
            schedule_advance_target=schedule_advance_target,
            schedule_advance_reason=advance_reason,
            loss_improved=loss_improved,
            no_improve_steps=self._no_improve_steps,
            no_improve_patience=self.no_improve_patience,
            message="",
        )
