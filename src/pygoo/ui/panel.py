from __future__ import annotations

from pathlib import Path

import pygame
import pygame_gui
import torch

from ..config import SimulationConfig


class ControlPanel:
    def __init__(
        self,
        size: tuple[int, int],
        width: int,
        cfg: SimulationConfig,
        pyramid_levels: int = 4,
    ):
        self.width = width
        self.cfg = cfg
        self.new_particle_fixed = False
        self.click_add_mode = True
        self.manager = pygame_gui.UIManager(size)
        self._make_theme()
        self.manager = pygame_gui.UIManager(size, self._theme_path)

        self.run_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(16, 16, width - 32, 34),
            text="Run / Pause",
            manager=self.manager,
        )
        self.reset_all_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(16, 56, (width - 40) // 2, 34),
            text="Reset All",
            manager=self.manager,
        )
        self.reset_masses_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                24 + (width - 40) // 2, 56, (width - 40) // 2, 34
            ),
            text="Clear Particles Only",
            manager=self.manager,
        )
        self.random_scene_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(16, 96, width - 32, 30),
            text="Random Scene",
            manager=self.manager,
        )
        self.export_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(16, 132, width - 32, 30),
            text="Export Scene + Video",
            manager=self.manager,
        )
        self.import_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(16, 168, (width - 40) // 2, 30),
            text="Import Scene",
            manager=self.manager,
        )
        self.import_observable_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                24 + (width - 40) // 2, 168, (width - 40) // 2, 30
            ),
            text="Import Observables Only",
            manager=self.manager,
        )
        self.click_mode_toggle = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(16, 204, width - 32, 30),
            text="Click: Add",
            manager=self.manager,
        )
        self.fixed_toggle = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(16, 240, width - 32, 30),
            text="New Particle: Free",
            manager=self.manager,
        )
        self.render_mode = "Normal"
        self.render_mode_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(16, 276, width - 32, 30),
            text="Render: Normal",
            manager=self.manager,
        )

        # Store default toggle states for color tracking
        self.default_click_add_mode = True
        self.default_new_particle_fixed = False
        self.default_render_mode = "Normal"

        self.optimize_autorun = False

        self.value_specs = {
            "time_step": ("time_step", 0.0001, 0.01),
            "gravity_g": ("gravity_g", -18.0, -0.5),
            "particle_mass": ("particle_mass", 0.08, 12.0),
            "spring_stiffness": ("spring_stiffness", 25.0, 850.0),
            "damping_stiffness": ("damping_stiffness", 0.0, 12.0),
            "max_spring_dist": ("max_spring_dist", 0.02, 1.0),
            "floor_y": ("floor_y", -0.95, 0.2),
            "floor_bounce": ("floor_bounce", 0.05, 1.0),
        }
        self.optimizable_value_order = [
            "time_step",
            "gravity_g",
            "damping_stiffness",
            "floor_bounce",
        ]
        self.value_order = [
            "time_step",
            "gravity_g",
            "particle_mass",
            "spring_stiffness",
            "damping_stiffness",
            "max_spring_dist",
            "floor_y",
            "floor_bounce",
        ]
        self.value_labels = {
            "time_step": "time step",
            "gravity_g": "gravity g",
            "particle_mass": "new mass (fixed)",
            "spring_stiffness": "spring stiffness",
            "damping_stiffness": "damping stiffness",
            "max_spring_dist": "max spring dist (fixed)",
            "floor_y": "floor y (fixed)",
            "floor_bounce": "floor bounce",
        }
        self.labels = {}
        self.sliders = {}
        self.entries = {}
        self.value_locked = {k: True for k in self.optimizable_value_order}
        self.value_lock_buttons = {}
        self.mass_cfg_locked = False
        self.spring_k_locked = False

        label_x = 16
        label_w = 124
        field_gap = 8
        entry_w = 72
        lock_w = 28
        slider_x = label_x + label_w + field_gap
        slider_w = max(
            100,
            width - slider_x - field_gap - entry_w - field_gap - lock_w - label_x,
        )
        entry_x = slider_x + slider_w + field_gap
        lock_x = entry_x + entry_w + field_gap

        y = 322
        for key in self.value_order:
            self.labels[key] = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(label_x, y, label_w, 28),
                text="",
                manager=self.manager,
            )
            _, vmin, vmax = self.value_specs[key]
            self.sliders[key] = pygame_gui.elements.UIHorizontalSlider(
                relative_rect=pygame.Rect(slider_x, y + 2, slider_w, 24),
                start_value=vmin,
                value_range=(vmin, vmax),
                manager=self.manager,
            )
            self.entries[key] = pygame_gui.elements.UITextEntryLine(
                relative_rect=pygame.Rect(entry_x, y, entry_w, 28),
                manager=self.manager,
            )
            if key in self.optimizable_value_order:
                self.value_lock_buttons[key] = pygame_gui.elements.UIButton(
                    relative_rect=pygame.Rect(lock_x, y, lock_w, 28),
                    text="",
                    manager=self.manager,
                )
            y += 34

        self.selected_particle_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(16, y + 6, width - 32, 24),
            text="Selected Particle: none",
            manager=self.manager,
        )
        y += 30
        self.selected_particle_mass_entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(16, y, width - 136, 28),
            manager=self.manager,
        )
        self.mass_cfg_lock_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(width - 112, y, 28, 28),
            text="",
            manager=self.manager,
        )
        self.apply_particle_mass_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(width - 80, y, 64, 28),
            text="Apply",
            manager=self.manager,
        )
        y += 38
        self.selected_spring_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(16, y + 6, width - 32, 24),
            text="Selected Spring: none",
            manager=self.manager,
        )
        y += 30
        self.selected_spring_k_entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(16, y, width - 136, 28),
            manager=self.manager,
        )
        self.spring_k_lock_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(width - 112, y, 28, 28),
            text="",
            manager=self.manager,
        )
        self.apply_spring_k_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(width - 80, y, 64, 28),
            text="Apply",
            manager=self.manager,
        )

        y += 42
        self.load_video_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(16, y, width - 32, 30),
            text="Load Target Video",
            manager=self.manager,
        )
        y += 34
        self.optimize_start_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(16, y, width - 32, 30),
            text="Start Optimization",
            manager=self.manager,
        )

        self._ui_locked = False
        self._lockable_elements = [
            self.run_button,
            self.reset_all_button,
            self.reset_masses_button,
            self.random_scene_button,
            self.export_button,
            self.import_button,
            self.import_observable_button,
            self.click_mode_toggle,
            self.fixed_toggle,
            self.render_mode_button,
            self.selected_particle_mass_entry,
            self.mass_cfg_lock_button,
            self.apply_particle_mass_button,
            self.selected_spring_k_entry,
            self.spring_k_lock_button,
            self.apply_spring_k_button,
            self.load_video_button,
        ]
        self._lockable_elements.extend(self.sliders.values())
        self._lockable_elements.extend(self.entries.values())
        self._lockable_elements.extend(self.value_lock_buttons.values())

        self._selected_particle_idx = -1
        self._selected_spring_idx = -1
        self._selected_particle_mass: float | None = None
        self._selected_spring_k: float | None = None
        self._file_dialog: pygame_gui.windows.UIFileDialog | None = None
        self._file_dialog_mode: str | None = None

        self._sync_buttons()
        self._sync_value_widgets()
        self._update_toggle_colors()

    def selected_pyramid_level(self) -> int | None:
        return None

    def file_dialog_open(self) -> bool:
        return self._file_dialog is not None

    def _update_toggle_colors(self) -> None:
        """Update toggle button colors based on whether they're in default state."""

        def apply_button_color(
            button: pygame_gui.elements.UIButton,
            is_active: bool,
            active_colour: str,
            inactive_colour: str,
        ) -> None:
            """Apply color to button based on active state."""
            colour = pygame.Color(active_colour if is_active else inactive_colour)
            button.colours["dark_bg"] = colour
            button.colours["normal_bg"] = colour
            button.colours["hovered_bg"] = colour.lerp(pygame.Color("white"), 0.12)
            button.colours["active_bg"] = colour.lerp(pygame.Color("black"), 0.12)
            button.rebuild()

        apply_button_color(
            self.click_mode_toggle,
            self.click_add_mode != self.default_click_add_mode,
            "#d97706",
            "#3f9a80",
        )
        apply_button_color(
            self.fixed_toggle,
            self.new_particle_fixed != self.default_new_particle_fixed,
            "#d97706",
            "#3f9a80",
        )
        apply_button_color(
            self.render_mode_button,
            self.render_mode != self.default_render_mode,
            "#d97706",
            "#3f9a80",
        )
        apply_button_color(
            self.mass_cfg_lock_button,
            self.mass_cfg_locked,
            "#8b1d1d",
            "#245b46",
        )
        apply_button_color(
            self.spring_k_lock_button,
            self.spring_k_locked,
            "#8b1d1d",
            "#245b46",
        )
        for key in self.optimizable_value_order:
            apply_button_color(
                self.value_lock_buttons[key],
                self.value_locked[key],
                "#8b1d1d",
                "#245b46",
            )

    def _make_theme(self) -> None:
        self._theme_path = "/tmp/pygoo_theme.json"
        with open(self._theme_path, "w", encoding="utf-8") as f:
            f.write(
                '{"button": {"colours": {"dark_bg": "#2f8f76", "normal_bg": "#3f9a80", "hovered_bg": "#57aa92", "active_bg": "#2a826b", "normal_text": "#e6f2ef"}, "misc": {"shape": "rounded_rectangle", "shape_corner_radius": "3"}}, "panel": {"colours": {"dark_bg": "#2e2e2e"}}, "label": {"colours": {"normal_text": "#d8d8d8"}}}'
            )

    def _fmt(self, v: float) -> str:
        s = f"{v:.5f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s

    def _parse_float(self, text: str) -> float | None:
        try:
            return float(text.strip())
        except ValueError:
            return None

    def _sync_buttons(self) -> None:
        def lock_text(locked: bool) -> str:
            return "🔒" if locked else "🔓"

        self.optimize_start_button.set_text(
            "Stop Optimization" if self.optimize_autorun else "Start Optimization"
        )
        self.click_mode_toggle.set_text(
            "Click: Add" if self.click_add_mode else "Click: Select"
        )
        self.render_mode_button.set_text(f"Render: {self.render_mode}")
        self.fixed_toggle.set_text(
            f"New Particle: {'Fixed' if self.new_particle_fixed else 'Free'}"
        )
        self.mass_cfg_lock_button.set_text(lock_text(self.mass_cfg_locked))
        self.spring_k_lock_button.set_text(lock_text(self.spring_k_locked))
        for key in self.optimizable_value_order:
            self.value_lock_buttons[key].set_text(lock_text(self.value_locked[key]))
        self._update_toggle_colors()

    def _set_ui_enabled(self, enabled: bool) -> None:
        if self._ui_locked == (not enabled):
            return
        self._ui_locked = not enabled
        for element in self._lockable_elements:
            if enabled:
                element.enable()
            else:
                element.disable()

    def set_optimization_running(self, running: bool) -> None:
        if running and self._file_dialog is not None:
            self._file_dialog.kill()
            self._file_dialog = None
            self._file_dialog_mode = None
        self.optimize_autorun = running
        self._set_ui_enabled(not running)
        self._sync_buttons()

    def _set_cfg_value(self, key: str, value: float, sync_widgets: bool = True) -> None:
        attr, vmin, vmax = self.value_specs[key]
        value = float(value)
        v = min(max(value, vmin), vmax)
        self.cfg.set_value(attr, v)
        if sync_widgets:
            self.sliders[key].set_current_value(v)
            self.entries[key].set_text(self._fmt(v))
        self.labels[key].set_text(self.value_labels[key])

    def _sync_value_widgets(self) -> None:
        for key in self.value_order:
            attr, _, _ = self.value_specs[key]
            val = getattr(self.cfg, attr)
            self._set_cfg_value(key, float(val), sync_widgets=True)

    def sync_config_values(self) -> None:
        for key in self.value_order:
            attr, _, _ = self.value_specs[key]
            with torch.no_grad():
                val = float(getattr(self.cfg, attr))
            self.sliders[key].set_current_value(val)
            if not getattr(self.entries[key], "is_focused", False):
                self.entries[key].set_text(self._fmt(val))
            self.labels[key].set_text(self.value_labels[key])

    def refresh_from_config(self) -> None:
        self._sync_buttons()
        self._sync_value_widgets()

    def _open_file_dialog(self, mode: str) -> None:
        if self._file_dialog is not None:
            self._file_dialog.kill()
            self._file_dialog = None
            self._file_dialog_mode = None
        title = "Export Scene As"
        if mode == "import":
            title = "Import Scene"
        elif mode == "import_observable":
            title = "Import Observable JSON"
        elif mode == "video":
            title = "Load Target Video"
        default_path = "."
        if mode in {"import", "import_observable", "video", "export"}:
            data_dir = Path("data")
            if data_dir.exists():
                default_path = str(data_dir.resolve())

        self._file_dialog = pygame_gui.windows.UIFileDialog(
            rect=pygame.Rect(64, 64, 760, 560),
            manager=self.manager,
            window_title=title,
            initial_file_path=default_path,
            allow_existing_files_only=(mode in {"import", "video"}),
            allowed_suffixes=(
                {".avi", ".mov", ".m4v", ".avi", ".webm"}
                if mode == "video"
                else {".json"}
            ),
        )
        self._file_dialog_mode = mode

    def _parse_int(self, text: str) -> int | None:
        try:
            return int(text.strip())
        except ValueError:
            return None

    def optimization_settings(self) -> dict[str, object]:
        optimize_cfg = {k: not v for k, v in self.value_locked.items()}

        return {
            "train_mass": not self.mass_cfg_locked,
            "train_edge_k": not self.spring_k_locked,
            "optimize_cfg": optimize_cfg,
        }

    def set_selection_info(
        self,
        particle_idx: int,
        spring_idx: int,
        particle_mass: float | None,
        spring_k: float | None,
    ) -> None:
        mass_changed = (
            particle_mass is not None
            and self._selected_particle_mass is not None
            and abs(particle_mass - self._selected_particle_mass) > 1e-6
        )
        if particle_idx != self._selected_particle_idx or mass_changed:
            self._selected_particle_idx = particle_idx
            self._selected_particle_mass = particle_mass
            if particle_idx >= 0 and particle_mass is not None:
                self.selected_particle_label.set_text(
                    f"Selected Particle: {particle_idx}"
                )
                if not getattr(self.selected_particle_mass_entry, "is_focused", False):
                    self.selected_particle_mass_entry.set_text(self._fmt(particle_mass))
            else:
                self.selected_particle_label.set_text("Selected Particle: none")
                if not getattr(self.selected_particle_mass_entry, "is_focused", False):
                    self.selected_particle_mass_entry.set_text("")

        spring_changed = (
            spring_k is not None
            and self._selected_spring_k is not None
            and abs(spring_k - self._selected_spring_k) > 1e-6
        )
        if spring_idx != self._selected_spring_idx or spring_changed:
            self._selected_spring_idx = spring_idx
            self._selected_spring_k = spring_k
            if spring_idx >= 0 and spring_k is not None:
                self.selected_spring_label.set_text(f"Selected Spring: {spring_idx}")
                if not getattr(self.selected_spring_k_entry, "is_focused", False):
                    self.selected_spring_k_entry.set_text(self._fmt(spring_k))
            else:
                self.selected_spring_label.set_text("Selected Spring: none")
                if not getattr(self.selected_spring_k_entry, "is_focused", False):
                    self.selected_spring_k_entry.set_text("")

    def process(
        self, event: pygame.event.Event, running: bool
    ) -> tuple[bool, bool, dict[str, float | bool | str | None]]:
        reset = False
        actions: dict[str, float | bool | str | None] = {
            "reset_all": False,
            "reset_masses": False,
            "apply_particle_mass": None,
            "apply_spring_k": None,
            "random_scene": False,
            "export_path": None,
            "import_path": None,
            "import_observable_path": None,
            "target_video_path": None,
            "optimize_start": False,
            "reset_optimizer": False,
            "demo_mass_only": False,
        }

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.run_button:
                running = not running
            elif event.ui_element == self.reset_all_button:
                reset = True
                actions["reset_all"] = True
                running = False
            elif event.ui_element == self.reset_masses_button:
                actions["reset_masses"] = True
                running = False
            elif event.ui_element == self.random_scene_button:
                actions["random_scene"] = True
                running = False
            elif event.ui_element == self.export_button:
                self._open_file_dialog("export")
                running = False
            elif event.ui_element == self.import_button:
                self._open_file_dialog("import")
                running = False
            elif event.ui_element == self.import_observable_button:
                self._open_file_dialog("import_observable")
                running = False
            elif event.ui_element == self.load_video_button:
                self._open_file_dialog("video")
                running = False
            elif event.ui_element == self.optimize_start_button:
                if self.optimize_autorun:
                    # Stop
                    self.set_optimization_running(False)
                else:
                    # Start
                    self.optimize_autorun = True
                    actions["optimize_start"] = True
            elif event.ui_element == self.click_mode_toggle:
                self.click_add_mode = not self.click_add_mode
            elif event.ui_element == self.render_mode_button:
                self.render_mode = (
                    "Pyramid" if self.render_mode == "Normal" else "Normal"
                )
            elif event.ui_element == self.fixed_toggle:
                self.new_particle_fixed = not self.new_particle_fixed
            elif event.ui_element == self.apply_particle_mass_button:
                parsed = self._parse_float(self.selected_particle_mass_entry.get_text())
                if parsed is not None:
                    actions["apply_particle_mass"] = max(parsed, 1e-6)
            elif event.ui_element == self.mass_cfg_lock_button:
                self.mass_cfg_locked = not self.mass_cfg_locked
                actions["reset_optimizer"] = True
            elif event.ui_element == self.spring_k_lock_button:
                self.spring_k_locked = not self.spring_k_locked
                actions["reset_optimizer"] = True
            elif event.ui_element == self.apply_spring_k_button:
                parsed = self._parse_float(self.selected_spring_k_entry.get_text())
                if parsed is not None:
                    actions["apply_spring_k"] = max(parsed, 0.0)
            else:
                handled_lock = False
                for key in self.optimizable_value_order:
                    if event.ui_element == self.value_lock_buttons[key]:
                        self.value_locked[key] = not self.value_locked[key]
                        handled_lock = True
                        break
                if not handled_lock:
                    self._sync_value_widgets()
            self._sync_buttons()

        elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            for key in self.value_order:
                if event.ui_element == self.sliders[key]:
                    self._set_cfg_value(key, float(event.value), sync_widgets=True)
                    break

        elif event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
            for key in self.value_order:
                if event.ui_element == self.entries[key]:
                    parsed = self._parse_float(self.entries[key].get_text())
                    if parsed is not None:
                        self._set_cfg_value(key, parsed, sync_widgets=True)
                    else:
                        self._set_cfg_value(
                            key, getattr(self.cfg, self.value_specs[key][0])
                        )
                    break

        elif event.type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
            if self._file_dialog_mode == "import":
                actions["import_path"] = event.text
            elif self._file_dialog_mode == "import_observable":
                actions["import_observable_path"] = event.text
            elif self._file_dialog_mode == "video":
                actions["target_video_path"] = event.text
            elif self._file_dialog_mode == "export":
                path = event.text
                if not path.lower().endswith(".json"):
                    path = f"{path}.json"
                actions["export_path"] = path
            if self._file_dialog is not None:
                self._file_dialog.kill()
            self._file_dialog = None
            self._file_dialog_mode = None

        elif event.type == pygame_gui.UI_WINDOW_CLOSE:
            if self._file_dialog is not None and event.ui_element == self._file_dialog:
                self._file_dialog = None
                self._file_dialog_mode = None

        self.manager.process_events(event)
        return running, reset, actions

    def update(self, dt: float) -> None:
        self.manager.update(dt)

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(
            surface, (65, 65, 65), pygame.Rect(0, 0, self.width, surface.get_height())
        )
        self.manager.draw_ui(surface)
