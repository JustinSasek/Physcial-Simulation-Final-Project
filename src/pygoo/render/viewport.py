from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Viewport:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    screen_x: int
    screen_y: int
    screen_w: int
    screen_h: int

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        nx = (x - self.x_min) / (self.x_max - self.x_min)
        ny = (y - self.y_min) / (self.y_max - self.y_min)
        sx = self.screen_x + int(nx * self.screen_w)
        sy = self.screen_y + int((1.0 - ny) * self.screen_h)
        return sx, sy

    def screen_to_world(self, sx: int, sy: int) -> tuple[float, float]:
        nx = (sx - self.screen_x) / max(self.screen_w, 1)
        ny = 1.0 - (sy - self.screen_y) / max(self.screen_h, 1)
        x = self.x_min + nx * (self.x_max - self.x_min)
        y = self.y_min + ny * (self.y_max - self.y_min)
        return x, y

    def scale(self, zoom: float = 1.30) -> Viewport:
        cx = 0.5 * (self.x_min + self.x_max)
        cy = 0.5 * (self.y_min + self.y_max)
        half_w = 0.5 * (self.x_max - self.x_min) * zoom
        half_h = 0.5 * (self.y_max - self.y_min) * zoom
        return Viewport(
            x_min=cx - half_w,
            x_max=cx + half_w,
            y_min=cy - half_h,
            y_max=cy + half_h,
            screen_x=self.screen_x,
            screen_y=self.screen_y,
            screen_w=self.screen_w,
            screen_h=self.screen_h,
        )
