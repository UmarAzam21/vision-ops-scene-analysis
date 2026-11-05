"""Crowd-density heatmap from person positions (Gaussian kernel density)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import cv2
import numpy as np

DensityLevel = Literal["NONE", "LOW", "MEDIUM", "HIGH"]


class CrowdDensityEstimator:
    """Builds a smoothed density grid of people per neighbourhood.

    The kernel size follows the apparent person size (median box height), so a cell
    value is roughly "people within ~1.5 body-heights". That makes LOW / MEDIUM / HIGH
    mean the same thing for a close-up CCTV view and a high-altitude drone view.
    """

    def __init__(
        self,
        *,
        grid_width: int = 160,
        sigma_ratio: float = 0.035,  # fallback when person size is unknown
        sigma_body_heights: float = 1.5,
        smoothing: float = 0.35,
        medium_threshold: float = 3.0,
        high_threshold: float = 7.0,
    ) -> None:
        self.grid_width = grid_width
        self.sigma_ratio = sigma_ratio
        self.sigma_body_heights = sigma_body_heights
        self.smoothing = smoothing
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self._grid: np.ndarray | None = None

    def update(
        self,
        points: Iterable[tuple[int, int]],
        frame_shape: tuple[int, ...],
        person_height: float | None = None,
    ) -> np.ndarray:
        h, w = frame_shape[:2]
        gw = self.grid_width
        gh = max(1, round(gw * h / w))
        grid = np.zeros((gh, gw), dtype=np.float32)
        for x, y in points:
            gx = min(gw - 1, max(0, int(x / w * gw)))
            gy = min(gh - 1, max(0, int(y / h * gh)))
            grid[gy, gx] += 1.0

        if person_height:
            sigma = person_height * self.sigma_body_heights / w * gw
        else:
            sigma = gw * self.sigma_ratio
        sigma = float(np.clip(sigma, 1.0, gw * 0.12))
        grid = cv2.GaussianBlur(grid, (0, 0), sigma)
        # Rescale so a cell holds "people within the kernel" rather than a pdf value.
        grid *= 2.0 * np.pi * sigma * sigma

        if self._grid is not None and self._grid.shape == grid.shape and self.smoothing > 0:
            grid = cv2.addWeighted(self._grid, self.smoothing, grid, 1 - self.smoothing, 0)
        self._grid = grid
        return grid

    def level(self, grid: np.ndarray | None) -> DensityLevel:
        if grid is None:
            return "NONE"
        peak = float(grid.max()) if grid.size else 0.0
        if peak < 0.5:
            return "NONE"
        if peak < self.medium_threshold:
            return "LOW"
        if peak < self.high_threshold:
            return "MEDIUM"
        return "HIGH"

    def normalize(self, grid: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Float32 0..1 intensity of `size` (w, h). Normalised against the HIGH level
        so a sparse scene stays cool instead of being stretched to red."""
        scale = max(float(grid.max()), self.high_threshold * 1.2)
        norm = np.clip(grid / scale, 0, 1).astype(np.float32)
        return cv2.resize(norm, size, interpolation=cv2.INTER_CUBIC).clip(0, 1)

    def colorize(self, grid: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Return a BGR JET heatmap of `size` (w, h)."""
        norm = (self.normalize(grid, size) * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)
