"""Visualization helpers for ERA5 pointwise prediction comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_snapshot_indices(raw: str | None, num_times: int) -> list[int]:
    """Parse a comma-separated list of time indices.

    Negative indices follow Python semantics, e.g. `-1` means the last time
    step. An empty string returns an empty list so callers can fall back to
    defaults.
    """

    if raw is None or not raw.strip():
        return []

    parsed: list[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        value = int(token)
        if value < 0:
            value = num_times + value
        if value < 0 or value >= num_times:
            raise ValueError(f"Snapshot index {token} is out of bounds for {num_times} time steps.")
        parsed.append(value)
    return parsed


def choose_snapshot_indices(
    num_times: int,
    requested: Iterable[int] | None = None,
    max_snapshots: int = 3,
) -> list[int]:
    """Choose representative time indices for visualization."""

    if num_times <= 0:
        raise ValueError("num_times must be positive.")
    if max_snapshots <= 0:
        raise ValueError("max_snapshots must be positive.")

    requested_list = list(requested or [])
    if requested_list:
        deduped: list[int] = []
        for value in requested_list:
            if value not in deduped:
                deduped.append(value)
        return deduped[:max_snapshots]

    if num_times == 1:
        return [0]

    anchors = [0, num_times // 2, num_times - 1]
    deduped = []
    for value in anchors:
        if value not in deduped:
            deduped.append(value)
    return deduped[:max_snapshots]


def save_era5_prediction_maps(
    *,
    spatial_coords: torch.Tensor,
    times: torch.Tensor,
    observations: torch.Tensor,
    predicted_mean: torch.Tensor,
    output_dir: str | Path,
    filename_prefix: str,
    variable_name: str,
    snapshot_indices: Iterable[int],
    point_size: float = 18.0,
) -> list[Path]:
    """Save ground-truth / prediction / error map triptychs for selected times.

    This helper assumes pointwise ERA5 inputs with spatial coordinates stored as
    `(lon, lat)`. The result is a scatter-based map, which works both for
    regular grids and irregular subsets of locations.
    """

    spatial = torch.as_tensor(spatial_coords).detach().cpu().numpy()
    time_values = torch.as_tensor(times).detach().cpu().numpy().reshape(-1)
    y_true = torch.as_tensor(observations).detach().cpu().numpy()
    y_pred = torch.as_tensor(predicted_mean).detach().cpu().numpy()

    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise ValueError("Expected spatial_coords with shape [N_s, 2] storing (lon, lat).")
    if y_true.shape != y_pred.shape:
        raise ValueError("observations and predicted_mean must have the same shape.")
    if y_true.shape[0] != time_values.shape[0]:
        raise ValueError("times length must match the first dimension of observations.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    lon = spatial[:, 0]
    lat = spatial[:, 1]
    saved_files: list[Path] = []
    for time_idx in snapshot_indices:
        if time_idx < 0 or time_idx >= y_true.shape[0]:
            raise ValueError(f"Snapshot index {time_idx} is out of bounds.")

        truth = y_true[time_idx]
        pred = y_pred[time_idx]
        err = pred - truth

        field_min = float(min(np.min(truth), np.min(pred)))
        field_max = float(max(np.max(truth), np.max(pred)))
        err_lim = float(np.max(np.abs(err)))
        if err_lim <= 0:
            err_lim = 1.0

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
        titles = ["Ground Truth", "Prediction", "Error"]
        arrays = [truth, pred, err]
        cmaps = ["coolwarm", "coolwarm", "RdBu_r"]
        limits = [
            (field_min, field_max),
            (field_min, field_max),
            (-err_lim, err_lim),
        ]

        for ax, title, values, cmap, (vmin, vmax) in zip(axes, titles, arrays, cmaps, limits):
            scatter = ax.scatter(
                lon,
                lat,
                c=values,
                cmap=cmap,
                s=point_size,
                vmin=vmin,
                vmax=vmax,
                edgecolors="none",
            )
            ax.set_title(title)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.15, linewidth=0.5)
            fig.colorbar(scatter, ax=ax, shrink=0.88)

        fig.suptitle(
            f"{variable_name} | t_index={time_idx} | time={time_values[time_idx]:.3f}",
            fontsize=14,
        )
        file_path = output_path / f"{filename_prefix}_t{time_idx:03d}.png"
        fig.savefig(file_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(file_path)

    return saved_files



def save_era5_field_maps(
    *,
    spatial_coords: torch.Tensor,
    times: torch.Tensor,
    values: torch.Tensor,
    output_dir: str | Path,
    filename_prefix: str,
    variable_name: str,
    snapshot_indices: Iterable[int],
    point_size: float = 18.0,
    title: str = "Field",
    cmap: str = "coolwarm",
) -> list[Path]:
    """Save single-field scatter maps for selected times."""

    spatial = torch.as_tensor(spatial_coords).detach().cpu().numpy()
    time_values = torch.as_tensor(times).detach().cpu().numpy().reshape(-1)
    field = torch.as_tensor(values).detach().cpu().numpy()

    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise ValueError("Expected spatial_coords with shape [N_s, 2] storing (lon, lat).")
    if field.shape[0] != time_values.shape[0]:
        raise ValueError("times length must match the first dimension of values.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    lon = spatial[:, 0]
    lat = spatial[:, 1]
    field_min = float(np.min(field))
    field_max = float(np.max(field))
    saved_files: list[Path] = []

    for time_idx in snapshot_indices:
        if time_idx < 0 or time_idx >= field.shape[0]:
            raise ValueError(f"Snapshot index {time_idx} is out of bounds.")

        fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.8), constrained_layout=True)
        scatter = ax.scatter(
            lon,
            lat,
            c=field[time_idx],
            cmap=cmap,
            s=point_size,
            vmin=field_min,
            vmax=field_max,
            edgecolors="none",
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15, linewidth=0.5)
        fig.colorbar(scatter, ax=ax, shrink=0.88)
        fig.suptitle(
            f"{variable_name} | t_index={time_idx} | time={time_values[time_idx]:.3f}",
            fontsize=14,
        )
        file_path = output_path / f"{filename_prefix}_t{time_idx:03d}.png"
        fig.savefig(file_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(file_path)

    return saved_files
