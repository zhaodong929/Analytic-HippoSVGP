"""ERA5 loading helpers for early spatio-temporal experiments.

Processed ERA5 tasks are aggregated into a single-variable spatio-temporal
field with:
- time handled by the analytic temporal module
- space handled by a standard kernel/SVGP-style inducing-point layer
- spatial coordinates stored in `(lon, lat)` order
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Optional

import numpy as np
import torch

from .temporal_analytic import TemporalBlockSpec


ERA5_LAND_VARIABLES = [
    "2m_dewpoint_temperature",
    "2m_temperature",
    "skin_temperature",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "skin_reservoir_content",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "forecast_albedo",
    "surface_latent_heat_flux",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "surface_sensible_heat_flux",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "evaporation_from_bare_soil",
    "evaporation_from_open_water_surfaces_excluding_oceans",
    "evaporation_from_the_top_of_canopy",
    "evaporation_from_vegetation_transpiration",
    "potential_evaporation",
    "runoff",
    "snow_evaporation",
    "sub_surface_runoff",
    "surface_runoff",
    "total_evaporation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "total_precipitation",
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
]


@dataclass
class ERA5GridBatch:
    """Small tensor bundle for a single ERA5 variable on a space-time grid.

    `spatial_coords` uses shape `[N_s, 2]` with columns `(lon, lat)`.
    """

    times: torch.Tensor
    spatial_coords: torch.Tensor
    observations: torch.Tensor
    latitude: np.ndarray
    longitude: np.ndarray
    variable: str


@dataclass
class ERA5SplitTensor:
    """One aligned split of ERA5 pointwise time series.

    `spatial_coords` uses shape `[N_s, 2]` with columns `(lon, lat)`.
    """

    times: torch.Tensor
    spatial_coords: torch.Tensor
    observations: torch.Tensor
    covariates: Optional[torch.Tensor] = None


@dataclass
class ERA5ProcessedTaskBatch:
    """Aggregated processed ERA5 task with train/val/test splits."""

    train: ERA5SplitTensor
    val: ERA5SplitTensor
    test: ERA5SplitTensor
    variable_index: int
    variable_name: str
    task_dir: Path
    scaled: bool


def _find_coord_name(dataset, candidates: Iterable[str]) -> str:
    for candidate in candidates:
        if candidate in dataset.coords:
            return candidate
    raise KeyError(f"Could not find any coordinate in {list(candidates)}.")


def _to_time_offsets(time_values: np.ndarray) -> torch.Tensor:
    base = time_values[0]
    delta = (time_values - base) / np.timedelta64(1, "D")
    return torch.as_tensor(delta.astype(np.float64))


_SEQ_PATTERN = re.compile(r"lat_([-0-9.]+)_lon_([-0-9.]+)")


def _parse_lat_lon_from_name(path: Path) -> tuple[float, float]:
    match = _SEQ_PATTERN.search(path.stem.replace("_scaled", ""))
    if match is None:
        raise ValueError(f"Could not parse lat/lon from filename: {path.name}")
    return float(match.group(1)), float(match.group(2))


def era5_variable_name(variable_index: int) -> str:
    """Map processed ERA5 feature index to a human-readable variable name."""
    if variable_index < 0 or variable_index >= len(ERA5_LAND_VARIABLES):
        return f"variable_{variable_index}"
    return ERA5_LAND_VARIABLES[variable_index]


def _sorted_split_values(data: np.ndarray, times: np.ndarray, variable_index: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(data[variable_index], dtype=np.float64).reshape(-1)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    order = np.argsort(times)
    return times[order], values[order]


def _sorted_split_matrix(
    data: np.ndarray,
    times: np.ndarray,
    variable_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(data[variable_indices], dtype=np.float64)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    order = np.argsort(times)
    return times[order], values[:, order]


def _build_split_from_reference_times(
    reference_times: np.ndarray,
    location_series: list[tuple[dict[float, float], Optional[dict[float, np.ndarray]], tuple[float, float]]],
) -> ERA5SplitTensor:
    if not location_series:
        raise ValueError("No location series were provided.")

    common_times = np.asarray(reference_times, dtype=np.float64)
    for lookup, _, _ in location_series:
        available = np.asarray(sorted(lookup.keys()), dtype=np.float64)
        common_times = np.intersect1d(common_times, available)
    common_times = np.sort(common_times)
    if common_times.size == 0:
        raise ValueError("No common timestamps across selected ERA5 locations.")

    obs_columns = []
    covariate_columns = []
    spatial_coords = []
    for lookup, covariate_lookup, (lat, lon) in location_series:
        obs_columns.append([lookup[float(t)] for t in common_times])
        if covariate_lookup is not None:
            covariate_columns.append(np.stack([covariate_lookup[float(t)] for t in common_times], axis=0))
        spatial_coords.append([lon, lat])

    observations = torch.as_tensor(np.stack(obs_columns, axis=-1), dtype=torch.float64)
    spatial = torch.as_tensor(np.asarray(spatial_coords, dtype=np.float64), dtype=torch.float64)
    times_tensor = torch.as_tensor(common_times, dtype=torch.float64)
    covariates = None
    if covariate_columns:
        covariates = torch.as_tensor(np.stack(covariate_columns, axis=1), dtype=torch.float64)
    return ERA5SplitTensor(
        times=times_tensor,
        spatial_coords=spatial,
        observations=observations,
        covariates=covariates,
    )


def _build_full_common_series(
    location_series: list[tuple[dict[float, float], Optional[dict[float, np.ndarray]], tuple[float, float]]],
) -> ERA5SplitTensor:
    """Build one fully aligned time series across all selected locations."""

    if not location_series:
        raise ValueError("No location series were provided.")
    reference_times = np.asarray(sorted(location_series[0][0].keys()), dtype=np.float64)
    return _build_split_from_reference_times(reference_times, location_series)


def _slice_split_tensor(split: ERA5SplitTensor, start: int, stop: int) -> ERA5SplitTensor:
    return ERA5SplitTensor(
        times=split.times[start:stop].clone(),
        spatial_coords=split.spatial_coords.clone(),
        observations=split.observations[start:stop].clone(),
        covariates=split.covariates[start:stop].clone() if split.covariates is not None else None,
    )


def _resplit_full_series(
    full_series: ERA5SplitTensor,
    train_fraction: float,
    val_fraction: float,
) -> tuple[ERA5SplitTensor, ERA5SplitTensor, ERA5SplitTensor]:
    """Chronologically resplit one aligned full series into train/val/test."""

    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must lie strictly between 0 and 1.")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must lie strictly between 0 and 1.")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be smaller than 1.")

    total_steps = full_series.times.shape[0]
    train_stop = int(round(total_steps * train_fraction))
    val_stop = int(round(total_steps * (train_fraction + val_fraction)))
    train_stop = min(max(train_stop, 1), total_steps - 2)
    val_stop = min(max(val_stop, train_stop + 1), total_steps - 1)

    return (
        _slice_split_tensor(full_series, 0, train_stop),
        _slice_split_tensor(full_series, train_stop, val_stop),
        _slice_split_tensor(full_series, val_stop, total_steps),
    )


def _concatenate_split_tensors(splits: list[ERA5SplitTensor]) -> ERA5SplitTensor:
    """Concatenate chronologically ordered splits with shared spatial coordinates."""

    if not splits:
        raise ValueError("Expected at least one split to concatenate.")

    reference_spatial = splits[0].spatial_coords
    time_chunks: list[torch.Tensor] = []
    obs_chunks: list[torch.Tensor] = []
    covariate_chunks: list[torch.Tensor] = []
    has_covariates = splits[0].covariates is not None
    last_end: float | None = None
    for split in splits:
        if split.spatial_coords.shape != reference_spatial.shape or not torch.allclose(
            split.spatial_coords,
            reference_spatial,
        ):
            raise ValueError("All concatenated ERA5 task splits must share the same spatial coordinates.")
        if (split.covariates is not None) != has_covariates:
            raise ValueError("All concatenated ERA5 task splits must either all have covariates or all omit them.")

        times = split.times.clone()
        if last_end is not None and times.numel() > 0 and float(times[0]) <= last_end:
            times = times + (last_end + 1.0 - float(times[0]))
        if times.numel() > 0:
            last_end = float(times[-1])
        time_chunks.append(times)
        obs_chunks.append(split.observations.clone())
        if split.covariates is not None:
            covariate_chunks.append(split.covariates.clone())

    return ERA5SplitTensor(
        times=torch.cat(time_chunks, dim=0),
        spatial_coords=reference_spatial.clone(),
        observations=torch.cat(obs_chunks, dim=0),
        covariates=torch.cat(covariate_chunks, dim=0) if covariate_chunks else None,
    )


def _load_processed_location_series(
    task_dir: str | Path,
    variable_index: int = 0,
    covariate_indices: Optional[list[int]] = None,
    max_locations: Optional[int] = None,
    scaled: bool = True,
    location_stride: int = 1,
) -> tuple[list[tuple[dict[float, float], Optional[dict[float, np.ndarray]], tuple[float, float]]], np.ndarray, np.ndarray, np.ndarray]:
    """Load per-location time/value lookups from one processed ERA5 task directory."""

    task_dir = Path(task_dir)
    sequence_dir = task_dir / "sequences"
    if not sequence_dir.exists():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}")

    suffix = "*_scaled.npz" if scaled else "*.npz"
    files = sorted(sequence_dir.glob(suffix))
    if not scaled:
        files = [path for path in files if not path.name.endswith("_scaled.npz")]
    if not files:
        raise FileNotFoundError(f"No processed ERA5 sequence files found under {sequence_dir}")

    files = files[:: max(location_stride, 1)]
    if max_locations is not None:
        files = files[:max_locations]
    if not files:
        raise ValueError("No processed ERA5 sequence files remain after subsampling.")

    covariate_indices = [int(index) for index in (covariate_indices or []) if int(index) != variable_index]
    location_series: list[tuple[dict[float, float], Optional[dict[float, np.ndarray]], tuple[float, float]]] = []
    reference_train_times: np.ndarray | None = None
    reference_val_times: np.ndarray | None = None
    reference_test_times: np.ndarray | None = None

    for path in files:
        lat_lon = _parse_lat_lon_from_name(path)
        with np.load(path) as arr:
            train_times, train_values = _sorted_split_values(arr["data_train"], arr["time_train"], variable_index)
            val_times, val_values = _sorted_split_values(arr["data_val"], arr["time_val"], variable_index)
            test_times, test_values = _sorted_split_values(arr["data_test"], arr["time_test"], variable_index)

            full_lookup = {
                float(t): float(v)
                for times, values in [
                    (train_times, train_values),
                    (val_times, val_values),
                    (test_times, test_values),
                ]
                for t, v in zip(times, values)
            }
            covariate_lookup: Optional[dict[float, np.ndarray]] = None
            if covariate_indices:
                cov_train_times, cov_train_values = _sorted_split_matrix(
                    arr["data_train"],
                    arr["time_train"],
                    covariate_indices,
                )
                cov_val_times, cov_val_values = _sorted_split_matrix(
                    arr["data_val"],
                    arr["time_val"],
                    covariate_indices,
                )
                cov_test_times, cov_test_values = _sorted_split_matrix(
                    arr["data_test"],
                    arr["time_test"],
                    covariate_indices,
                )
                covariate_lookup = {
                    float(t): value.astype(np.float64, copy=False)
                    for times, values in [
                        (cov_train_times, cov_train_values),
                        (cov_val_times, cov_val_values),
                        (cov_test_times, cov_test_values),
                    ]
                    for t, value in zip(times, values.transpose(1, 0))
                }
            location_series.append((full_lookup, covariate_lookup, lat_lon))
            if reference_train_times is None:
                reference_train_times = train_times
                reference_val_times = val_times
                reference_test_times = test_times

    return location_series, reference_train_times, reference_val_times, reference_test_times


def load_era5_grid(
    path: str | Path,
    variable: str,
    time_range: Optional[tuple[str, str]] = None,
    lat_range: Optional[tuple[float, float]] = None,
    lon_range: Optional[tuple[float, float]] = None,
    spatial_stride: int = 1,
    time_stride: int = 1,
) -> ERA5GridBatch:
    """Load a small ERA5 cube from NetCDF or Zarr via xarray."""

    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("xarray is required for ERA5 loading.") from exc

    path = Path(path)
    if path.suffix == ".zarr":
        ds = xr.open_zarr(path)
    else:
        ds = xr.open_dataset(path)

    if variable not in ds:
        raise KeyError(f"Variable `{variable}` not found in dataset.")

    da = ds[variable]
    time_name = _find_coord_name(da.to_dataset(name=variable), ["time", "valid_time"])
    lat_name = _find_coord_name(da.to_dataset(name=variable), ["latitude", "lat"])
    lon_name = _find_coord_name(da.to_dataset(name=variable), ["longitude", "lon"])

    if time_range is not None:
        da = da.sel({time_name: slice(*time_range)})
    if lat_range is not None:
        lo, hi = min(lat_range), max(lat_range)
        da = da.sel({lat_name: slice(hi, lo) if da[lat_name][0] > da[lat_name][-1] else slice(lo, hi)})
    if lon_range is not None:
        lo, hi = min(lon_range), max(lon_range)
        da = da.sel({lon_name: slice(lo, hi)})

    da = da.isel(
        {
            time_name: slice(None, None, time_stride),
            lat_name: slice(None, None, spatial_stride),
            lon_name: slice(None, None, spatial_stride),
        }
    )

    data = da.transpose(time_name, lat_name, lon_name).values
    times = _to_time_offsets(da[time_name].values)
    lat = np.asarray(da[lat_name].values)
    lon = np.asarray(da[lon_name].values)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    spatial_coords = torch.as_tensor(
        np.stack([lon_grid.reshape(-1), lat_grid.reshape(-1)], axis=-1),
        dtype=torch.float64,
    )
    observations = torch.as_tensor(data.reshape(data.shape[0], -1), dtype=torch.float64)
    return ERA5GridBatch(
        times=times,
        spatial_coords=spatial_coords,
        observations=observations,
        latitude=lat,
        longitude=lon,
        variable=variable,
    )


def load_processed_era5_task(
    task_dir: str | Path,
    variable_index: int = 0,
    covariate_indices: Optional[list[int]] = None,
    max_locations: Optional[int] = None,
    scaled: bool = True,
    location_stride: int = 1,
    resplit: bool = False,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> ERA5ProcessedTaskBatch:
    """Load a processed ERA5 task directory made of per-location `.npz` files.

    Expected format:
    - `task_dir/sequences/*.npz`
    - each file contains `data_train/time_train/data_val/time_val/data_test/time_test`
    - `data_*` has shape `[num_variables, num_time_points]`
    """

    task_dir = Path(task_dir)
    (
        location_series,
        reference_train_times,
        reference_val_times,
        reference_test_times,
    ) = _load_processed_location_series(
        task_dir=task_dir,
        variable_index=variable_index,
        covariate_indices=covariate_indices,
        max_locations=max_locations,
        scaled=scaled,
        location_stride=location_stride,
    )

    if resplit:
        full_series = _build_full_common_series(location_series)
        train_split, val_split, test_split = _resplit_full_series(
            full_series,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
    else:
        train_split = _build_split_from_reference_times(reference_train_times, location_series)
        val_split = _build_split_from_reference_times(reference_val_times, location_series)
        test_split = _build_split_from_reference_times(reference_test_times, location_series)

    return ERA5ProcessedTaskBatch(
        train=train_split,
        val=val_split,
        test=test_split,
        variable_index=variable_index,
        variable_name=era5_variable_name(variable_index),
        task_dir=task_dir,
        scaled=scaled,
    )


def load_processed_era5_tasks(
    task_dirs: list[str | Path],
    variable_index: int = 0,
    covariate_indices: Optional[list[int]] = None,
    max_locations: Optional[int] = None,
    scaled: bool = True,
    location_stride: int = 1,
    resplit: bool = True,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> ERA5ProcessedTaskBatch:
    """Load and concatenate multiple processed ERA5 task directories."""

    if not task_dirs:
        raise ValueError("Expected at least one processed ERA5 task directory.")

    per_task_batches: list[ERA5ProcessedTaskBatch] = []
    for task_dir in task_dirs:
        per_task_batches.append(
            load_processed_era5_task(
                task_dir=task_dir,
                variable_index=variable_index,
                covariate_indices=covariate_indices,
                max_locations=max_locations,
                scaled=scaled,
                location_stride=location_stride,
                resplit=False,
            )
        )

    full_series = _concatenate_split_tensors(
        [
            _concatenate_split_tensors([batch.train, batch.val, batch.test])
            for batch in per_task_batches
        ]
    )

    if resplit:
        train_split, val_split, test_split = _resplit_full_series(
            full_series,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
    else:
        train_split = _concatenate_split_tensors([batch.train for batch in per_task_batches])
        val_split = _concatenate_split_tensors([batch.val for batch in per_task_batches])
        test_split = _concatenate_split_tensors([batch.test for batch in per_task_batches])

    last_task_dir = Path(task_dirs[-1])
    return ERA5ProcessedTaskBatch(
        train=train_split,
        val=val_split,
        test=test_split,
        variable_index=variable_index,
        variable_name=era5_variable_name(variable_index),
        task_dir=last_task_dir,
        scaled=scaled,
    )


def discover_processed_era5_task_dirs(
    task_root: str | Path,
    task_names: Optional[Iterable[str]] = None,
) -> list[Path]:
    """Discover processed ERA5 task directories under `task_root`.

    When `task_names` is omitted, all `task_*` directories are returned.
    """

    task_root = Path(task_root)
    if task_names is None:
        task_dirs = sorted(path for path in task_root.glob("task_*") if path.is_dir())
    else:
        task_dirs = [task_root / name for name in task_names]
        missing = [path for path in task_dirs if not path.is_dir()]
        if missing:
            missing_names = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing processed ERA5 task directories: {missing_names}")
    if not task_dirs:
        raise FileNotFoundError(f"No task_* directories found under {task_root}")
    return task_dirs


def count_processed_era5_locations(
    task_dir: str | Path,
    scaled: bool = True,
    location_stride: int = 1,
) -> int:
    """Count available processed ERA5 locations in one task directory."""

    task_dir = Path(task_dir)
    sequence_dir = task_dir / "sequences"
    if not sequence_dir.exists():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}")

    suffix = "*_scaled.npz" if scaled else "*.npz"
    files = sorted(sequence_dir.glob(suffix))
    if not scaled:
        files = [path for path in files if not path.name.endswith("_scaled.npz")]
    files = files[:: max(location_stride, 1)]
    return len(files)


def build_temporal_blocks(
    times: torch.Tensor,
    block_size: int,
    overlap: int = 0,
    num_discrete_steps: Optional[int] = None,
) -> list[tuple[slice, TemporalBlockSpec]]:
    """Create temporal block slices plus local horizon specs."""

    times = torch.as_tensor(times, dtype=torch.float64).reshape(-1)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if overlap >= block_size:
        raise ValueError("overlap must be smaller than block_size.")

    blocks: list[tuple[slice, TemporalBlockSpec]] = []
    step = block_size - overlap
    start = 0
    while start < times.shape[0]:
        stop = min(start + block_size, times.shape[0])
        block_slice = slice(start, stop)
        block_spec = TemporalBlockSpec.from_times(
            times[block_slice],
            num_discrete_steps=num_discrete_steps or (stop - start),
            prev_discrete_steps=start,
        )
        blocks.append((block_slice, block_spec))
        if stop == times.shape[0]:
            break
        start += step
    return blocks
