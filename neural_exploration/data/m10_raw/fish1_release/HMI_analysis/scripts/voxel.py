#!/usr/bin/env python3
"""Generate voxelized z-slice overlay figures from CAVE soma positions.

The script loads cached CAVE soma positions by default, matches them to the project
dataframe, voxelizes the selected hindbrain region, and saves one SVG per z-slice
plus one collapsed xy SVG. The CAVE cache can be refreshed with --refresh-cache.
"""

import argparse
from pathlib import Path

import caveclient
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zfish.labels import class_border_color, class_fill_color


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "em_zfish1_dataframe.xlsx"
DEFAULT_SOMA_CACHE_PATH = PROJECT_ROOT / "data" / "cave_somas_in_big_box.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "voxel"

GLOBAL_URL = "https://global.brain-wire-test.org"
DATASTACK = "fish1_full"
SOMAS_TABLE = "somas"
MATERIALIZATION_VERSION = "latest"

CELL_ID = "Cell ID"
CLASSIFIER = "classifier"

BIG_BOX_MIN = np.array([56302, 15373, 3338], dtype=float)
BIG_BOX_MAX = np.array([76765, 49808, 7367], dtype=float)

MIDLINE_Y = 32804.0
BASE_XY_UNITS = 15.0
BASE_Z_UNITS = 4.0
VOXEL_SIZE_MULTIPLIER = 84

COLLAPSE_SLICE_START = 1
COLLAPSE_SLICE_END = 6

OVERLAY_CLASSES = ["1", "2", "L-2"]
SEED_CELLS = [89189, 137722, 149747, 119243]

SAVE_FORMAT = "svg"
DPI = 180


def main():
    args = parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    soma_cache_path = Path(args.soma_cache)
    dataframe_cell_ids, classifier_by_cell = read_project_dataframe(args.data)

    client = None
    materialization_version = args.materialization_version

    if args.refresh_cache or not soma_cache_path.exists():
        client = make_cave_client()
        materialization_version = get_materialization_version(
            client,
            args.materialization_version,
        )

    x_edges, y_edges, z_edges = make_voxel_edges()

    somas = load_or_query_cave_somas(
        client=client,
        materialization_version=materialization_version,
        cache_path=soma_cache_path,
        refresh_cache=args.refresh_cache,
    )

    somas = annotate_somas(somas, dataframe_cell_ids, classifier_by_cell)
    somas = crop_somas_to_voxel_region(somas, x_edges, y_edges, z_edges)

    total_somas, matched_dataframe_cells, matched_fraction = compute_voxel_arrays(
        somas,
        x_edges,
        y_edges,
        z_edges,
    )

    print_run_summary(
        somas=somas,
        dataframe_cell_ids=dataframe_cell_ids,
        materialization_version=materialization_version,
        soma_cache_path=soma_cache_path,
        x_edges=x_edges,
        y_edges=y_edges,
        z_edges=z_edges,
        total_somas=total_somas,
        matched_dataframe_cells=matched_dataframe_cells,
        matched_fraction=matched_fraction,
    )

    plot_z_slices(
        somas,
        x_edges,
        y_edges,
        z_edges,
        total_somas,
        matched_dataframe_cells,
        matched_fraction,
        output_dir,
    )

    plot_collapsed_xy(
        somas,
        x_edges,
        y_edges,
        z_edges,
        total_somas,
        matched_dataframe_cells,
        output_dir,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate voxelized CAVE soma z-slice overlay figures."
    )
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--soma-cache", default=DEFAULT_SOMA_CACHE_PATH)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--materialization-version", default=MATERIALIZATION_VERSION)
    return parser.parse_args()


def read_project_dataframe(path):
    dataframe = pd.read_excel(path).copy()
    dataframe.columns = dataframe.columns.astype(str).str.strip()

    if CELL_ID not in dataframe.columns:
        raise KeyError(f"Missing required column: {CELL_ID}")

    if CLASSIFIER not in dataframe.columns:
        dataframe[CLASSIFIER] = "n/a"

    dataframe["_cell_id"] = dataframe[CELL_ID].apply(safe_int)
    dataframe = dataframe[dataframe["_cell_id"].notna()].copy()
    dataframe["_cell_id"] = dataframe["_cell_id"].astype(int)
    dataframe[CLASSIFIER] = dataframe[CLASSIFIER].astype(str).str.strip()
    dataframe = dataframe.drop_duplicates(subset=["_cell_id"])

    dataframe_cell_ids = set(dataframe["_cell_id"])
    classifier_by_cell = dict(zip(dataframe["_cell_id"], dataframe[CLASSIFIER]))

    return dataframe_cell_ids, classifier_by_cell


def make_cave_client():
    try:
        client = caveclient.CAVEclient(
            datastack_name=DATASTACK,
            server_address=GLOBAL_URL,
        )
        client.materialize.get_tables()
        return client

    except Exception as error:
        raise RuntimeError(
            "\nCAVE authentication failed.\n\n"
            "Set up your CAVE token first, then rerun this script.\n\n"
            "In Python, run:\n\n"
            "    import caveclient\n"
            "    client = caveclient.CAVEclient(\n"
            "        server_address='https://global.brain-wire-test.org'\n"
            "    )\n"
            "    client.auth.get_new_token(open=True)\n"
            "    client.auth.save_token(token='YOUR_TOKEN_HERE', overwrite=True)\n\n"
            "Do not put the token into this repository.\n"
        ) from error


def get_materialization_version(client, value):
    if value is None or str(value).lower() == "latest":
        return client.materialize.most_recent_version()

    return int(value)


def load_or_query_cave_somas(client, materialization_version, cache_path, refresh_cache):
    cache_path = Path(cache_path)

    if cache_path.exists() and not refresh_cache:
        print(f"Loaded cached CAVE somas: {cache_path}")
        print(f"Cache last modified: {file_modified_time(cache_path)}")
        return pd.read_csv(cache_path)

    if client is None:
        raise RuntimeError(
            "CAVE client is required because the soma cache is missing or "
            "--refresh-cache was requested."
        )

    somas = load_cave_somas(client, materialization_version)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    somas.to_csv(cache_path, index=False)

    print(f"Saved CAVE soma cache: {cache_path}")
    print(f"Cache last modified: {file_modified_time(cache_path)}")

    return somas


def load_cave_somas(client, materialization_version):
    client.materialize.version = materialization_version
    somas = client.materialize.query_table(SOMAS_TABLE).copy()

    x_column, y_column, z_column = get_position_columns(somas)

    somas["_x"] = pd.to_numeric(somas[x_column], errors="coerce")
    somas["_y"] = pd.to_numeric(somas[y_column], errors="coerce")
    somas["_z"] = pd.to_numeric(somas[z_column], errors="coerce")
    somas["_cell_id"] = pd.to_numeric(somas["id"], errors="coerce")

    somas = somas.dropna(subset=["_cell_id", "_x", "_y", "_z"]).copy()
    somas["_cell_id"] = somas["_cell_id"].astype(int)

    in_big_box = (
        (somas["_x"] >= BIG_BOX_MIN[0])
        & (somas["_x"] <= BIG_BOX_MAX[0])
        & (somas["_y"] >= BIG_BOX_MIN[1])
        & (somas["_y"] <= BIG_BOX_MAX[1])
        & (somas["_z"] >= BIG_BOX_MIN[2])
        & (somas["_z"] <= BIG_BOX_MAX[2])
    )

    return somas[in_big_box].copy()


def get_position_columns(somas):
    if {"pt_position_x", "pt_position_y", "pt_position_z"}.issubset(somas.columns):
        return "pt_position_x", "pt_position_y", "pt_position_z"

    if "pt_position" in somas.columns:
        positions = np.vstack(somas["pt_position"].apply(parse_position).values)
        somas["_x"] = positions[:, 0]
        somas["_y"] = positions[:, 1]
        somas["_z"] = positions[:, 2]
        return "_x", "_y", "_z"

    raise KeyError(
        "Could not find soma position columns. Expected either "
        "pt_position or pt_position_x/pt_position_y/pt_position_z."
    )


def parse_position(value):
    if isinstance(value, dict):
        return (
            float(value.get("x", np.nan)),
            float(value.get("y", np.nan)),
            float(value.get("z", np.nan)),
        )

    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 3:
        return float(value[0]), float(value[1]), float(value[2])

    if isinstance(value, str):
        cleaned = (
            value.replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
            .replace(",", " ")
        )
        parts = cleaned.split()

        if len(parts) >= 3:
            return float(parts[0]), float(parts[1]), float(parts[2])

    return np.nan, np.nan, np.nan


def make_voxel_edges():
    x_step = BASE_XY_UNITS * VOXEL_SIZE_MULTIPLIER
    y_step = BASE_XY_UNITS * VOXEL_SIZE_MULTIPLIER
    z_step = BASE_Z_UNITS * VOXEL_SIZE_MULTIPLIER

    x_edges = make_edges_from_min(BIG_BOX_MIN[0], BIG_BOX_MAX[0], x_step)
    y_edges = make_y_edges_around_midline(BIG_BOX_MIN[1], BIG_BOX_MAX[1], MIDLINE_Y, y_step)
    z_edges = make_edges_from_min(BIG_BOX_MIN[2], BIG_BOX_MAX[2], z_step)

    return x_edges, y_edges, z_edges


def make_edges_from_min(axis_min, axis_max, step):
    number_of_bins = int(np.floor((axis_max - axis_min) / step))
    return axis_min + np.arange(number_of_bins + 1) * step


def make_y_edges_around_midline(y_min, y_max, midline_y, step):
    number_of_lower_bins = int(np.floor((midline_y - y_min) / step))
    number_of_upper_bins = int(np.floor((y_max - midline_y) / step))

    lower_edges = midline_y - np.arange(number_of_lower_bins, -1, -1) * step
    upper_edges = midline_y + np.arange(1, number_of_upper_bins + 1) * step

    return np.concatenate([lower_edges, upper_edges])


def annotate_somas(somas, dataframe_cell_ids, classifier_by_cell):
    somas = somas.copy()
    somas["_is_dataframe_cell"] = somas["_cell_id"].isin(dataframe_cell_ids)
    somas["_classifier"] = somas["_cell_id"].map(classifier_by_cell).fillna("n/a")
    return somas


def crop_somas_to_voxel_region(somas, x_edges, y_edges, z_edges):
    in_voxel_region = (
        (somas["_x"] >= x_edges[0])
        & (somas["_x"] < x_edges[-1])
        & (somas["_y"] >= y_edges[0])
        & (somas["_y"] < y_edges[-1])
        & (somas["_z"] >= z_edges[0])
        & (somas["_z"] < z_edges[-1])
    )

    return somas[in_voxel_region].copy()


def compute_voxel_arrays(somas, x_edges, y_edges, z_edges):
    number_of_z_bins = len(z_edges) - 1
    number_of_y_bins = len(y_edges) - 1
    number_of_x_bins = len(x_edges) - 1

    total_somas = np.zeros((number_of_z_bins, number_of_y_bins, number_of_x_bins), dtype=int)
    matched_dataframe_cells = np.zeros_like(total_somas)

    for z_index in range(number_of_z_bins):
        z_min = z_edges[z_index]
        z_max = z_edges[z_index + 1]

        z_slice = somas[(somas["_z"] >= z_min) & (somas["_z"] < z_max)]
        matched_z_slice = z_slice[z_slice["_is_dataframe_cell"]]

        total_somas[z_index] = histogram_xy(z_slice, x_edges, y_edges)
        matched_dataframe_cells[z_index] = histogram_xy(matched_z_slice, x_edges, y_edges)

    matched_fraction = np.zeros_like(matched_dataframe_cells, dtype=float)
    nonempty_voxels = total_somas > 0
    matched_fraction[nonempty_voxels] = (
        matched_dataframe_cells[nonempty_voxels] / total_somas[nonempty_voxels]
    )

    return total_somas, matched_dataframe_cells, matched_fraction


def histogram_xy(dataframe, x_edges, y_edges):
    if len(dataframe) == 0:
        return np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=int)

    histogram, _, _ = np.histogram2d(
        dataframe["_x"],
        dataframe["_y"],
        bins=[x_edges, y_edges],
    )

    return histogram.astype(int).T


def plot_z_slices(
    somas,
    x_edges,
    y_edges,
    z_edges,
    total_somas,
    matched_dataframe_cells,
    matched_fraction,
    output_dir,
):
    maximum_fraction = matched_fraction.max()

    if not np.isfinite(maximum_fraction) or maximum_fraction <= 0:
        maximum_fraction = 1.0

    for z_index in range(len(z_edges) - 1):
        z_min = z_edges[z_index]
        z_max = z_edges[z_index + 1]

        overlay_cells = get_overlay_cells(somas, z_min, z_max)
        seed_cells = get_seed_cells(somas, z_min, z_max)

        print(
            f"Slice {z_index + 1:02d}: "
            f"z=[{int(z_min)}, {int(z_max)}), "
            f"total={total_somas[z_index].sum():,}, "
            f"matched={matched_dataframe_cells[z_index].sum():,}, "
            f"max_fraction={matched_fraction[z_index].max():.6f}"
        )

        figure, axis = plt.subplots(figsize=(7.4, 6.2), dpi=DPI)
        image = plot_background(
            axis,
            matched_fraction[z_index],
            x_edges,
            y_edges,
            maximum_fraction,
        )

        overlay_class_points(axis, overlay_cells)
        overlay_seed_points(axis, seed_cells)
        format_axis(
            axis,
            f"z slice {z_index + 1}/{len(z_edges) - 1} | z=[{int(z_min)}, {int(z_max)})",
        )
        add_colorbar(
            figure,
            axis,
            image,
            "df-matching cave somas / total cave somas in voxel",
        )

        save_figure(figure, output_dir / f"z_slice_{z_index + 1:02d}")


def plot_collapsed_xy(
    somas,
    x_edges,
    y_edges,
    z_edges,
    total_somas,
    matched_dataframe_cells,
    output_dir,
):
    start_index = COLLAPSE_SLICE_START - 1
    end_index = COLLAPSE_SLICE_END - 1

    collapsed_total_somas = total_somas[start_index:end_index + 1].sum(axis=0)
    collapsed_matched_cells = matched_dataframe_cells[start_index:end_index + 1].sum(axis=0)

    collapsed_fraction = np.zeros_like(collapsed_matched_cells, dtype=float)
    nonempty_voxels = collapsed_total_somas > 0
    collapsed_fraction[nonempty_voxels] = (
        collapsed_matched_cells[nonempty_voxels] / collapsed_total_somas[nonempty_voxels]
    )

    maximum_fraction = collapsed_fraction.max()

    if not np.isfinite(maximum_fraction) or maximum_fraction <= 0:
        maximum_fraction = 1.0

    z_min = z_edges[start_index]
    z_max = z_edges[end_index + 1]

    overlay_cells = get_overlay_cells(somas, z_min, z_max)
    seed_cells = get_seed_cells(somas, z_min, z_max)

    print("\nCollapsed xy summary")
    print("--------------------")
    print(f"Collapsed slices: {COLLAPSE_SLICE_START} to {COLLAPSE_SLICE_END}")
    print(f"Collapsed total somas: {collapsed_total_somas.sum():,}")
    print(f"Collapsed matched dataframe cells: {collapsed_matched_cells.sum():,}")
    print(f"Collapsed maximum fraction: {collapsed_fraction.max():.6f}")

    figure, axis = plt.subplots(figsize=(8.4, 7.0), dpi=DPI)
    image = plot_background(axis, collapsed_fraction, x_edges, y_edges, maximum_fraction)

    overlay_class_points(axis, overlay_cells)
    overlay_seed_points(axis, seed_cells)
    format_axis(
        axis,
        f"collapsed xy across z slices {COLLAPSE_SLICE_START} to {COLLAPSE_SLICE_END}",
    )
    add_colorbar(
        figure,
        axis,
        image,
        "collapsed df-matching cave somas / collapsed total cave somas in voxel",
    )

    save_figure(
        figure,
        output_dir / f"collapsed_xy_slices_{COLLAPSE_SLICE_START:02d}_to_{COLLAPSE_SLICE_END:02d}",
    )


def get_overlay_cells(somas, z_min, z_max):
    return somas[
        (somas["_z"] >= z_min)
        & (somas["_z"] < z_max)
        & (somas["_classifier"].isin(OVERLAY_CLASSES))
    ]


def get_seed_cells(somas, z_min, z_max):
    return somas[
        (somas["_z"] >= z_min)
        & (somas["_z"] < z_max)
        & (somas["_cell_id"].isin(SEED_CELLS))
    ]


def plot_background(axis, data, x_edges, y_edges, maximum_fraction):
    return axis.imshow(
        data,
        cmap="Greys",
        origin="lower",
        vmin=0,
        vmax=maximum_fraction,
        interpolation="nearest",
        aspect="equal",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        zorder=1,
    )


def overlay_class_points(axis, somas):
    for classifier in OVERLAY_CLASSES:
        subset = somas[somas["_classifier"] == classifier]

        if len(subset) == 0:
            continue

        axis.scatter(
            subset["_x"],
            subset["_y"],
            s=10,
            c=class_fill_color(classifier),
            alpha=0.65,
            edgecolors=class_border_color(classifier),
            linewidths=0.4,
            zorder=8,
            label=classifier,
        )


def overlay_seed_points(axis, somas):
    if len(somas) == 0:
        return

    axis.scatter(
        somas["_x"],
        somas["_y"],
        s=60,
        marker="^",
        c="#0000FF",
        edgecolors="white",
        linewidths=0.2,
        alpha=1.0,
        zorder=10,
    )


def format_axis(axis, title):
    axis.set_xlim(BIG_BOX_MIN[0], BIG_BOX_MAX[0])
    axis.set_ylim(BIG_BOX_MAX[1], BIG_BOX_MIN[1])  # flip figure vertically
    axis.set_xlabel("x", fontsize=11)
    axis.set_ylabel("y", fontsize=11)
    axis.tick_params(labelsize=9)
    axis.set_title(title, fontsize=13)


def add_colorbar(figure, axis, image, label):
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(label, fontsize=11)
    colorbar.ax.tick_params(labelsize=9)


def save_figure(figure, path):
    path = Path(path).with_suffix(f".{SAVE_FORMAT}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved SVG: {path}")


def safe_int(value):
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def file_modified_time(path):
    timestamp = Path(path).stat().st_mtime
    return pd.Timestamp.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def print_run_summary(
    somas,
    dataframe_cell_ids,
    materialization_version,
    soma_cache_path,
    x_edges,
    y_edges,
    z_edges,
    total_somas,
    matched_dataframe_cells,
    matched_fraction,
):
    seed_cells_found = sorted(
        somas.loc[somas["_cell_id"].isin(SEED_CELLS), "_cell_id"].tolist()
    )

    print("\nVoxel z-slice figures")
    print("---------------------")
    print(f"Materialization version: {materialization_version}")
    print(f"Soma cache: {soma_cache_path}")
    print(f"Soma cache last modified: {file_modified_time(soma_cache_path)}")
    print(f"Big box min: {BIG_BOX_MIN.tolist()}")
    print(f"Big box max: {BIG_BOX_MAX.tolist()}")
    print(f"Overlay classes: {OVERLAY_CLASSES}")
    print(f"Dataframe cell IDs: {len(dataframe_cell_ids):,}")
    print(f"CAVE somas inside voxelized region: {len(somas):,}")
    print(f"CAVE somas matching dataframe: {int(somas['_is_dataframe_cell'].sum()):,}")
    print(f"Seed cells found: {seed_cells_found}")
    print(f"Voxel grid: x={len(x_edges) - 1}, y={len(y_edges) - 1}, z={len(z_edges) - 1}")
    print(f"Total somas in voxels: {total_somas.sum():,}")
    print(f"Matched dataframe cells in voxels: {matched_dataframe_cells.sum():,}")
    print(f"Maximum voxel fraction: {matched_fraction.max():.6f}")
    print(f"Nonempty voxels: {(total_somas > 0).sum():,}")
    print(f"Dataframe-positive voxels: {(matched_dataframe_cells > 0).sum():,}")


if __name__ == "__main__":
    main()