#!/usr/bin/env python3
"""Generate the pairwise connectivity matrix figure.

The script builds an annotated graph from the project dataframe, filters the graph to
match the selected cell classes and network layers, overlays SPN cells that touch
that filtered graph, and saves one SVG connectivity matrix.
"""

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch, Rectangle

from zfish.graph import (
    add_computed_graph_metadata,
    add_neural_metadata,
    build_graph,
    load_df,
)
from zfish.labels import (
    class_label,
    is_proximal_spn,
    is_spn_cell,
    neurotransmitter_color,
    neurotransmitter_label,
    parse_network_levels,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "em_zfish1_dataframe.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "connectivity_matrix"

OUTPUT_FILENAME = "connectivity_matrix.svg"

CLASSIFIER = "classifier"
NETWORK_LEVEL = "network_level"
NEUROTRANSMITTER = "final_neurotransmitter_ID"

SEED_CELLS = [89189, 137722, 149747, 119243]
ALLOWED_NETWORK_LEVELS = {"output_one", "input_one", "network_seed"}  # excludes input_one_cell_09
ALLOWED_CLASSIFIERS = {1, 1.0, "1", 2, 2.0, "2", "Class 2", "contralateral axon", "L-2"}

CLASS_ORDER = {
    "I": 0,
    "II": 1,
    "L-2": 2,
    "SPN_turning": 3,
    "SPN_forward": 4,
    "SPN_other": 5,
    "?": 9,
    "SPN_prox": 99,
}


def main():
    args = parse_args()
    output_path = Path(args.out) / OUTPUT_FILENAME

    dataframe = load_df(
        args.data,
        required_columns=["inputs", "outputs", CLASSIFIER, NETWORK_LEVEL, NEUROTRANSMITTER],
    )

    full_graph = build_annotated_graph(dataframe)
    matrix_graph = build_matrix_graph(full_graph)
    matrix_data = prepare_matrix_data(matrix_graph)

    print_run_summary(full_graph, matrix_graph, matrix_data)
    plot_connectivity_matrix(matrix_graph, matrix_data, output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the pairwise connectivity matrix figure."
    )
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_annotated_graph(dataframe):
    graph = build_graph(dataframe)
    add_neural_metadata(graph, dataframe)
    add_computed_graph_metadata(graph)
    return graph


def build_matrix_graph(full_graph):
    graph = full_graph.copy()

    for cell_id in list(graph.nodes()):
        network_levels = parse_network_levels(graph.nodes[cell_id].get(NETWORK_LEVEL, "n/a"))

        if not ALLOWED_NETWORK_LEVELS.intersection(network_levels):
            graph.remove_node(cell_id)

    kept_cells = [
        cell_id
        for cell_id, attributes in graph.nodes(data=True)
        if attributes.get(CLASSIFIER) in ALLOWED_CLASSIFIERS
    ]

    graph = graph.subgraph(kept_cells).copy()

    add_touching_spn_cells(graph, full_graph)
    remove_isolated_spn_cells(graph)

    return graph


def prepare_matrix_data(graph):
    ordered_cells, class_by_cell = order_cells_for_matrix(graph)

    presynaptic_cells = [
        cell_id
        for cell_id in ordered_cells
        if class_by_cell[cell_id] != "II"
    ]

    postsynaptic_cells = [
        cell_id
        for cell_id in ordered_cells
        if class_by_cell[cell_id] != "L-2"
    ]

    neurotransmitter_matrix, synapse_matrix = build_connectivity_matrices(
        graph,
        presynaptic_cells,
        postsynaptic_cells,
    )

    row_synapse_counts = np.asarray(synapse_matrix.sum(axis=1)).ravel()

    presynaptic_cells = [
        cell_id
        for row_index, cell_id in enumerate(presynaptic_cells)
        if not (class_by_cell[cell_id].startswith("SPN") and row_synapse_counts[row_index] == 0)
    ]

    neurotransmitter_matrix, synapse_matrix = build_connectivity_matrices(
        graph,
        presynaptic_cells,
        postsynaptic_cells,
    )

    (
        neurotransmitter_matrix,
        synapse_matrix,
        presynaptic_cells,
        postsynaptic_cells,
    ) = remove_empty_rows_and_columns(
        neurotransmitter_matrix,
        synapse_matrix,
        presynaptic_cells,
        postsynaptic_cells,
    )

    return {
        "neurotransmitter_matrix": neurotransmitter_matrix,
        "synapse_matrix": synapse_matrix,
        "presynaptic_cells": presynaptic_cells,
        "postsynaptic_cells": postsynaptic_cells,
        "class_by_cell": class_by_cell,
    }


def plot_connectivity_matrix(graph, matrix_data, output_path):
    neurotransmitter_matrix = matrix_data["neurotransmitter_matrix"]
    synapse_matrix = matrix_data["synapse_matrix"]
    presynaptic_cells = matrix_data["presynaptic_cells"]
    postsynaptic_cells = matrix_data["postsynaptic_cells"]
    class_by_cell = matrix_data["class_by_cell"]

    number_of_rows, number_of_columns = synapse_matrix.shape
    maximum_synapse_count = int(synapse_matrix.max()) if synapse_matrix.size else 1
    synapse_count_cap = max(3, maximum_synapse_count)

    color_matrix = make_color_matrix(
        neurotransmitter_matrix,
        synapse_matrix,
        synapse_count_cap,
    )

    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"

    figure, axis = plt.subplots(figsize=(9, 8), facecolor="white")
    axis.imshow(color_matrix, interpolation="nearest", aspect="auto")

    axis.set_xlim(-3.8, number_of_columns - 0.5)
    axis.set_ylim(number_of_rows - 0.5, -6.05)
    axis.set_xticks(np.arange(-0.5, number_of_columns, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, number_of_rows, 1), minor=True)
    axis.grid(False, which="minor")
    axis.tick_params(which="minor", bottom=False, left=False)

    add_neurotransmitter_strips(axis, graph, presynaptic_cells, postsynaptic_cells)
    add_class_labels_and_boundaries(axis, presynaptic_cells, postsynaptic_cells, class_by_cell)
    add_seed_labels(axis, presynaptic_cells, postsynaptic_cells)
    add_legends(axis, synapse_count_cap)

    axis.set_xlabel("postsynaptic cell")
    axis.set_ylabel("presynaptic cell")
    axis.set_title(
        "Pairwise connectivity matrix (color = NT + syn count) | "
        f"presyn n={number_of_rows}, postsyn n={number_of_columns}"
    )
    axis.set_xticks([])
    axis.set_yticks([])

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(figure)

    print(f"Saved SVG: {output_path}")


def cell_neurotransmitter(graph, cell_id):
    return neurotransmitter_label(
        graph.nodes[cell_id].get(NEUROTRANSMITTER, "Unknown")
    )


def synapse_count(edge_attributes):
    try:
        return int(edge_attributes.get("syn_count", 1))
    except Exception:
        return 1


def add_touching_spn_cells(graph, full_graph, exclude_proximal_spns=True):
    base_cells = set(graph.nodes())

    for cell_id, attributes in full_graph.nodes(data=True):
        classifier = attributes.get(CLASSIFIER, "")

        if not is_spn_cell(classifier):
            continue

        if exclude_proximal_spns and is_proximal_spn(classifier):
            continue

        has_base_target = any(
            target_cell in base_cells
            for _, target_cell in full_graph.out_edges(cell_id)
        )

        has_base_source = any(
            source_cell in base_cells
            for source_cell, _ in full_graph.in_edges(cell_id)
        )

        if not has_base_target and not has_base_source:
            continue

        graph.add_node(cell_id, **full_graph.nodes[cell_id])
        visible_cells = set(graph.nodes())

        for source_cell, target_cell in full_graph.out_edges(cell_id):
            if target_cell in visible_cells:
                graph.add_edge(
                    source_cell,
                    target_cell,
                    **full_graph[source_cell][target_cell],
                )

        for source_cell, target_cell in full_graph.in_edges(cell_id):
            if source_cell in visible_cells:
                graph.add_edge(
                    source_cell,
                    target_cell,
                    **full_graph[source_cell][target_cell],
                )


def remove_isolated_spn_cells(graph):
    for cell_id in list(graph.nodes()):
        classifier = graph.nodes[cell_id].get(CLASSIFIER, "")

        if is_spn_cell(classifier) and graph.in_degree(cell_id) == 0 and graph.out_degree(cell_id) == 0:
            graph.remove_node(cell_id)


def order_cells_for_matrix(graph):
    class_by_cell = {
        cell_id: class_label(graph.nodes[cell_id].get(CLASSIFIER, "?"))
        for cell_id in graph.nodes()
    }

    sorted_cells = sorted(
        graph.nodes(),
        key=lambda cell_id: (CLASS_ORDER.get(class_by_cell[cell_id], 999), int(cell_id)),
    )

    seed_cells_in_graph = [
        cell_id
        for cell_id in SEED_CELLS
        if cell_id in sorted_cells
    ]

    ordered_cells = seed_cells_in_graph + [
        cell_id
        for cell_id in sorted_cells
        if cell_id not in seed_cells_in_graph
    ]

    return ordered_cells, class_by_cell


def build_connectivity_matrices(graph, presynaptic_cells, postsynaptic_cells):
    presynaptic_index = {
        cell_id: index
        for index, cell_id in enumerate(presynaptic_cells)
    }

    postsynaptic_index = {
        cell_id: index
        for index, cell_id in enumerate(postsynaptic_cells)
    }

    neurotransmitter_matrix = np.empty(
        (len(presynaptic_cells), len(postsynaptic_cells)),
        dtype=object,
    )
    neurotransmitter_matrix[:] = ""

    synapse_matrix = np.zeros(
        (len(presynaptic_cells), len(postsynaptic_cells)),
        dtype=int,
    )

    for source_cell, target_cell, edge_attributes in graph.edges(data=True):
        if source_cell not in presynaptic_index or target_cell not in postsynaptic_index:
            continue

        row_index = presynaptic_index[source_cell]
        column_index = postsynaptic_index[target_cell]

        neurotransmitter_matrix[row_index, column_index] = cell_neurotransmitter(
            graph,
            source_cell,
        )
        synapse_matrix[row_index, column_index] = synapse_count(edge_attributes)

    return neurotransmitter_matrix, synapse_matrix


def remove_empty_rows_and_columns(
    neurotransmitter_matrix,
    synapse_matrix,
    presynaptic_cells,
    postsynaptic_cells,
):
    row_mask = np.any(synapse_matrix > 0, axis=1)
    column_mask = np.any(synapse_matrix > 0, axis=0)

    presynaptic_cells = [
        cell_id
        for cell_id, keep_cell in zip(presynaptic_cells, row_mask)
        if keep_cell
    ]

    postsynaptic_cells = [
        cell_id
        for cell_id, keep_cell in zip(postsynaptic_cells, column_mask)
        if keep_cell
    ]

    return (
        neurotransmitter_matrix[row_mask][:, column_mask],
        synapse_matrix[row_mask][:, column_mask],
        presynaptic_cells,
        postsynaptic_cells,
    )


def class_blocks(cells, class_by_cell):
    if not cells:
        return [], []

    blocks = []
    boundaries = []
    block_start = 0
    current_class = class_by_cell[cells[0]]

    for index in range(1, len(cells)):
        next_class = class_by_cell[cells[index]]

        if next_class != current_class:
            blocks.append((block_start, index, current_class))
            boundaries.append(index)
            block_start = index
            current_class = next_class

    blocks.append((block_start, len(cells), current_class))

    return blocks, boundaries


def darken_color(base_color, count, count_cap):
    color = np.array(mcolors.to_rgb(base_color))
    effective_count = min(int(count), int(count_cap))
    darkness_fraction = 0.0 if count_cap <= 1 else (effective_count - 1) / (count_cap - 1)

    return color * (1.0 - darkness_fraction * 0.65)


def make_color_matrix(neurotransmitter_matrix, synapse_matrix, count_cap):
    number_of_rows, number_of_columns = synapse_matrix.shape
    color_matrix = np.ones((number_of_rows, number_of_columns, 4), dtype=float)

    for row_index in range(number_of_rows):
        for column_index in range(number_of_columns):
            count = synapse_matrix[row_index, column_index]

            if count <= 0:
                continue

            base_color = neurotransmitter_color(
                neurotransmitter_matrix[row_index, column_index]
            )

            color_matrix[row_index, column_index, :3] = darken_color(
                base_color,
                count,
                count_cap,
            )
            color_matrix[row_index, column_index, 3] = 1.0

    return color_matrix


def add_neurotransmitter_strips(axis, graph, presynaptic_cells, postsynaptic_cells):
    for column_index, cell_id in enumerate(postsynaptic_cells):
        axis.add_patch(
            Rectangle(
                (column_index - 0.5, -3.2),
                1.0,
                0.7,
                facecolor=neurotransmitter_color(cell_neurotransmitter(graph, cell_id)),
                edgecolor="none",
                alpha=0.95,
            )
        )

    for row_index, cell_id in enumerate(presynaptic_cells):
        axis.add_patch(
            Rectangle(
                (-2.7, row_index - 0.5),
                0.7,
                1.0,
                facecolor=neurotransmitter_color(cell_neurotransmitter(graph, cell_id)),
                edgecolor="none",
                alpha=0.95,
            )
        )


def add_class_labels_and_boundaries(axis, presynaptic_cells, postsynaptic_cells, class_by_cell):
    presynaptic_blocks, presynaptic_boundaries = class_blocks(
        presynaptic_cells,
        class_by_cell,
    )
    postsynaptic_blocks, postsynaptic_boundaries = class_blocks(
        postsynaptic_cells,
        class_by_cell,
    )

    for block_start, block_end, label in postsynaptic_blocks:
        axis.text(
            (block_start + block_end - 1) / 2,
            -4.55,
            label,
            ha="center",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.8},
            clip_on=False,
        )

    for block_start, block_end, label in presynaptic_blocks:
        axis.text(
            -2.9,
            (block_start + block_end - 1) / 2,
            label,
            ha="right",
            va="center",
            rotation=90,
            fontsize=8,
        )

    for boundary in presynaptic_boundaries:
        axis.axhline(boundary - 0.5, linewidth=1.0, alpha=0.6)

    for boundary in postsynaptic_boundaries:
        axis.axvline(boundary - 0.5, linewidth=1.0, alpha=0.6)


def add_seed_labels(axis, presynaptic_cells, postsynaptic_cells):
    presynaptic_index = {
        cell_id: index
        for index, cell_id in enumerate(presynaptic_cells)
    }

    postsynaptic_index = {
        cell_id: index
        for index, cell_id in enumerate(postsynaptic_cells)
    }

    for seed_cell in SEED_CELLS:
        if seed_cell in postsynaptic_index:
            axis.text(
                postsynaptic_index[seed_cell],
                -5.45,
                str(seed_cell),
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2, "alpha": 0.9},
                clip_on=False,
            )

        if seed_cell in presynaptic_index:
            axis.text(
                -3.2,
                presynaptic_index[seed_cell],
                str(seed_cell),
                ha="right",
                va="center",
                fontsize=7,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2, "alpha": 0.9},
            )


def add_legends(axis, count_cap):
    neurotransmitter_handles = [
        Patch(facecolor=neurotransmitter_color("VGluT2"), edgecolor="none", label="Exc / VGluT2"),
        Patch(facecolor=neurotransmitter_color("Gad1B"), edgecolor="none", label="Inh / Gad1B"),
        Patch(facecolor=neurotransmitter_color("unclear"), edgecolor="none", label="Unknown / unclear"),
        Patch(facecolor=neurotransmitter_color("unlabeled"), edgecolor="none", label="Unlabeled"),
    ]

    neurotransmitter_legend = axis.legend(
        handles=neurotransmitter_handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.18),
        frameon=True,
        fontsize=8,
        title="NT Identity",
    )

    axis.add_artist(neurotransmitter_legend)

    synapse_handles = []

    for count in [1, 2, 3]:
        synapse_handles.append(
            (
                Patch(facecolor=darken_color(neurotransmitter_color("VGluT2"), count, count_cap), edgecolor="none"),
                Patch(facecolor=darken_color(neurotransmitter_color("Gad1B"), count, count_cap), edgecolor="none"),
                Patch(facecolor=darken_color(neurotransmitter_color("unclear"), count, count_cap), edgecolor="none"),
            )
        )

    axis.legend(
        handles=synapse_handles,
        labels=["1 syn", "2 syns", "3+ syns"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="lower right",
        bbox_to_anchor=(1.0, 0.02),
        frameon=True,
        fontsize=8,
        title="Synapse count\nExc + Inh + Gray",
    )


def print_run_summary(full_graph, matrix_graph, matrix_data):
    synapse_matrix = matrix_data["synapse_matrix"]

    print("\nConnectivity matrix")
    print("-------------------")
    print(f"Full graph cells: {full_graph.number_of_nodes()}")
    print(f"Matrix graph cells: {matrix_graph.number_of_nodes()}")
    print(f"Matrix graph edges: {matrix_graph.number_of_edges()}")
    print(f"Matrix rows: {synapse_matrix.shape[0]}")
    print(f"Matrix columns: {synapse_matrix.shape[1]}")
    print(f"Maximum synapse count: {int(synapse_matrix.max()) if synapse_matrix.size else 0}")


if __name__ == "__main__":
    main()