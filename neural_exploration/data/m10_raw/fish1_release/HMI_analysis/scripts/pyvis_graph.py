#!/usr/bin/env python3
"""Generate the interactive PyVis connectivity graph.

The script builds an annotated graph from the project dataframe, filters it to the
selected network layers and cell classes, overlays SPN cells that touch the filtered
graph, and saves one interactive HTML graph using saved node positions.
"""

import argparse
from pathlib import Path

import networkx as nx

from zfish.graph import (
    add_computed_graph_metadata,
    add_neural_metadata,
    build_graph,
    load_df,
)
from zfish.labels import (
    class_border_and_fill,
    is_proximal_spn,
    is_spn_cell,
    neurotransmitter_color,
    parse_network_levels,
)
from zfish.viz import render_pyvis


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "em_zfish1_dataframe.xlsx"
DEFAULT_POSITIONS_PATH = PROJECT_ROOT / "data" / "pyvis_node_positions.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "pyvis"

OUTPUT_FILENAME = "pyvis.html"

CLASSIFIER = "classifier"
NETWORK_LEVEL = "network_level"
NEUROTRANSMITTER = "final_neurotransmitter_ID"

SEED_CELLS = {89189, 137722, 149747, 119243}
ALLOWED_NETWORK_LEVELS = {"output_one", "input_one", "network_seed"}  # excludes input_one_cell_09
ALLOWED_CLASSIFIERS = {1, 1.0, "1", 2, 2.0, "2", "L-2"}

NODE_SIZE_LARGE = 60
NODE_SIZE_DEFAULT = 40
EDGE_WIDTH = 7
RECIPROCAL_EDGE_WIDTH_BOOST = 2

ARROW_OPTIONS = {
    "to": {
        "enabled": True,
        "scaleFactor": 3,
    }
}

PHYSICS_OPTIONS = """
{
    "physics": {
        "enabled": true,
        "solver": "barnesHut",
        "barnesHut": {
            "gravitationalConstant": -10000,
            "centralGravity": 0,
            "springLength": 250,
            "springConstant": 0.01,
            "damping": 0.9
        },
        "repulsion": {
            "nodeDistance": 300,
            "springLength": 250,
            "springConstant": 0.01
        },
        "minVelocity": 0.1
    },
    "nodes": {
        "physics": true
    },
    "edges": {
        "smooth": {
            "enabled": true,
            "type": "dynamic"
        }
    },
    "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
    }
}
"""


def main():
    args = parse_args()
    output_path = Path(args.out) / OUTPUT_FILENAME

    dataframe = load_df(
        args.data,
        required_columns=["inputs", "outputs", CLASSIFIER, NETWORK_LEVEL, NEUROTRANSMITTER],
    )

    full_graph = build_annotated_graph(dataframe)
    pyvis_graph = build_pyvis_graph(full_graph)
    added_spn_cells = add_touching_spn_cells(pyvis_graph, full_graph)

    print_run_summary(full_graph, pyvis_graph, added_spn_cells)

    render_pyvis(
        graph=pyvis_graph,
        output_path=output_path,
        positions_path=Path(args.positions),
        node_kwargs=pyvis_node_kwargs,
        edge_kwargs=pyvis_edge_kwargs,
        options=PHYSICS_OPTIONS,
        seed=42,
        height="80%",
        width="100%",
        directed=True,
        add_position_button=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the interactive PyVis connectivity graph."
    )
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--positions", default=DEFAULT_POSITIONS_PATH)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_annotated_graph(dataframe):
    graph = build_graph(dataframe)
    add_neural_metadata(graph, dataframe)
    add_computed_graph_metadata(graph)
    add_in_out_degree_metadata(graph)
    return graph


def build_pyvis_graph(full_graph):
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
    add_in_out_degree_metadata(graph)

    return graph


def add_in_out_degree_metadata(graph):
    nx.set_node_attributes(graph, dict(graph.in_degree()), "total_inputs")
    nx.set_node_attributes(graph, dict(graph.out_degree()), "total_outputs")


def add_touching_spn_cells(graph, full_graph, exclude_proximal_spns=True):
    base_cells = set(graph.nodes())
    added_spn_cells = set()

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

        if cell_id not in graph:
            graph.add_node(cell_id, **full_graph.nodes[cell_id])

        added_spn_cells.add(cell_id)
        visible_cells = set(graph.nodes())

        for source_cell, target_cell in full_graph.out_edges(cell_id):
            if target_cell in visible_cells and not graph.has_edge(source_cell, target_cell):
                graph.add_edge(
                    source_cell,
                    target_cell,
                    **full_graph[source_cell][target_cell],
                )

        for source_cell, target_cell in full_graph.in_edges(cell_id):
            if source_cell in visible_cells and not graph.has_edge(source_cell, target_cell):
                graph.add_edge(
                    source_cell,
                    target_cell,
                    **full_graph[source_cell][target_cell],
                )

    add_in_out_degree_metadata(graph)

    return added_spn_cells


def node_border_and_fill(attributes):
    classifier = attributes.get(CLASSIFIER, "")

    if is_proximal_spn(classifier):
        return None

    return class_border_and_fill(classifier)


def node_symbol(attributes):
    return {
        "VGluT2": "+",
        "Gad1B": "-",
        "unclear": "",
    }.get(attributes.get(NEUROTRANSMITTER, ""), "")


def pyvis_node_kwargs(cell_id, attributes, x_position, y_position):
    border_color, fill_color = node_border_and_fill(attributes)
    is_seed_cell = cell_id in SEED_CELLS

    return {
        "title": (
            f"Node: {cell_id}\n"
            f"Morphotype: {attributes.get('final_prediction', 'n/a')}\n"
            f"Neurotransmitter: {attributes.get(NEUROTRANSMITTER, 'n/a')}\n"
            f"Classifier: {attributes.get(CLASSIFIER, 'n/a')}\n"
            f"Inputs: {attributes.get('total_inputs', 'n/a')} "
            f"Out: {attributes.get('total_outputs', 'n/a')}"
        ),
        "label": node_symbol(attributes),
        "color": {
            "background": fill_color,
            "border": border_color,
        },
        "size": NODE_SIZE_LARGE if is_seed_cell else NODE_SIZE_DEFAULT,
        "font": {
            "size": 127.5 if is_seed_cell else 85,
            "color": "black",
            "vadjust": -195 if is_seed_cell else -135,
        },
        "borderWidth": 3,
        "x": x_position,
        "y": y_position,
        "fixed": False,
        "physics": False,
    }


def pyvis_edge_kwargs(source_cell, target_cell, edge_attributes, graph):
    neurotransmitter = graph.nodes[source_cell].get(NEUROTRANSMITTER, "n/a")
    edge_color = neurotransmitter_color(neurotransmitter)
    edge_width = EDGE_WIDTH * 2

    if graph.has_edge(target_cell, source_cell):
        edge_color = darken_hex_color(edge_color, factor=0.5)
        edge_width += RECIPROCAL_EDGE_WIDTH_BOOST

    return {
        "width": edge_width,
        "color": edge_color,
        "arrows": ARROW_OPTIONS,
    }


def darken_hex_color(hex_color, factor=0.5):
    red, green, blue = hex_to_rgb(hex_color)
    return rgb_to_hex((red * factor, green * factor, blue * factor))


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(
        int(hex_color[index:index + 2], 16)
        for index in (0, 2, 4)
    )


def rgb_to_hex(rgb_tuple):
    return "#" + "".join(
        f"{max(0, min(255, int(round(value)))):02X}"
        for value in rgb_tuple
    )


def count_spn_cells(graph):
    return sum(
        1
        for _, attributes in graph.nodes(data=True)
        if is_spn_cell(attributes.get(CLASSIFIER, ""))
    )


def count_eligible_spn_cells(graph):
    return sum(
        1
        for _, attributes in graph.nodes(data=True)
        if is_spn_cell(attributes.get(CLASSIFIER, ""))
        and not is_proximal_spn(attributes.get(CLASSIFIER, ""))
    )


def print_run_summary(full_graph, pyvis_graph, added_spn_cells):
    print("\nPyVis connectivity graph")
    print("------------------------")
    print(f"Full graph cells: {full_graph.number_of_nodes()}")
    print(f"Full graph edges: {full_graph.number_of_edges()}")
    print(f"Full graph SPN cells: {count_spn_cells(full_graph)}")
    print(f"Eligible SPN cells: {count_eligible_spn_cells(full_graph)}")
    print(f"SPN cells added to PyVis graph: {len(added_spn_cells)}")
    print(f"PyVis graph cells: {pyvis_graph.number_of_nodes()}")
    print(f"PyVis graph edges: {pyvis_graph.number_of_edges()}")


if __name__ == "__main__":
    main()