#!/usr/bin/env python3
"""Generate figure-panel-specific graph summary statistics for the HMI analysis.

Panel C:
    Uses dataframe network_level annotations to report the reconstruction pipeline:
    input_one, network_seed, output_one, output_two, and inter-layer edge counts.

Recurrence / rediscovery:
    Uses network_level annotations to quantify which output_two cells were already
    present in input_one, network_seed, or output_one.

Panel E:
    Uses the filtered figure graph:
    input_one + network_seed + output_one.
    Reports neurotransmitter distribution for Class I and Class II cells. L-2 cells
    are counted together with Class II.

Panel H:
    Uses the filtered figure graph:
    input_one + network_seed + output_one.
    Reports direct predecessors of the four seed cells within this filtered graph.

Panel I:
    Uses the Panel I graph:
    input_one + network_seed + output_one + output_two.
    Reports direct successors of all Class I cells that are annotated as network_seed
    or output_one. P targets and literal classifier='other' targets are collapsed
    into O. SPN subclasses are collapsed into one SPN category. For each
    postsynaptic class, the source-neurotransmitter distribution is also reported.

The terminal report is also saved as a text file. Count and percentage data are
saved as a CSV file.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

from zfish.graph import (
    add_computed_graph_metadata,
    add_neural_metadata,
    build_graph,
    load_df,
)
from zfish.labels import (
    class_label,
    neurotransmitter_category,
    parse_network_levels,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "em_zfish1_dataframe.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "graph_summary"

REPORT_FILENAME = "graph_summary.txt"
CSV_FILENAME = "graph_summary_counts.csv"

CELL_ID = "Cell ID"
CLASSIFIER = "classifier"
NETWORK_LEVEL = "network_level"
NEUROTRANSMITTER = "final_neurotransmitter_ID"

SEED_CELLS = [89189, 137722, 149747, 119243]

PANEL_C_LAYER_ORDER = [
    "input_one",
    "network_seed",
    "output_one",
    "output_two",
]

FILTERED_GRAPH_LEVELS = {
    "input_one",
    "network_seed",
    "output_one",
}

PANEL_I_GRAPH_LEVELS = {
    "input_one",
    "network_seed",
    "output_one",
    "output_two",
}

PANEL_E_GROUP_ORDER = [
    "I",
    "II",
]

PANEL_H_SOURCE_CLASS_ORDER = [
    "L-2",
    "II",
    "I",
    "F",
    "R",
    "P",
    "SPN",
    "?",
]

PANEL_I_TARGET_CLASS_ORDER = [
    "II",
    "I",
    "F",
    "R",
    "SPN",
    "O",
    "?",
]

NEUROTRANSMITTER_ORDER = [
    "exc",
    "inh",
    "unc/unl",
    "other",
]

DT_ONLY_NETWORK_LEVELS = {
    "dt_seed",
    "dt_147009_input_one",
    "putative_dt",
}


def main():
    args = parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = load_df(
        args.data,
        required_columns=[
            "inputs",
            "outputs",
            CLASSIFIER,
            NETWORK_LEVEL,
            NEUROTRANSMITTER,
        ],
    )

    full_graph = build_annotated_graph(dataframe)
    levels_by_cell = network_levels_by_cell(dataframe)
    filtered_graph = build_filtered_figure_graph(full_graph, levels_by_cell)
    panel_i_graph = build_panel_i_graph(full_graph, levels_by_cell)

    summary_rows = []
    report_sections = []

    panel_c_rows, panel_c_report = compute_panel_c_summary(
        full_graph,
        levels_by_cell,
    )
    summary_rows.extend(panel_c_rows)
    report_sections.append(panel_c_report)

    recurrence_rows, recurrence_report = compute_recurrence_summary(
        levels_by_cell,
    )
    summary_rows.extend(recurrence_rows)
    report_sections.append(recurrence_report)

    panel_e_rows, panel_e_report = compute_panel_e_summary(filtered_graph)
    summary_rows.extend(panel_e_rows)
    report_sections.append(panel_e_report)

    panel_h_rows, panel_h_report = compute_panel_h_seed_input_summary(filtered_graph)
    summary_rows.extend(panel_h_rows)
    report_sections.append(panel_h_report)

    panel_i_rows, panel_i_report = compute_panel_i_summary(
        panel_i_graph,
        levels_by_cell,
    )
    summary_rows.extend(panel_i_rows)
    report_sections.append(panel_i_report)

    report_text = "\n\n".join(report_sections)
    print(report_text)

    report_path = output_dir / REPORT_FILENAME
    csv_path = output_dir / CSV_FILENAME

    report_path.write_text(report_text + "\n", encoding="utf-8")
    write_summary_csv(summary_rows, csv_path)

    print(f"\nSaved text report: {report_path}")
    print(f"Saved CSV summary: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate figure-panel-specific graph summary statistics."
    )
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_annotated_graph(dataframe):
    graph = build_graph(dataframe)
    add_neural_metadata(graph, dataframe)
    add_computed_graph_metadata(graph)
    return graph


def network_levels_by_cell(dataframe):
    return {
        int(row[CELL_ID]): parse_network_levels(row[NETWORK_LEVEL])
        for _, row in dataframe.iterrows()
    }


def build_filtered_figure_graph(full_graph, levels_by_cell):
    filtered_cells = {
        cell_id
        for cell_id, levels in levels_by_cell.items()
        if full_graph.has_node(cell_id)
        and has_any_level(levels, FILTERED_GRAPH_LEVELS)
        and not is_dt_only_cell(levels)
    }

    return full_graph.subgraph(filtered_cells).copy()


def build_panel_i_graph(full_graph, levels_by_cell):
    panel_i_cells = {
        cell_id
        for cell_id, levels in levels_by_cell.items()
        if full_graph.has_node(cell_id)
        and has_any_level(levels, PANEL_I_GRAPH_LEVELS)
        and not is_dt_only_cell(levels)
    }

    return full_graph.subgraph(panel_i_cells).copy()


def get_cells_with_level(levels_by_cell, network_level):
    return {
        cell_id
        for cell_id, levels in levels_by_cell.items()
        if network_level in levels
    }


def has_any_level(levels, allowed_levels):
    return bool(set(levels).intersection(allowed_levels))


def is_dt_only_cell(levels):
    return bool(levels) and set(levels).issubset(DT_ONLY_NETWORK_LEVELS)


def edge_synapse_count(graph, source_cell, target_cell):
    try:
        return int(graph[source_cell][target_cell].get("syn_count", 1))
    except Exception:
        return 1


def count_edges_between_sets(graph, source_cells, target_cells):
    return [
        (source_cell, target_cell)
        for source_cell, target_cell in graph.edges()
        if source_cell in source_cells and target_cell in target_cells
    ]


def count_synapses_for_edges(graph, edges):
    return sum(
        edge_synapse_count(graph, source_cell, target_cell)
        for source_cell, target_cell in edges
    )


def compute_percent(count, total):
    if not total:
        return 0.0

    return count / total * 100.0


def make_row(section, subgroup, label, count, total=None, percent=None, notes=""):
    if total is None:
        total = ""

    if percent is None:
        percent = ""

    return {
        "section": section,
        "subgroup": subgroup,
        "label": label,
        "count": count,
        "total": total,
        "percent": percent,
        "notes": notes,
    }


def format_count_line(label, count, total=None, indent="  "):
    if total is None:
        return f"{indent}{label}: {count}"

    percent = compute_percent(count, total)
    return f"{indent}{label:>8}: {count:4d} / {total:<4d} ({percent:5.1f}%)"


def add_count_rows(rows, section, subgroup, counts, total, label_order, notes=""):
    for label in label_order:
        count = counts.get(label, 0)
        rows.append(
            make_row(
                section=section,
                subgroup=subgroup,
                label=label,
                count=count,
                total=total,
                percent=round(compute_percent(count, total), 3),
                notes=notes,
            )
        )


def normalized_class_label(classifier):
    label = class_label(classifier)

    if label.startswith("SPN"):
        return "SPN"

    return label


def normalized_panel_i_target_label(classifier):
    raw_classifier = str(classifier).strip().lower()

    if raw_classifier == "other":
        return "O"

    label = normalized_class_label(classifier)

    if label == "P":
        return "O"

    return label



def panel_i_target_label(classifier):
    return normalized_panel_i_target_label(classifier)


def panel_e_group_label(classifier):
    label = class_label(classifier)

    if label == "L-2":
        return "II"

    if label in {"I", "II"}:
        return label

    return None


def compute_class_counts_for_edges(graph, edges, use_source=True, target_label_function=None):
    counts = Counter()

    for source_cell, target_cell in edges:
        cell_id = source_cell if use_source else target_cell
        classifier = graph.nodes[cell_id].get(CLASSIFIER)

        if target_label_function is None:
            label = normalized_class_label(classifier)
        else:
            label = target_label_function(classifier)

        if label is None:
            continue

        counts[label] += 1

    return counts


def compute_neurotransmitter_counts_for_edges(graph, edges, use_source=True):
    return Counter(
        neurotransmitter_category(
            graph.nodes[source_cell if use_source else target_cell].get(NEUROTRANSMITTER)
        )
        for source_cell, target_cell in edges
    )


def compute_synapse_counts_by_class(graph, edges, use_source=True, target_label_function=None):
    counts = Counter()

    for source_cell, target_cell in edges:
        cell_id = source_cell if use_source else target_cell
        classifier = graph.nodes[cell_id].get(CLASSIFIER)

        if target_label_function is None:
            label = normalized_class_label(classifier)
        else:
            label = target_label_function(classifier)

        if label is None:
            continue

        counts[label] += edge_synapse_count(graph, source_cell, target_cell)

    return counts


def compute_synapse_counts_by_neurotransmitter(graph, edges, use_source=True):
    counts = Counter()

    for source_cell, target_cell in edges:
        cell_id = source_cell if use_source else target_cell
        neurotransmitter = neurotransmitter_category(
            graph.nodes[cell_id].get(NEUROTRANSMITTER)
        )
        counts[neurotransmitter] += edge_synapse_count(graph, source_cell, target_cell)

    return counts


def compute_neurotransmitter_counts_by_target_class(graph, edges, target_label_function):
    counts_by_target_class = {}

    for source_cell, target_cell in edges:
        target_label = target_label_function(graph.nodes[target_cell].get(CLASSIFIER))

        if target_label is None:
            continue

        neurotransmitter = neurotransmitter_category(
            graph.nodes[source_cell].get(NEUROTRANSMITTER)
        )
        counts_by_target_class.setdefault(target_label, Counter())
        counts_by_target_class[target_label][neurotransmitter] += 1

    return counts_by_target_class


def compute_synapse_counts_by_target_class_and_neurotransmitter(
    graph,
    edges,
    target_label_function,
):
    counts_by_target_class = {}

    for source_cell, target_cell in edges:
        target_label = target_label_function(graph.nodes[target_cell].get(CLASSIFIER))

        if target_label is None:
            continue

        neurotransmitter = neurotransmitter_category(
            graph.nodes[source_cell].get(NEUROTRANSMITTER)
        )
        counts_by_target_class.setdefault(target_label, Counter())
        counts_by_target_class[target_label][neurotransmitter] += edge_synapse_count(
            graph,
            source_cell,
            target_cell,
        )

    return counts_by_target_class


def compute_panel_c_summary(graph, levels_by_cell):
    input_one_cells = get_cells_with_level(levels_by_cell, "input_one")
    seed_cells = get_cells_with_level(levels_by_cell, "network_seed")
    output_one_cells = get_cells_with_level(levels_by_cell, "output_one")
    output_two_cells = get_cells_with_level(levels_by_cell, "output_two")

    layer_cells = {
        "input_one": input_one_cells,
        "network_seed": seed_cells,
        "output_one": output_one_cells,
        "output_two": output_two_cells,
    }

    all_reported_layer_cells = set().union(*layer_cells.values())

    input_to_seed_edges = count_edges_between_sets(
        graph,
        input_one_cells,
        seed_cells,
    )
    seed_to_output_edges = count_edges_between_sets(
        graph,
        seed_cells,
        output_one_cells,
    )
    output_to_output_two_edges = count_edges_between_sets(
        graph,
        output_one_cells,
        output_two_cells,
    )

    inter_layer_edges = {
        "input_one_to_network_seed": input_to_seed_edges,
        "network_seed_to_output_one": seed_to_output_edges,
        "output_one_to_output_two": output_to_output_two_edges,
    }

    rows = [
        make_row(
            section="panel_c_reconstruction_levels",
            subgroup="cells",
            label="unique_cells_across_reported_layers",
            count=len(all_reported_layer_cells),
        )
    ]

    for label in PANEL_C_LAYER_ORDER:
        count = len(layer_cells[label])
        rows.append(
            make_row(
                section="panel_c_reconstruction_levels",
                subgroup="cells",
                label=label,
                count=count,
                total=len(all_reported_layer_cells),
                percent=round(compute_percent(count, len(all_reported_layer_cells)), 3),
            )
        )

    for label, edges in inter_layer_edges.items():
        rows.append(
            make_row(
                section="panel_c_reconstruction_levels",
                subgroup="unique_edges",
                label=label,
                count=len(edges),
            )
        )
        rows.append(
            make_row(
                section="panel_c_reconstruction_levels",
                subgroup="synapse_contacts",
                label=label,
                count=count_synapses_for_edges(graph, edges),
            )
        )

    lines = [
        "Panel C — reconstruction-level counts",
        "-------------------------------------",
        "Logic: uses dataframe network_level annotations.",
        "Reports reconstruction stages: input_one → network_seed → output_one → output_two.",
        "Seed cells are counted as network_seed, even if they also connect to each other.",
        "",
        f"Unique cells across reported layers: {len(all_reported_layer_cells)}",
        "",
        "Layer cell counts:",
    ]

    for label in PANEL_C_LAYER_ORDER:
        lines.append(format_count_line(label, len(layer_cells[label])))

    lines.extend(
        [
            "",
            "Inter-layer unique edge counts:",
            format_count_line("input_one → network_seed", len(input_to_seed_edges)),
            format_count_line("network_seed → output_one", len(seed_to_output_edges)),
            format_count_line("output_one → output_two", len(output_to_output_two_edges)),
            "",
            "Inter-layer synapse-contact counts:",
            format_count_line(
                "input_one → network_seed",
                count_synapses_for_edges(graph, input_to_seed_edges),
            ),
            format_count_line(
                "network_seed → output_one",
                count_synapses_for_edges(graph, seed_to_output_edges),
            ),
            format_count_line(
                "output_one → output_two",
                count_synapses_for_edges(graph, output_to_output_two_edges),
            ),
        ]
    )

    return rows, "\n".join(lines)


def compute_recurrence_summary(levels_by_cell):
    input_one_cells = get_cells_with_level(levels_by_cell, "input_one")
    seed_cells = get_cells_with_level(levels_by_cell, "network_seed")
    output_one_cells = get_cells_with_level(levels_by_cell, "output_one")
    output_two_cells = get_cells_with_level(levels_by_cell, "output_two")

    first_three_layer_cells = input_one_cells | seed_cells | output_one_cells
    rediscovered_output_two_cells = output_two_cells & first_three_layer_cells
    newly_discovered_output_two_cells = output_two_cells - first_three_layer_cells

    denominator = len(first_three_layer_cells)
    rediscovered_fraction = compute_percent(
        len(rediscovered_output_two_cells),
        denominator,
    )

    rows = [
        make_row(
            section="recurrence_rediscovery",
            subgroup="cells",
            label="input_one_network_seed_output_one_unique_cells",
            count=len(first_three_layer_cells),
        ),
        make_row(
            section="recurrence_rediscovery",
            subgroup="cells",
            label="output_two_cells",
            count=len(output_two_cells),
        ),
        make_row(
            section="recurrence_rediscovery",
            subgroup="cells",
            label="output_two_already_in_input_seed_or_output_one",
            count=len(rediscovered_output_two_cells),
            total=len(output_two_cells),
            percent=round(compute_percent(len(rediscovered_output_two_cells), len(output_two_cells)), 3),
            notes=";".join(str(cell_id) for cell_id in sorted(rediscovered_output_two_cells)),
        ),
        make_row(
            section="recurrence_rediscovery",
            subgroup="cells",
            label="output_two_newly_discovered",
            count=len(newly_discovered_output_two_cells),
            total=len(output_two_cells),
            percent=round(compute_percent(len(newly_discovered_output_two_cells), len(output_two_cells)), 3),
            notes=";".join(str(cell_id) for cell_id in sorted(newly_discovered_output_two_cells)),
        ),
        make_row(
            section="recurrence_rediscovery",
            subgroup="fraction",
            label="rediscovered_output_two_over_first_three_layers",
            count=len(rediscovered_output_two_cells),
            total=denominator,
            percent=round(rediscovered_fraction, 3),
        ),
    ]

    lines = [
        "Recurrence / rediscovery statistics",
        "-----------------------------------",
        "Logic: uses dataframe network_level annotations.",
        "Rediscovered output_two cells are output_two cells that also appear in input_one, network_seed, or output_one.",
        "",
        f"Unique cells in input_one + network_seed + output_one: {len(first_three_layer_cells)}",
        f"Output_two cells: {len(output_two_cells)}",
        f"Output_two cells already present in earlier layers: {len(rediscovered_output_two_cells)}",
        f"Output_two cells newly discovered: {len(newly_discovered_output_two_cells)}",
        f"Rediscovered fraction over input_one + network_seed + output_one: "
        f"{len(rediscovered_output_two_cells)} / {denominator} ({rediscovered_fraction:.1f}%)",
    ]

    return rows, "\n".join(lines)


def compute_panel_e_summary(filtered_graph):
    rows = []

    lines = [
        "Panel E — neurotransmitter distribution",
        "---------------------------------------",
        "Logic: uses the filtered figure graph only.",
        "Filtered graph = input_one + network_seed + output_one.",
        "Reports neurotransmitter categories for Class I and Class II cells.",
        "L-2 cells are counted together with Class II.",
    ]

    for target_group in PANEL_E_GROUP_ORDER:
        cell_ids = {
            cell_id
            for cell_id, attributes in filtered_graph.nodes(data=True)
            if panel_e_group_label(attributes.get(CLASSIFIER)) == target_group
        }

        counts = Counter(
            neurotransmitter_category(filtered_graph.nodes[cell_id].get(NEUROTRANSMITTER))
            for cell_id in cell_ids
        )

        total = len(cell_ids)
        group_name = "Class II" if target_group == "II" else "Class I"

        rows.append(
            make_row(
                section="panel_e_neurotransmitter_distribution",
                subgroup=group_name,
                label="total_cells",
                count=total,
            )
        )

        add_count_rows(
            rows=rows,
            section="panel_e_neurotransmitter_distribution",
            subgroup=group_name,
            counts=counts,
            total=total,
            label_order=NEUROTRANSMITTER_ORDER,
        )

        lines.extend(["", f"{group_name} cells: n = {total}"])

        for label in NEUROTRANSMITTER_ORDER:
            lines.append(format_count_line(label, counts.get(label, 0), total))

    return rows, "\n".join(lines)


def compute_panel_h_seed_input_summary(filtered_graph):
    seed_cells = {
        cell_id
        for cell_id in SEED_CELLS
        if filtered_graph.has_node(cell_id)
    }

    input_edges = [
        (source_cell, target_cell)
        for source_cell, target_cell in filtered_graph.edges()
        if target_cell in seed_cells
    ]

    source_cells = {
        source_cell
        for source_cell, _ in input_edges
    }

    represented_cells = seed_cells | source_cells
    total_edges = len(input_edges)
    total_synapses = count_synapses_for_edges(filtered_graph, input_edges)

    source_class_counts = compute_class_counts_for_edges(
        filtered_graph,
        input_edges,
        use_source=True,
    )
    source_class_synapse_counts = compute_synapse_counts_by_class(
        filtered_graph,
        input_edges,
        use_source=True,
    )
    source_neurotransmitter_counts = compute_neurotransmitter_counts_for_edges(
        filtered_graph,
        input_edges,
        use_source=True,
    )
    source_neurotransmitter_synapse_counts = compute_synapse_counts_by_neurotransmitter(
        filtered_graph,
        input_edges,
        use_source=True,
    )

    rows = [
        make_row(
            section="panel_h_seed_inputs",
            subgroup="summary",
            label="seed_cells_present_in_filtered_graph",
            count=len(seed_cells),
        ),
        make_row(
            section="panel_h_seed_inputs",
            subgroup="summary",
            label="unique_presynaptic_cells",
            count=len(source_cells),
        ),
        make_row(
            section="panel_h_seed_inputs",
            subgroup="summary",
            label="seed_cells_plus_presynaptic_cells",
            count=len(represented_cells),
        ),
        make_row(
            section="panel_h_seed_inputs",
            subgroup="summary",
            label="unique_input_edges_to_seed_cells",
            count=total_edges,
        ),
        make_row(
            section="panel_h_seed_inputs",
            subgroup="summary",
            label="input_synapse_contacts_to_seed_cells",
            count=total_synapses,
        ),
    ]

    add_count_rows(
        rows=rows,
        section="panel_h_seed_inputs",
        subgroup="presynaptic_class_unique_edges",
        counts=source_class_counts,
        total=total_edges,
        label_order=PANEL_H_SOURCE_CLASS_ORDER,
    )
    add_count_rows(
        rows=rows,
        section="panel_h_seed_inputs",
        subgroup="presynaptic_class_synapse_contacts",
        counts=source_class_synapse_counts,
        total=total_synapses,
        label_order=PANEL_H_SOURCE_CLASS_ORDER,
    )
    add_count_rows(
        rows=rows,
        section="panel_h_seed_inputs",
        subgroup="presynaptic_neurotransmitter_unique_edges",
        counts=source_neurotransmitter_counts,
        total=total_edges,
        label_order=NEUROTRANSMITTER_ORDER,
    )
    add_count_rows(
        rows=rows,
        section="panel_h_seed_inputs",
        subgroup="presynaptic_neurotransmitter_synapse_contacts",
        counts=source_neurotransmitter_synapse_counts,
        total=total_synapses,
        label_order=NEUROTRANSMITTER_ORDER,
    )

    lines = [
        "Panel H — inputs to the four seed cells",
        "---------------------------------------",
        "Logic: uses the filtered figure graph only.",
        "Filtered graph = input_one + network_seed + output_one.",
        "Counts direct predecessors of the four seed cells inside this filtered graph.",
        f"Seed cells present in filtered graph: n = {len(seed_cells)}",
        f"Unique presynaptic cells: {len(source_cells)}",
        f"Seed cells + presynaptic cells represented: {len(represented_cells)}",
        f"Input edges to seed cells: {total_edges}",
        f"Input synapse contacts to seed cells: {total_synapses}",
        "",
        "Input edges by presynaptic class:",
    ]

    for label in PANEL_H_SOURCE_CLASS_ORDER:
        lines.append(
            format_count_line(label, source_class_counts.get(label, 0), total_edges)
        )

    lines.extend(["", "Input synapse contacts by presynaptic class:"])

    for label in PANEL_H_SOURCE_CLASS_ORDER:
        lines.append(
            format_count_line(
                label,
                source_class_synapse_counts.get(label, 0),
                total_synapses,
            )
        )

    lines.extend(["", "Input edges by presynaptic neurotransmitter:"])

    for label in NEUROTRANSMITTER_ORDER:
        lines.append(
            format_count_line(
                label,
                source_neurotransmitter_counts.get(label, 0),
                total_edges,
            )
        )

    lines.extend(["", "Input synapse contacts by presynaptic neurotransmitter:"])

    for label in NEUROTRANSMITTER_ORDER:
        lines.append(
            format_count_line(
                label,
                source_neurotransmitter_synapse_counts.get(label, 0),
                total_synapses,
            )
        )

    return rows, "\n".join(lines)




def compute_panel_i_summary(panel_i_graph, levels_by_cell):
    source_cells = {
        cell_id
        for cell_id, attributes in panel_i_graph.nodes(data=True)
        if class_label(attributes.get(CLASSIFIER)) == "I"
        and (
            "network_seed" in levels_by_cell.get(cell_id, [])
            or "output_one" in levels_by_cell.get(cell_id, [])
        )
    }

    output_edges = [
        (source_cell, target_cell)
        for source_cell, target_cell in panel_i_graph.edges()
        if source_cell in source_cells
    ]

    p_target_edges = [
        (source_cell, target_cell)
        for source_cell, target_cell in output_edges
        if normalized_class_label(panel_i_graph.nodes[target_cell].get(CLASSIFIER)) == "P"
    ]

    literal_other_target_edges = [
        (source_cell, target_cell)
        for source_cell, target_cell in output_edges
        if str(panel_i_graph.nodes[target_cell].get(CLASSIFIER)).strip().lower() == "other"
    ]

    target_cells = {
        target_cell
        for _, target_cell in output_edges
    }

    represented_cells = source_cells | target_cells
    total_edges = len(output_edges)
    total_synapses = count_synapses_for_edges(panel_i_graph, output_edges)

    target_class_counts = compute_class_counts_for_edges(
        panel_i_graph,
        output_edges,
        use_source=False,
        target_label_function=panel_i_target_label,
    )
    target_class_synapse_counts = compute_synapse_counts_by_class(
        panel_i_graph,
        output_edges,
        use_source=False,
        target_label_function=panel_i_target_label,
    )
    source_neurotransmitter_counts = compute_neurotransmitter_counts_for_edges(
        panel_i_graph,
        output_edges,
        use_source=True,
    )
    source_neurotransmitter_synapse_counts = compute_synapse_counts_by_neurotransmitter(
        panel_i_graph,
        output_edges,
        use_source=True,
    )

    source_neurotransmitter_counts_by_target_class = (
        compute_neurotransmitter_counts_by_target_class(
            panel_i_graph,
            output_edges,
            panel_i_target_label,
        )
    )
    source_neurotransmitter_synapse_counts_by_target_class = (
        compute_synapse_counts_by_target_class_and_neurotransmitter(
            panel_i_graph,
            output_edges,
            panel_i_target_label,
        )
    )

    rows = [
        make_row(
            section="panel_i_outputs",
            subgroup="summary",
            label="class_i_network_seed_or_output_one_source_cells",
            count=len(source_cells),
        ),
        make_row(
            section="panel_i_outputs",
            subgroup="summary",
            label="unique_postsynaptic_cells",
            count=len(target_cells),
        ),
        make_row(
            section="panel_i_outputs",
            subgroup="summary",
            label="source_cells_plus_postsynaptic_cells",
            count=len(represented_cells),
        ),
        make_row(
            section="panel_i_outputs",
            subgroup="summary",
            label="unique_output_edges_from_class_i_seed_or_output_one_cells",
            count=total_edges,
        ),
        make_row(
            section="panel_i_outputs",
            subgroup="summary",
            label="output_synapse_contacts_from_class_i_seed_or_output_one_cells",
            count=total_synapses,
        ),
        make_row(
            section="panel_i_outputs",
            subgroup="summary",
            label="p_target_edges_collapsed_into_other",
            count=len(p_target_edges),
        ),
        make_row(
            section="panel_i_outputs",
            subgroup="summary",
            label="literal_other_target_edges_collapsed_into_O",
            count=len(literal_other_target_edges),
        ),
    ]

    add_count_rows(
        rows=rows,
        section="panel_i_outputs",
        subgroup="postsynaptic_class_unique_edges",
        counts=target_class_counts,
        total=total_edges,
        label_order=PANEL_I_TARGET_CLASS_ORDER,
    )
    add_count_rows(
        rows=rows,
        section="panel_i_outputs",
        subgroup="postsynaptic_class_synapse_contacts",
        counts=target_class_synapse_counts,
        total=total_synapses,
        label_order=PANEL_I_TARGET_CLASS_ORDER,
    )
    add_count_rows(
        rows=rows,
        section="panel_i_outputs",
        subgroup="source_neurotransmitter_unique_edges",
        counts=source_neurotransmitter_counts,
        total=total_edges,
        label_order=NEUROTRANSMITTER_ORDER,
    )
    add_count_rows(
        rows=rows,
        section="panel_i_outputs",
        subgroup="source_neurotransmitter_synapse_contacts",
        counts=source_neurotransmitter_synapse_counts,
        total=total_synapses,
        label_order=NEUROTRANSMITTER_ORDER,
    )

    for target_class in PANEL_I_TARGET_CLASS_ORDER:
        target_class_edges = target_class_counts.get(target_class, 0)
        target_class_synapses = target_class_synapse_counts.get(target_class, 0)

        rows.append(
            make_row(
                section="panel_i_outputs",
                subgroup="postsynaptic_class_by_source_neurotransmitter_unique_edges",
                label=f"{target_class}_total_edges",
                count=target_class_edges,
                total=total_edges,
                percent=round(compute_percent(target_class_edges, total_edges), 3),
            )
        )
        add_count_rows(
            rows=rows,
            section="panel_i_outputs",
            subgroup=f"{target_class}_source_neurotransmitter_unique_edges",
            counts=source_neurotransmitter_counts_by_target_class.get(
                target_class,
                Counter(),
            ),
            total=target_class_edges,
            label_order=NEUROTRANSMITTER_ORDER,
        )

        rows.append(
            make_row(
                section="panel_i_outputs",
                subgroup="postsynaptic_class_by_source_neurotransmitter_synapse_contacts",
                label=f"{target_class}_total_synapses",
                count=target_class_synapses,
                total=total_synapses,
                percent=round(compute_percent(target_class_synapses, total_synapses), 3),
            )
        )
        add_count_rows(
            rows=rows,
            section="panel_i_outputs",
            subgroup=f"{target_class}_source_neurotransmitter_synapse_contacts",
            counts=source_neurotransmitter_synapse_counts_by_target_class.get(
                target_class,
                Counter(),
            ),
            total=target_class_synapses,
            label_order=NEUROTRANSMITTER_ORDER,
        )

    lines = [
        "Panel I — outputs from Class I seed/output_one cells",
        "----------------------------------------------------",
        "Logic: uses the Panel I graph only.",
        "Panel I graph = input_one + network_seed + output_one + output_two.",
        "Counts direct successors of Class I cells annotated as network_seed or output_one.",
        "P targets and literal classifier='other' targets are collapsed into O.",
        "? contains targets without a recognized classifier.",
        "SPN subclasses are collapsed into SPN.",
        f"Class I network_seed/output_one source cells: n = {len(source_cells)}",
        f"Unique postsynaptic cells: {len(target_cells)}",
        f"Source cells + postsynaptic cells represented: {len(represented_cells)}",
        f"Output edges from Class I network_seed/output_one cells: {total_edges}",
        f"Output synapse contacts from Class I network_seed/output_one cells: {total_synapses}",
        f"P-target edges collapsed into O: {len(p_target_edges)}",
        f"Literal other-target edges collapsed into O: {len(literal_other_target_edges)}",
        "",
        "Output edges by postsynaptic class:",
    ]

    for label in PANEL_I_TARGET_CLASS_ORDER:
        lines.append(
            format_count_line(label, target_class_counts.get(label, 0), total_edges)
        )

    lines.extend(["", "Output synapse contacts by postsynaptic class:"])

    for label in PANEL_I_TARGET_CLASS_ORDER:
        lines.append(
            format_count_line(
                label,
                target_class_synapse_counts.get(label, 0),
                total_synapses,
            )
        )

    lines.extend(["", "Output edges by source neurotransmitter:"])

    for label in NEUROTRANSMITTER_ORDER:
        lines.append(
            format_count_line(
                label,
                source_neurotransmitter_counts.get(label, 0),
                total_edges,
            )
        )

    lines.extend(["", "Output synapse contacts by source neurotransmitter:"])

    for label in NEUROTRANSMITTER_ORDER:
        lines.append(
            format_count_line(
                label,
                source_neurotransmitter_synapse_counts.get(label, 0),
                total_synapses,
            )
        )

    lines.extend(["", "Source neurotransmitter breakdown by postsynaptic class:"])

    for target_class in PANEL_I_TARGET_CLASS_ORDER:
        target_class_edges = target_class_counts.get(target_class, 0)
        target_class_synapses = target_class_synapse_counts.get(target_class, 0)

        if target_class_edges == 0 and target_class_synapses == 0:
            continue

        lines.append(f"  I → {target_class}: {target_class_edges} edges, {target_class_synapses} synapse contacts")
        lines.append("    Unique edges:")

        for neurotransmitter in NEUROTRANSMITTER_ORDER:
            lines.append(
                format_count_line(
                    neurotransmitter,
                    source_neurotransmitter_counts_by_target_class.get(
                        target_class,
                        Counter(),
                    ).get(neurotransmitter, 0),
                    target_class_edges,
                    indent="      ",
                )
            )

        lines.append("    Synapse contacts:")

        for neurotransmitter in NEUROTRANSMITTER_ORDER:
            lines.append(
                format_count_line(
                    neurotransmitter,
                    source_neurotransmitter_synapse_counts_by_target_class.get(
                        target_class,
                        Counter(),
                    ).get(neurotransmitter, 0),
                    target_class_synapses,
                    indent="      ",
                )
            )

    return rows, "\n".join(lines)


def write_summary_csv(rows, path):
    fieldnames = [
        "section",
        "subgroup",
        "label",
        "count",
        "total",
        "percent",
        "notes",
    ]

    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()