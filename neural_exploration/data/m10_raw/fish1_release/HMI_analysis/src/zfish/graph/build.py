"""Graph construction utilities for zebrafish EM connectivity data."""

import ast

import networkx as nx
import pandas as pd


CELL_ID = "Cell ID"
INPUTS = "inputs"
OUTPUTS = "outputs"

EMPTY_VALUES = {"", "n/a", "na", "nan", "none", "[]", "[ ]", "{}", "()"}


def build_graph(
    dataframe,
    include_inputs=True,
    include_outputs=True,
    allow_self_loops=True,
    error_mode="silent",
):
    """Build a directed synaptic graph from the project dataframe."""
    validate_error_mode(error_mode)
    validate_required_columns(dataframe, include_inputs, include_outputs)

    graph = nx.DiGraph()
    build_stats = initialize_build_stats(dataframe)
    tracing_stats = initialize_tracing_stats()

    add_dataframe_cells_as_nodes(graph, dataframe)

    for _, row in dataframe.iterrows():
        cell_id = parse_cell_id(row.get(CELL_ID))

        if cell_id is None:
            continue

        if not graph.has_node(cell_id):
            graph.add_node(cell_id)

        graph.nodes[cell_id]["inputs_tracing_attempted"] = False
        graph.nodes[cell_id]["outputs_tracing_attempted"] = False

        if include_inputs:
            process_synapse_column(
                graph=graph,
                row=row,
                cell_id=cell_id,
                column=INPUTS,
                direction="inputs",
                allow_self_loops=allow_self_loops,
                error_mode=error_mode,
                build_stats=build_stats,
                tracing_stats=tracing_stats,
            )

        if include_outputs:
            process_synapse_column(
                graph=graph,
                row=row,
                cell_id=cell_id,
                column=OUTPUTS,
                direction="outputs",
                allow_self_loops=allow_self_loops,
                error_mode=error_mode,
                build_stats=build_stats,
                tracing_stats=tracing_stats,
            )

    graph.graph["build_stats"] = build_stats
    graph.graph["tracing_stats"] = finalize_tracing_stats(tracing_stats)

    if error_mode == "count":
        print("[zfish.build_graph]", graph.graph["build_stats"])

    return graph


def validate_error_mode(error_mode):
    allowed_modes = {"silent", "count", "raise"}

    if error_mode not in allowed_modes:
        raise ValueError(f"error_mode must be one of {sorted(allowed_modes)}")


def validate_required_columns(dataframe, include_inputs, include_outputs):
    required_columns = {CELL_ID}

    if include_inputs:
        required_columns.add(INPUTS)

    if include_outputs:
        required_columns.add(OUTPUTS)

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise KeyError(f"Missing required columns: {sorted(missing_columns)}")


def initialize_build_stats(dataframe):
    return {
        "n_rows": int(len(dataframe)),
        "parse_fail_rows": 0,
        "nonlist_rows": 0,
        "skipped_tuples": 0,
        "kept_contacts": 0,
        "store_xyz": False,
    }


def initialize_tracing_stats():
    return {
        "inputs_total": 0,
        "inputs_identified": 0,
        "inputs_unidentified": 0,
        "inputs_malformed": 0,
        "outputs_total": 0,
        "outputs_identified": 0,
        "outputs_unidentified": 0,
        "outputs_malformed": 0,
    }


def add_dataframe_cells_as_nodes(graph, dataframe):
    for value in dataframe[CELL_ID]:
        cell_id = parse_cell_id(value)

        if cell_id is not None:
            graph.add_node(cell_id)


def parse_cell_id(value):
    try:
        if pd.isna(value):
            return None

        return int(value)

    except Exception:
        return None


def process_synapse_column(
    graph,
    row,
    cell_id,
    column,
    direction,
    allow_self_loops,
    error_mode,
    build_stats,
    tracing_stats,
):
    value = row.get(column, "n/a")

    if is_empty_synapse_value(value):
        return

    graph.nodes[cell_id][f"{direction}_tracing_attempted"] = True

    synapses, parsed_successfully = parse_synapse_list(value)

    if not parsed_successfully:
        build_stats["parse_fail_rows"] += 1

        if error_mode == "raise":
            raise ValueError(f"Failed parsing {column} for cell {cell_id}")

        return

    if not isinstance(synapses, list):
        build_stats["nonlist_rows"] += 1
        return

    contact_stats = process_synapses(
        graph=graph,
        synapses=synapses,
        cell_id=cell_id,
        direction=direction,
        allow_self_loops=allow_self_loops,
    )

    build_stats["skipped_tuples"] += contact_stats["skipped"]
    build_stats["kept_contacts"] += contact_stats["identified"]

    store_node_tracing_stats(graph, cell_id, direction, contact_stats)
    update_global_tracing_stats(tracing_stats, direction, contact_stats)


def is_empty_synapse_value(value):
    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except Exception:
        pass

    if isinstance(value, str) and value.strip().lower() in EMPTY_VALUES:
        return True

    return False


def parse_synapse_list(value):
    if isinstance(value, list):
        return value, True

    if not isinstance(value, str):
        return value, True

    try:
        return ast.literal_eval(value), True
    except Exception:
        return None, False


def process_synapses(graph, synapses, cell_id, direction, allow_self_loops):
    contact_stats = {
        "total": 0,
        "identified": 0,
        "unidentified": 0,
        "malformed": 0,
        "skipped": 0,
    }

    for synapse in synapses:
        connected_cell = parse_connected_cell_id(synapse)

        if connected_cell == "malformed":
            contact_stats["malformed"] += 1
            contact_stats["skipped"] += 1
            continue

        contact_stats["total"] += 1

        if connected_cell is None:
            contact_stats["unidentified"] += 1
            contact_stats["skipped"] += 1
            continue

        source_cell, target_cell = get_edge_direction(
            cell_id,
            connected_cell,
            direction,
        )

        if not allow_self_loops and source_cell == target_cell:
            contact_stats["malformed"] += 1
            contact_stats["skipped"] += 1
            continue

        add_synapse_contact(graph, source_cell, target_cell)
        contact_stats["identified"] += 1

    return contact_stats


def parse_connected_cell_id(synapse):
    if not synapse:
        return "malformed"

    try:
        connected_cell = synapse[0]

        if len(synapse) < 4:  # require x/y/z coordinates
            return "malformed"

    except Exception:
        return "malformed"

    if connected_cell == "-":
        return None

    try:
        if pd.isna(connected_cell):
            return None
    except Exception:
        pass

    try:
        return int(connected_cell)
    except Exception:
        return "malformed"


def get_edge_direction(cell_id, connected_cell, direction):
    if direction == "inputs":
        return connected_cell, cell_id

    if direction == "outputs":
        return cell_id, connected_cell

    raise ValueError("direction must be 'inputs' or 'outputs'")


def add_synapse_contact(graph, source_cell, target_cell):
    if graph.has_edge(source_cell, target_cell):
        synapse_count = int(graph[source_cell][target_cell].get("syn_count", 0)) + 1
        graph[source_cell][target_cell]["syn_count"] = synapse_count
        graph[source_cell][target_cell]["multisynapse"] = synapse_count > 1
        return

    graph.add_edge(
        source_cell,
        target_cell,
        syn_count=1,
        multisynapse=False,
    )


def store_node_tracing_stats(graph, cell_id, direction, contact_stats):
    total_contacts = contact_stats["total"]
    identified_contacts = contact_stats["identified"]

    graph.nodes[cell_id][f"{direction}_traced_contacts"] = total_contacts
    graph.nodes[cell_id][f"{direction}_identified_contacts"] = identified_contacts
    graph.nodes[cell_id][f"{direction}_unidentified_contacts"] = contact_stats["unidentified"]
    graph.nodes[cell_id][f"{direction}_malformed_contacts"] = contact_stats["malformed"]
    graph.nodes[cell_id][f"{direction}_id_rate"] = safe_rate(
        identified_contacts,
        total_contacts,
    )


def update_global_tracing_stats(tracing_stats, direction, contact_stats):
    tracing_stats[f"{direction}_total"] += contact_stats["total"]
    tracing_stats[f"{direction}_identified"] += contact_stats["identified"]
    tracing_stats[f"{direction}_unidentified"] += contact_stats["unidentified"]
    tracing_stats[f"{direction}_malformed"] += contact_stats["malformed"]


def finalize_tracing_stats(tracing_stats):
    tracing_stats = tracing_stats.copy()

    tracing_stats["inputs_id_rate"] = safe_rate(
        tracing_stats["inputs_identified"],
        tracing_stats["inputs_total"],
    )

    tracing_stats["outputs_id_rate"] = safe_rate(
        tracing_stats["outputs_identified"],
        tracing_stats["outputs_total"],
    )

    return tracing_stats


def safe_rate(numerator, denominator):
    if not denominator:
        return None

    return float(numerator) / float(denominator)