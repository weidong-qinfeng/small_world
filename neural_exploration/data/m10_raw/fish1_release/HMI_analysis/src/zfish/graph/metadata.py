"""Node metadata utilities for zebrafish EM connectivity graphs."""

import ast

import networkx as nx
import pandas as pd


CELL_ID = "Cell ID"

DEFAULT_METADATA_COLUMNS = [
    "final_neurotransmitter_ID",
    "final_prediction",
    "original_prediction",
    "classifier",
    "soma_position",
    "network_level",
]


def add_neural_metadata(
    graph,
    dataframe,
    columns=None,
    missing_value="n/a",
):
    """Attach dataframe metadata columns as node attributes."""
    if columns is None:
        columns = DEFAULT_METADATA_COLUMNS

    if CELL_ID not in dataframe.columns:
        raise KeyError(f"Expected column '{CELL_ID}' not found in dataframe.")

    dataframe_by_cell = make_cell_id_index(dataframe)
    node_attributes = {}

    for cell_id in graph.nodes:
        if cell_id in dataframe_by_cell.index:
            row = dataframe_by_cell.loc[cell_id]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            node_attributes[cell_id] = get_metadata_from_row(
                row,
                columns,
                dataframe.columns,
                missing_value,
            )
            continue

        node_attributes[cell_id] = {
            column: missing_value
            for column in columns
        }

    nx.set_node_attributes(graph, node_attributes)

    return graph


def add_computed_graph_metadata(graph):
    """Attach graph-derived node metrics as node attributes."""
    nx.set_node_attributes(graph, dict(graph.in_degree()), "in_degree")
    nx.set_node_attributes(graph, dict(graph.out_degree()), "out_degree")
    nx.set_node_attributes(graph, dict(graph.degree()), "total_degree")

    return graph


def make_cell_id_index(dataframe):
    dataframe = dataframe.copy()
    dataframe[CELL_ID] = pd.to_numeric(dataframe[CELL_ID], errors="coerce")
    dataframe = dataframe.dropna(subset=[CELL_ID]).copy()
    dataframe[CELL_ID] = dataframe[CELL_ID].astype(int)

    return dataframe.set_index(CELL_ID, drop=False)


def get_metadata_from_row(row, columns, dataframe_columns, missing_value):
    attributes = {}

    for column in columns:
        if column not in dataframe_columns:
            attributes[column] = missing_value
            continue

        attributes[column] = clean_metadata_value(
            row[column],
            column,
            missing_value,
        )

    return attributes


def clean_metadata_value(value, column, missing_value):
    if is_missing(value):
        return missing_value

    if column == "network_level":
        return parse_network_level(value, missing_value)

    return value


def parse_network_level(value, missing_value):
    if isinstance(value, (list, tuple, set)):
        return list(value)

    if not isinstance(value, str):
        return missing_value

    value = value.strip()

    if value.lower() in {"", "n/a", "na", "nan", "none"}:
        return missing_value

    try:
        parsed_value = ast.literal_eval(value)
    except Exception:
        return value

    if isinstance(parsed_value, (list, tuple, set)):
        return list(parsed_value)

    return parsed_value


def is_missing(value):
    if value is None:
        return True

    try:
        return pd.isna(value)
    except Exception:
        return False