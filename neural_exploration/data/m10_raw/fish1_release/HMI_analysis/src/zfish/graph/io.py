"""Input/output utilities for zebrafish EM graph analyses."""

from pathlib import Path

import pandas as pd


CELL_ID = "Cell ID"


def load_df(
    path,
    sheet_name=0,
    cell_id_column=CELL_ID,
    required_columns=None,
):
    """Load the project dataframe and keep rows with valid cell IDs."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input dataframe not found: {path}")

    dataframe = pd.read_excel(path, sheet_name=sheet_name)
    validate_dataframe_columns(dataframe, cell_id_column, required_columns)

    dataframe = dataframe.copy()
    dataframe[cell_id_column] = pd.to_numeric(dataframe[cell_id_column], errors="coerce")
    dataframe = dataframe.dropna(subset=[cell_id_column]).copy()
    dataframe[cell_id_column] = dataframe[cell_id_column].astype(int)

    return dataframe


def validate_dataframe_columns(dataframe, cell_id_column, required_columns):
    required_columns = set(required_columns or [])
    required_columns.add(cell_id_column)

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise KeyError(f"Missing required dataframe columns: {sorted(missing_columns)}")