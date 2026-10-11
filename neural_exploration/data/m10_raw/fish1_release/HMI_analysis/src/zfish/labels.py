"""Label normalization utilities for zebrafish EM analyses."""

import ast


MISSING_VALUES = {"", "n/a", "na", "nan", "none"}

CLASS_FILL_COLORS = {
    "I": "#B4F8FF",
    "II": "#FFCCCC",
    "L-2": "white",
    "SPN_turning": "#3F3F3F",
    "SPN_forward": "#BDBDBD",
    "SPN_other": "#9A9A9A",
    "?": "#a9a9a9",
}

CLASS_BORDER_COLORS = {
    "I": "#2BA5D7",
    "II": "#C1272D",
    "L-2": "#C1272D",
    "SPN_turning": "#000000",
    "SPN_forward": "#6A6A6A",
    "SPN_other": "#4A4A4A",
    "?": "black",
}

NEUROTRANSMITTER_COLORS = {
    "Exc": "#00DF00",
    "Inh": "#FF00FF",
    "Unknown": "#9E9E9E",
    "Unlabeled": "#9E9E9E",
}


def parse_network_levels(value):
    """Return network-level labels as a list."""
    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if not isinstance(value, str):
        return []

    value = value.strip()

    if value.lower() in MISSING_VALUES:
        return []

    try:
        parsed_value = ast.literal_eval(value)
    except Exception:
        return []

    if isinstance(parsed_value, list):
        return parsed_value

    if isinstance(parsed_value, tuple):
        return list(parsed_value)

    return []


def class_label(value):
    """Return the compact class label used across project analyses."""
    text = str(value).strip().lower()

    if "spn" in text:
        if text.endswith("_prox"):
            return "SPN_prox"

        if "turn" in text:
            return "SPN_turning"

        if "forw" in text:
            return "SPN_forward"

        return "SPN_other"

    if value == "L-2":
        return "L-2"

    if value in (1, "1", 1.0):
        return "I"

    if value in (2, "2", 2.0, "Class 2", "contralateral axon"):
        return "II"

    if value in (7, "7", 7.0):
        return "F"

    if value in (4, "4", 4.0):
        return "R"

    if value in (5, "5", 5.0, 6, "6", 6.0, "5 or 6"):
        return "P"

    return "?"


def class_fill_color(value):
    """Return the project-wide fill color for a cell class."""
    label = class_label(value)
    return CLASS_FILL_COLORS.get(label, CLASS_FILL_COLORS["?"])


def class_border_color(value):
    """Return the project-wide border color for a cell class."""
    label = class_label(value)
    return CLASS_BORDER_COLORS.get(label, CLASS_BORDER_COLORS["?"])


def class_border_and_fill(value):
    """Return project-wide border and fill colors for a cell class."""
    return class_border_color(value), class_fill_color(value)


def neurotransmitter_label(value):
    """Return the compact neurotransmitter label used for figure colors."""
    text = str(value).strip().lower()

    if text in {"vglut2", "vglut", "glut", "glutamatergic", "excitatory", "exc"}:
        return "Exc"

    if text in {"gad1b", "gad", "gaba", "gabaergic", "inhibitory", "inh"}:
        return "Inh"

    if text in {"unlabeled", "unlabelled"}:
        return "Unlabeled"

    return "Unknown"


def neurotransmitter_color(value):
    """Return the project-wide color for a neurotransmitter label."""
    label = neurotransmitter_label(value)
    return NEUROTRANSMITTER_COLORS.get(label, NEUROTRANSMITTER_COLORS["Unknown"])


def neurotransmitter_category(value):
    """Return the neurotransmitter category used for summary statistics."""
    text = str(value).strip().lower()

    if text in {"vglut2", "vglut", "glut", "glutamatergic", "excitatory", "exc"}:
        return "exc"

    if text in {"gad1b", "gad", "gaba", "gabaergic", "inhibitory", "inh"}:
        return "inh"

    if text in {"unclear", "unknown", "unlabeled", "unlabelled", "n/a", "na", "nan", "none", ""}:
        return "unc/unl"

    return "other"


def is_spn_cell(value):
    """Return True if a classifier value is an SPN label."""
    return "spn" in str(value).strip().lower()


def is_proximal_spn(value):
    """Return True if a classifier value is an SPN proximal label."""
    text = str(value).strip().lower()
    return "spn" in text and text.endswith("_prox")