"""Graph filtering utilities for zebrafish EM connectivity graphs."""

import networkx as nx


MISSING_VALUES = {"", "n/a", "na", "nan", "none"}


def filter_graph_by_attribute(
    graph,
    attribute,
    allowed_values,
    include_missing=False,
):
    """Return a subgraph containing nodes with selected attribute values."""
    allowed_values = as_set(allowed_values)

    kept_nodes = [
        cell_id
        for cell_id, attributes in graph.nodes(data=True)
        if attribute_matches(
            attributes=attributes,
            attribute=attribute,
            allowed_values=allowed_values,
            include_missing=include_missing,
        )
    ]

    return graph.subgraph(kept_nodes).copy()


def filter_graph_by_layer(
    graph,
    allowed_layers,
    attribute="network_level",
    include_missing=False,
):
    """Return a subgraph containing nodes with selected network-level labels."""
    return filter_graph_by_attribute(
        graph=graph,
        attribute=attribute,
        allowed_values=allowed_layers,
        include_missing=include_missing,
    )


def filter_graph_by_class(
    graph,
    allowed_classes,
    attribute="classifier",
    include_missing=False,
):
    """Return a subgraph containing nodes with selected classifier labels."""
    return filter_graph_by_attribute(
        graph=graph,
        attribute=attribute,
        allowed_values=allowed_classes,
        include_missing=include_missing,
    )


def filter_largest_weakly_connected_component(graph):
    """Return the largest connected component of the graph."""
    if graph.number_of_nodes() == 0:
        return graph.copy()

    if graph.is_directed():
        components = nx.weakly_connected_components(graph)
    else:
        components = nx.connected_components(graph)

    largest_component = max(components, key=len)

    return graph.subgraph(largest_component).copy()


def attribute_matches(attributes, attribute, allowed_values, include_missing):
    value = attributes.get(attribute)

    if is_missing(value):
        return include_missing

    if isinstance(value, (list, tuple, set)):
        return bool(set(value).intersection(allowed_values))

    return value in allowed_values


def as_set(value):
    if isinstance(value, set):
        return value

    if isinstance(value, (list, tuple)):
        return set(value)

    return {value}


def is_missing(value):
    if value is None:
        return True

    if isinstance(value, str):
        return value.strip().lower() in MISSING_VALUES

    return False