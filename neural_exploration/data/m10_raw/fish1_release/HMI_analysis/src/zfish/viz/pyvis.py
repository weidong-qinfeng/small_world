"""Generic PyVis rendering utilities."""

import json
from pathlib import Path

import networkx as nx
from pyvis.network import Network


DEFAULT_PHYSICS_OPTIONS = """
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


def render_pyvis(
    graph,
    output_path,
    positions_path=None,
    node_kwargs=None,
    edge_kwargs=None,
    options=None,
    seed=42,
    height="80%",
    width="100%",
    directed=True,
    add_position_button=True,
):
    """Render a NetworkX graph as a PyVis HTML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_positions = load_pyvis_positions(positions_path)
    layout = nx.spring_layout(graph, seed=seed)
    network = Network(height=height, width=width, directed=directed)

    for cell_id, attributes in graph.nodes(data=True):
        x_position, y_position = get_node_position(cell_id, layout, saved_positions)

        node_options = {
            "label": str(cell_id),
            "x": x_position,
            "y": y_position,
            "physics": False,
        }

        if node_kwargs is not None:
            node_options.update(
                node_kwargs(cell_id, attributes, x_position, y_position)
            )

        network.add_node(cell_id, **node_options)

    for source_cell, target_cell, edge_attributes in graph.edges(data=True):
        edge_options = {}

        if edge_kwargs is not None:
            edge_options.update(
                edge_kwargs(source_cell, target_cell, edge_attributes, graph)
            )

        network.add_edge(source_cell, target_cell, **edge_options)

    if options is not None:
        network.set_options(options)

    html = network.generate_html(notebook=False)

    if add_position_button:
        html = add_save_positions_button(html)

    output_path.write_text(html, encoding="utf-8")
    print(f"Saved HTML: {output_path}")

    return output_path


def load_pyvis_positions(path):
    """Load saved PyVis node positions from JSON."""
    if path is None:
        return {}

    path = Path(path)

    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def get_node_position(cell_id, layout, saved_positions):
    """Return saved node position, otherwise return fallback layout position."""
    if str(cell_id) in saved_positions:
        position = saved_positions[str(cell_id)]
        return float(position["x"]), float(position["y"])

    x_position, y_position = layout[cell_id] * 1000

    return float(x_position), float(y_position)


def add_save_positions_button(html):
    """Add a button to export current PyVis node positions."""
    return html.replace(
        "</body>",
        """
        <button onclick="downloadPositions()" style="position:fixed;top:10px;right:10px;z-index:999;">Save Positions</button>
        <script>
        function downloadPositions() {
            const data = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(network.getPositions()));
            const link = document.createElement('a');
            link.href = data;
            link.download = 'pyvis_node_positions.json';
            document.body.appendChild(link);
            link.click();
            link.remove();
        }
        </script>
        </body>""",
    )