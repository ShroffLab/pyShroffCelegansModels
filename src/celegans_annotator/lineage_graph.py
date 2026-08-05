"""Build lineage graphs from C. elegans nuclei tracking data.

Constructs a directed graph where nodes are cell detections (one per cell
per timepoint) and edges connect cells across adjacent timepoints. Edges
represent either:
    1. Cell continuation: same cell name at t and t+1
    2. Cell division: parent at t connected to daughter cells at t+1

Division detection uses a combination of hard-coded rules for known
C. elegans naming exceptions and a prefix-based heuristic for standard
lineage naming (e.g. AB divides into ABa and ABp).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
from funtracks.data_model import SolutionTracks
from funtracks.import_export import tracks_from_df
from motile_tracker.data_views.views.tree_view.tree_widget import TreeWidget
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)

# Known C. elegans division rules that don't follow the standard prefix
# pattern. Format: parent_name -> [daughter1_name, daughter2_name]
DIVISION_RULES: dict[str, list[str]] = {
    "P0": ["AB", "P1"],
    "P1": ["EMS", "P2"],
    "P2": ["C", "P3"],
    "P3": ["D", "P4"],
    "EMS": ["MS", "E"],
}


def get_daughters(
    parent_name: str,
    candidates_at_next_t: list[tuple[str, int]],
    division_rules: dict[str, list[str]] | None = None,
) -> list[int]:
    """Find daughter cell indices for a parent cell at the next timepoint.

    First checks hard-coded division rules (for known naming exceptions
    like P0 -> AB + P1), then falls back to prefix-based detection
    (parent "AB" -> daughters "ABa", "ABp" where len(daughter) == len(parent) + 1).

    Args:
        parent_name: Name of the parent cell.
        candidates_at_next_t: List of (cell_name, node_index) tuples for
            all cells at the next timepoint.
        division_rules: Custom division rules dict. If None, uses the
            standard C. elegans DIVISION_RULES.

    Returns:
        List of node indices for identified daughter cells.
    """
    if division_rules is None:
        division_rules = DIVISION_RULES

    candidate_names = dict(candidates_at_next_t)

    # Check hard-coded rules first
    if parent_name in division_rules:
        daughters = []
        for daughter_name in division_rules[parent_name]:
            if daughter_name in candidate_names:
                daughters.append(candidate_names[daughter_name])
        if daughters:
            return daughters

    # Fall back to prefix-based detection:
    # Look for cells where parent name is a prefix and name is one char longer
    daughters = []
    for candidate_name, candidate_idx in candidates_at_next_t:
        if (
            candidate_name.startswith(parent_name)
            and len(candidate_name) == len(parent_name) + 1
        ):
            daughters.append(candidate_idx)

    return daughters


def build_lineage_graph(
    nuclei_coords: np.ndarray,
    nuclei_labels: list[str],
    division_rules: dict[str, list[str]] | None = None,
) -> nx.DiGraph:
    """Build a directed graph representing the cell lineage.

    Each cell detection becomes a node with time, position, and name
    attributes. Edges connect cells across adjacent timepoints via
    continuation (same name) or division (daughter detection).

    Args:
        nuclei_coords: Array of shape (N, 4) with columns [time, z, y, x].
        nuclei_labels: List of N cell name strings.
        division_rules: Custom division rules. Defaults to C. elegans rules.

    Returns:
        NetworkX DiGraph with node attributes: time, z, y, x, pos, name.
    """
    # Build lookup from (time, name) -> node index
    time_name_to_idx: dict[tuple[int, str], int] = {}
    for idx, (coords, label) in enumerate(zip(nuclei_coords, nuclei_labels, strict=True)):
        t = int(coords[0])
        time_name_to_idx[(t, label)] = idx

    graph = nx.DiGraph()

    # Add nodes with attributes
    for idx, (coords, label) in enumerate(zip(nuclei_coords, nuclei_labels, strict=True)):
        t = int(coords[0])
        z, y, x = float(coords[1]), float(coords[2]), float(coords[3])
        graph.add_node(
            idx,
            time=t,
            pos=[z, y, x],
            z=z,
            y=y,
            x=x,
            name=label,
        )

    # Add edges between adjacent timepoints
    edge_count = 0
    division_count = 0

    for idx, (coords, label) in enumerate(zip(nuclei_coords, nuclei_labels, strict=True)):
        t = int(coords[0])
        next_t = t + 1

        # Case 1: Same cell continues to next timepoint
        next_key = (next_t, label)
        if next_key in time_name_to_idx:
            next_idx = time_name_to_idx[next_key]
            graph.add_edge(idx, next_idx)
            edge_count += 1
        else:
            # Case 2: Cell might divide — find daughters at next timepoint
            candidates = [
                (name, node_idx)
                for (check_t, name), node_idx in time_name_to_idx.items()
                if check_t == next_t
            ]

            daughters = get_daughters(label, candidates, division_rules)
            for daughter_idx in daughters:
                graph.add_edge(idx, daughter_idx)
                edge_count += 1
                division_count += 1

    logger.info(
        "Built graph: %d nodes, %d edges (%d division edges)",
        graph.number_of_nodes(),
        edge_count,
        division_count,
    )
    return graph


def graph_to_solution_tracks(
    graph: nx.DiGraph,
    scale: list[float] | None = None,
) -> SolutionTracks:
    """Convert a lineage graph to a funtracks SolutionTracks object.

    Converts the NetworkX graph into a DataFrame with columns expected
    by funtracks (id, time, z, y, x, parent_id, name) and uses
    tracks_from_df to build SolutionTracks.

    Pre-computes ``track_id`` and ``lineage_id`` on the lightweight
    NetworkX graph so that funtracks can skip its expensive rustworkx-based
    recomputation during ``SolutionTracks`` initialization.

    Args:
        graph: Lineage graph from build_lineage_graph.
        scale: Scale factors [t, z, y, x]. Defaults to [1, 1, 1, 1].

    Returns:
        SolutionTracks instance compatible with motile_tracker widgets.
    """
    if scale is None:
        scale = [1.0, 1.0, 1.0, 1.0]

    # Pre-compute lineage_id: each weakly connected component
    lineage_ids: dict[int, int] = {}
    for lid, component in enumerate(nx.weakly_connected_components(graph), start=1):
        for node in component:
            lineage_ids[node] = lid

    # Pre-compute track_id: remove division edges, then find components
    graph_no_divs = graph.copy()
    for node in graph.nodes:
        if graph.out_degree(node) >= 2:
            for succ in list(graph.successors(node)):
                graph_no_divs.remove_edge(node, succ)
    track_ids: dict[int, int] = {}
    for tid, component in enumerate(
        nx.weakly_connected_components(graph_no_divs), start=1
    ):
        for node in component:
            track_ids[node] = tid

    # Build a DataFrame from the graph nodes and edges.
    rows = []
    for node_id in graph.nodes:
        attrs = graph.nodes[node_id]
        predecessors = list(graph.predecessors(node_id))
        parent_id = predecessors[0] if predecessors else -1
        rows.append(
            {
                "id": node_id,
                "time": attrs["time"],
                "z": attrs["z"],
                "y": attrs["y"],
                "x": attrs["x"],
                "parent_id": parent_id,
                "name": attrs.get("name", ""),
                "tracklet_id": track_ids[node_id],
                "lineage_id": lineage_ids[node_id],
            }
        )

    df = pd.DataFrame(rows)

    return tracks_from_df(
        df,
        scale=scale,
        node_name_map={
            "time": "time",
            "pos": ["z", "y", "x"],
            "id": "id",
            "parent_id": "parent_id",
            "name": "name",
            "tracklet_id": "tracklet_id",
            "lineage_id": "lineage_id",
        },
    )


def add_lineage_view(
    viewer: napari.Viewer,
    nuclei_coords: np.ndarray,
    nuclei_labels: list[str],
    name: str = "StarryNite Tracks",
) -> None:
    """Add a lineage tree view widget to a napari viewer.

    Builds the lineage graph from nuclei data, creates SolutionTracks,
    sets up the TracksViewer, adds cell names to the points layer,
    and docks the TreeWidget.

    Args:
        viewer: napari Viewer instance.
        nuclei_coords: Array of shape (N, 4) with [time, z, y, x].
        nuclei_labels: List of N cell name strings.
        name: Display name for the tracks in the viewer.
    """
    graph = build_lineage_graph(nuclei_coords, nuclei_labels)
    tracks = graph_to_solution_tracks(graph)

    # Set up TracksViewer (motile_tracker's viewer integration)
    tracks_viewer = TracksViewer.get_instance(viewer)
    tracks_viewer.update_tracks(tracks, name=name)

    # Add cell names to the track points layer created by TracksViewer
    points_layer = None
    for layer in viewer.layers:
        if layer.name.endswith("_points"):
            points_layer = layer
            break

    if points_layer is not None:
        node_ids = points_layer.properties["node_id"]
        # Use nuclei_labels directly — node IDs are indices into the
        # original arrays, so this avoids expensive per-node lookups
        # through funtracks' rustworkx graph.
        cell_names = [nuclei_labels[nid] for nid in node_ids]
        points_layer.features["name"] = cell_names
        # Update feature_defaults so ortho view copies have the same columns
        defaults = points_layer.feature_defaults
        defaults["name"] = ""
        points_layer.feature_defaults = defaults
        points_layer.text = "name"
        points_layer.refresh()
        logger.info("Added %d cell names to track points layer", len(cell_names))
    else:
        logger.warning("Could not find track points layer (expected *_points)")

    # Add the lineage tree widget
    tree_widget = TreeWidget(viewer)
    viewer.window.add_dock_widget(tree_widget, name="Lineage View", area="bottom")
    logger.info("Lineage view added")
