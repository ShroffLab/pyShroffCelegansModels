"""C. elegans annotation tools for napari.

This package provides functions for loading StarryNite tracking data,
building lineage graphs, and setting up napari viewers with clipping
planes and lineage tree widgets.

Example usage in a notebook::

    from celegans_annotator import load_nuclei_from_zip, add_lineage_view
    from celegans_annotator.clipping_planes import (
        init_clipping_planes,
        add_clipping_plane_widgets,
    )

    nuclei_coords, nuclei_labels = load_nuclei_from_zip("path/to/SN.zip")
    # ... set up napari viewer, add images ...
    add_lineage_view(viewer, nuclei_coords, nuclei_labels)
"""

from celegans_annotator.lineage_graph import (
    DIVISION_RULES,
    add_lineage_view,
    build_lineage_graph,
    graph_to_solution_tracks,
)
from celegans_annotator.starry_nite import load_nuclei_from_zip

# clipping_planes requires Qt and is imported directly by users:
#   from celegans_annotator.clipping_planes import ...


def __getattr__(name):
    """Lazy import for GUI-dependent modules (need Qt)."""
    if name == "init_clipping_planes":
        from celegans_annotator.clipping_planes import init_clipping_planes

        return init_clipping_planes
    if name == "add_clipping_plane_widgets":
        from celegans_annotator.clipping_planes import add_clipping_plane_widgets

        return add_clipping_plane_widgets
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DIVISION_RULES",
    "add_clipping_plane_widgets",
    "add_lineage_view",
    "build_lineage_graph",
    "graph_to_solution_tracks",
    "init_clipping_planes",
    "load_nuclei_from_zip",
]
