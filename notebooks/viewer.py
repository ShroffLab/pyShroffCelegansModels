# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # C. elegans Annotation Viewer
#
# Opens zarr image data in napari with clipping planes, StarryNite nuclei annotations,
# and a lineage tree view.
#
# Uses the `celegans_annotator` package — edit the functions there and they'll
# auto-reload here thanks to `%autoreload`.

# %%
from pathlib import Path

import napari
import toml
from motile_tracker.data_views.views.ortho_views import initialize_ortho_views

from celegans_annotator import add_lineage_view, load_nuclei_from_zip
from celegans_annotator.clipping_planes import (
    add_clipping_plane_widgets,
    init_clipping_planes,
)

# %% [markdown]
# ## Set your data paths here
#
# Edit these paths to point to your data. Use `Path(r"...")` on Windows.

# %%
config = toml.load("notebooks/config.toml")
# New CND-1
base = Path(config["data"])
green_zarr_path = base / "green_channel.zarr"
red_zarr_path = base / "red_channel.zarr"
sn_zip_path = base / "StarryNite/SN_files/Decon_emb1_edited.zip"

# %% [markdown]
# ## Load StarryNite data

# %%
nuclei_coords, nuclei_labels = load_nuclei_from_zip(sn_zip_path)
print(f"Loaded {len(nuclei_coords)} nuclei, {len(set(nuclei_labels))} unique cell names")

# %% [markdown]
# ## Create napari viewer

# %%
viewer = napari.Viewer()
initialize_ortho_views(viewer)

green = viewer.open(
    path=str(green_zarr_path),
    blending="additive",
    contrast_limits=(0, 300),
    colormap="green",
    rendering="attenuated_mip",
    attenuation=0.75,
)
red = viewer.open(
    path=str(red_zarr_path),
    blending="additive",
    contrast_limits=(0, 300),
    colormap="red",
    rendering="attenuated_mip",
    attenuation=0.75,
)

viewer.add_points(
    nuclei_coords,
    ndim=4,
    opacity=0.7,
    size=6,
    face_color="transparent",
    border_color="cyan",
    properties={"name": nuclei_labels},
    text="name",
    blending="additive",
)

green_layer = viewer.layers[0]
red_layer = viewer.layers[1]
point_layer = viewer.layers[2]

# %% [markdown]
# ## Add clipping planes

# %%
# Get spatial dimensions from the image data (t, x, y, z)
t, x, y, z = green_layer.data.shape
layers = [green_layer, red_layer, point_layer]

init_clipping_planes(layers, shape=(x, y, z))
add_clipping_plane_widgets(viewer, layers, shape=(x, y, z))

viewer.dims.ndisplay = 3

# %% [markdown]
# ## Add lineage tree view

# %%
add_lineage_view(viewer, nuclei_coords, nuclei_labels)

# %%
napari.run()
