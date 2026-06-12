"""Clipping plane utilities for napari 3D viewers.

Provides functions to set up 6-sided bounding-box clipping planes
(x_min, x_max, y_min, y_max, z_min, z_max) on napari layers, and
to add interactive slider widgets that adjust them in real time.
"""

import magicgui
import napari
from qtpy.QtWidgets import QSizePolicy


def init_clipping_planes(
    layers: list[napari.layers.Layer],
    shape: tuple[int, int, int],
) -> None:
    """Initialize 6 bounding-box clipping planes on each layer.

    Creates clipping planes at the bounds of the volume so that all
    data is initially visible. The planes are:
        0: x_min (normal +x)
        1: x_max (normal -x)
        2: y_max (normal -y)
        3: y_min (normal +y)
        4: z_min (normal +z)
        5: z_max (normal -z)

    Args:
        layers: List of napari layers to add clipping planes to.
        shape: Spatial dimensions (x, y, z) of the volume.
    """
    x, y, z = shape
    planes = [
        {"position": (0, 0, 0), "normal": (1, 0, 0), "enabled": True},
        {"position": (x, 0, 0), "normal": (-1, 0, 0), "enabled": True},
        {"position": (0, y, 0), "normal": (0, -1, 0), "enabled": True},
        {"position": (0, 0, 0), "normal": (0, 1, 0), "enabled": True},
        {"position": (0, 0, 0), "normal": (0, 0, 1), "enabled": True},
        {"position": (0, 0, z), "normal": (0, 0, -1), "enabled": True},
    ]
    for layer in layers:
        layer.experimental_clipping_planes = planes


def add_clipping_plane_widgets(
    viewer: napari.Viewer,
    layers: list[napari.layers.Layer],
    shape: tuple[int, int, int],
) -> None:
    """Add x/y/z clipping plane range slider widgets to the viewer.

    Creates three magicgui range sliders (x, y, z) and docks them
    into the napari viewer. Moving the sliders updates the clipping
    planes on all provided layers simultaneously.

    Args:
        viewer: napari Viewer instance.
        layers: Layers to clip (e.g. green image, red image, points).
        shape: Spatial dimensions (x, y, z) of the volume.
    """
    x, y, z = shape

    def _update_clip_planes(layer, plane_idx_a, plane_idx_b, position_a, position_b):
        layer.experimental_clipping_planes[plane_idx_a].position = position_a
        layer.experimental_clipping_planes[plane_idx_b].position = position_b

    # --- X slider ---
    @magicgui.magicgui(
        auto_call=True,
        threshold={
            "widget_type": "RangeSlider",
            "min": 0,
            "max": x,
            "label": "Clip x",
            "orientation": "vertical",
        },
    )
    def clip_x(threshold=(0, x)):
        for layer in layers:
            _update_clip_planes(
                layer,
                0,
                1,
                (threshold[0], 0, 0),
                (threshold[1], 0, 0),
            )

    clip_x.min_width = 200
    clip_x.min_height = 200
    clip_x.max_height = 600
    clip_x.native.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Expanding,
    )

    # --- Y slider ---
    @magicgui.magicgui(
        auto_call=True,
        threshold={
            "widget_type": "RangeSlider",
            "min": 0,
            "max": y,
            "label": "Clip y",
            "orientation": "vertical",
        },
    )
    def clip_y(threshold=(0, y)):
        for layer in layers:
            _update_clip_planes(
                layer,
                2,
                3,
                (0, threshold[1], 0),
                (0, threshold[0], 0),
            )

    clip_y.min_width = 200
    clip_y.min_height = 200
    clip_y.max_height = 600
    clip_y.native.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Expanding,
    )

    # --- Z slider ---
    @magicgui.magicgui(
        auto_call=True,
        threshold={
            "widget_type": "RangeSlider",
            "min": 0,
            "max": z,
            "label": "Clip z",
            "orientation": "horizontal",
        },
    )
    def clip_z(threshold=(0, z)):
        for layer in layers:
            _update_clip_planes(
                layer,
                4,
                5,
                (0, 0, threshold[0]),
                (0, 0, threshold[1]),
            )

    viewer.window.add_dock_widget(clip_x, name="Clipping Planes X", area="right")
    viewer.window.add_dock_widget(clip_y, name="Clipping Planes Y", area="right")
    viewer.window.add_dock_widget(clip_z, name="Clipping Planes Z", area="bottom")
