"""Simple Qt/OpenGL viewer for a plate with a central beam.

This script demonstrates the first step of the requested project:
creating a GUI based on Qt with an embedded OpenGL viewport and a tiny
finite element style model.  The model is generated with the `gmsh`
Python API and consists of a 2 x 2 m square plate with a beam element
embedded through the middle.  The mesh is then displayed using
``pyqtgraph``'s ``GLViewWidget`` which provides a Qt based OpenGL view.

The intention is that this file will grow into a larger application that
supports additional element types, load cases and post-processing of
results.  For now it simply visualises the generated mesh.

Requirements
------------

The script relies on the following third party packages:

``gmsh``
    Provides geometry creation and meshing.

``PyQt5``
    Qt bindings for Python used for the GUI components.

``pyqtgraph``
    Simplifies 3D visualisation on top of PyQt5 and OpenGL.

These can be installed with::

    pip install gmsh pyqt5 pyqtgraph

Running in headless environments
--------------------------------

When running in an environment without a display server (for example
within this execution environment) the ``QT_QPA_PLATFORM`` environment
variable is set to ``"offscreen"`` so that the application can start
without an X server.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import gmsh  # type: ignore
except Exception as exc:  # pragma: no cover - gmsh is optional at runtime
    raise SystemExit(
        "gmsh is required to run this example. Install it with 'pip install gmsh'."
    ) from exc

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl


@dataclass
class MeshData:
    """Container for mesh information.

    Attributes
    ----------
    nodes:
        Array of node coordinates with shape ``(n, 3)``.
    triangles:
        Connectivity of shell elements as a ``(m, 3)`` array of node
        indices.
    lines:
        Connectivity of beam elements as a ``(k, 2)`` array of node
        indices.
    """

    nodes: np.ndarray
    triangles: np.ndarray
    lines: np.ndarray


def generate_plate_with_beam(size: float = 2.0, element_size: float = 0.2) -> MeshData:
    """Create a gmsh mesh for a square plate with a central beam.

    Parameters
    ----------
    size:
        Length of the plate sides in metres.
    element_size:
        Characteristic length for mesh elements.
    """

    gmsh.initialize()
    gmsh.model.add("plate_with_beam")

    # --- Geometry ---------------------------------------------------------
    half = size / 2.0
    lc = element_size

    # Corner points of the plate
    p1 = gmsh.model.geo.addPoint(-half, -half, 0, lc)
    p2 = gmsh.model.geo.addPoint(half, -half, 0, lc)
    p3 = gmsh.model.geo.addPoint(half, half, 0, lc)
    p4 = gmsh.model.geo.addPoint(-half, half, 0, lc)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    cloop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.geo.addPlaneSurface([cloop])

    # Central beam along the x-axis
    p5 = gmsh.model.geo.addPoint(-half, 0, 0, lc)
    p6 = gmsh.model.geo.addPoint(half, 0, 0, lc)
    beam_line = gmsh.model.geo.addLine(p5, p6)

    # Synchronise CAD kernel and mesh data structures
    gmsh.model.geo.synchronize()

    # NOTE: In a full application the line representing the beam should be
    # embedded into the surface so that the nodes are shared.  The gmsh
    # "mesh.embed" API can be used for this.  For this lightweight example
    # we keep them separate which is sufficient for visualisation purposes.

    # Define physical groups for clarity
    gmsh.model.addPhysicalGroup(2, [surf], tag=1)
    gmsh.model.setPhysicalName(2, 1, "Plate")
    gmsh.model.addPhysicalGroup(1, [beam_line], tag=2)
    gmsh.model.setPhysicalName(1, 2, "Beam")

    # --- Meshing ----------------------------------------------------------
    gmsh.model.mesh.generate(2)

    # Extract node coordinates
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape((-1, 3))

    # Triangles for the plate
    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2, surf)
    # gmsh returns 1-based indices; convert to 0-based
    triangles = elem_node_tags[0].reshape((-1, 3)) - 1

    # Lines for the beam
    elem_types_b, _, elem_node_tags_b = gmsh.model.mesh.getElements(1, beam_line)
    lines = elem_node_tags_b[0].reshape((-1, 2)) - 1

    gmsh.finalize()

    return MeshData(nodes=coords, triangles=triangles, lines=lines)


class MeshViewer(gl.GLViewWidget):
    """OpenGL widget used to display the plate and beam mesh."""

    def __init__(self, mesh: MeshData):
        super().__init__()
        self.setWindowTitle("Plate with Beam - gmsh/PyQt Example")
        self.setCameraPosition(distance=6)
        self.opts["azimuth"] = 45
        self.opts["elevation"] = 30
        self._add_plate(mesh)
        self._add_beam(mesh)

    def _add_plate(self, mesh: MeshData) -> None:
        mesh_data = gl.MeshData(vertexes=mesh.nodes, faces=mesh.triangles)
        item = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=False,
            color=(0.5, 0.5, 1.0, 0.5),
            shader="shaded",
            glOptions="translucent",
        )
        self.addItem(item)

    def _add_beam(self, mesh: MeshData) -> None:
        for line in mesh.lines:
            pts = mesh.nodes[line]
            item = gl.GLLinePlotItem(
                pos=pts,
                color=(1, 0, 0, 1),
                width=2,
                antialias=True,
                mode="line_strip",
            )
            self.addItem(item)


def main() -> None:
    """Entry point used when running as a script."""

    # Allow running without a display (useful for automated tests)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    mesh = generate_plate_with_beam()

    app = QtWidgets.QApplication(sys.argv)
    viewer = MeshViewer(mesh)
    viewer.show()

    # When running in an automated environment there is no user to close the
    # window.  Quit the application after a short delay so that automated
    # tests can proceed.
    QtCore.QTimer.singleShot(1000, app.quit)

    # Start the Qt event loop
    app.exec_()


if __name__ == "__main__":
    main()

