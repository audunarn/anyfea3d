"""Simple PySide6 OpenGL viewer with gmsh-generated plate and beam mesh."""
from __future__ import annotations

import sys
import argparse
import re

import numpy as np

try:
    import gmsh  # type: ignore
except Exception as exc:  # pragma: no cover - gmsh is required
    gmsh = None
    GMESH_IMPORT_ERROR = exc
else:  # pragma: no cover
    GMESH_IMPORT_ERROR = None



class FEMesh:
    """Hold coordinates and connectivity for the FE mesh."""

    def __init__(self, coords: np.ndarray, triangles: np.ndarray, lines: np.ndarray):
        self.coords = coords  # (n_nodes, 3)
        self.triangles = triangles  # (n_tris, 3)
        self.lines = lines  # (n_lines, 2)

    @classmethod
    def plate_with_beam(cls) -> "FEMesh":
        """Create a 2 x 2 m plate with a centered beam using gmsh."""

        if gmsh is None:
            raise ImportError("gmsh Python API is required") from GMESH_IMPORT_ERROR

        gmsh.initialize([])
        gmsh.model.add("plate_with_beam")

        # Plate corner points and beam endpoints (mesh size 0.1 m)
        p1 = gmsh.model.geo.addPoint(0, 0, 0)
        p2 = gmsh.model.geo.addPoint(2, 0, 0)
        p3 = gmsh.model.geo.addPoint(2, 2, 0)
        p4 = gmsh.model.geo.addPoint(0, 2, 0)
        pb1 = gmsh.model.geo.addPoint(1, 0, 0)
        pb2 = gmsh.model.geo.addPoint(1, 2, 0)

        # Plate boundary lines split at beam connection points
        l1a = gmsh.model.geo.addLine(p1, pb1)
        l1b = gmsh.model.geo.addLine(pb1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3a = gmsh.model.geo.addLine(p3, pb2)
        l3b = gmsh.model.geo.addLine(pb2, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)
        cl = gmsh.model.geo.addCurveLoop([l1a, l1b, l2, l3a, l3b, l4])
        surface = gmsh.model.geo.addPlaneSurface([cl])

        # Beam line through center (already uses pb1 and pb2)
        beam_line = gmsh.model.geo.addLine(pb1, pb2)

        gmsh.model.geo.synchronize()
        # Uniform mesh size of 0.1 m
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.1)
        # Embed the beam into the surface so they share nodes
        gmsh.model.mesh.embed(1, [beam_line], 2, surface)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

        # Extract node coordinates
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords, dtype=float).reshape(-1, 3)

        # Map gmsh node tags to indices
        tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}

        # Get 2D elements (triangles)
        tri_conn = []
        etypes, _, enodes = gmsh.model.mesh.getElements(dim=2)
        for etype, node_tags_array in zip(etypes, enodes):
            if etype == 2:  # 3-node triangle
                tri_tags = np.array(node_tags_array, dtype=int).reshape(-1, 3)
                tri_conn.append(tri_tags)
        if tri_conn:
            tri_conn = np.vstack(tri_conn)
        else:
            tri_conn = np.zeros((0, 3), dtype=int)

        # Get 1D elements (lines for beam)
        line_conn = []
        etypes1, _, enodes1 = gmsh.model.mesh.getElements(dim=1)
        for etype, node_tags_array in zip(etypes1, enodes1):
            if etype == 1:  # 2-node line
                line_tags = np.array(node_tags_array, dtype=int).reshape(-1, 2)
                line_conn.append(line_tags)
        if line_conn:
            line_conn = np.vstack(line_conn)
        else:
            line_conn = np.zeros((0, 2), dtype=int)

        gmsh.finalize()

        # Convert gmsh node tags to zero-based indices
        triangles = np.vectorize(tag_to_idx.get)(tri_conn)
        lines = np.vectorize(tag_to_idx.get)(line_conn)

        return cls(coords, triangles, lines)


def beam_section_wireframe(
    length: float = 2.0, section: str = "T300x12_100x10"
) -> np.ndarray:
    """Return line segments for a T-section beam extruded along *length*.

    The *section* string is expected in the form ``TfwxfT_whxwt`` where the
    dimensions are in millimetres (flange width, flange thickness, web height,
    web thickness).
    """

    match = re.fullmatch(r"T(\d+)x(\d+)_(\d+)x(\d+)", section)
    if not match:
        raise ValueError("section format must be TfwxfT_whxwt")
    fw, ft, wh, wt = map(float, match.groups())
    fw /= 1000.0
    ft /= 1000.0
    wh /= 1000.0
    wt /= 1000.0

    half_fw = fw / 2.0
    half_wt = wt / 2.0
    top_web = wh
    top_flange = wh + ft

    # 2D polygon of the T-section in the local x-z plane
    poly = np.array(
        [
            [-half_wt, 0.0],
            [half_wt, 0.0],
            [half_wt, top_web],
            [half_fw, top_web],
            [half_fw, top_flange],
            [-half_fw, top_flange],
            [-half_fw, top_web],
            [-half_wt, top_web],
        ],
        dtype=float,
    )

    n = len(poly)
    segments = []
    # Bottom and top polygons
    for i in range(n):
        j = (i + 1) % n
        p1 = poly[i]
        p2 = poly[j]
        segments.append([[p1[0] + 1.0, 0.0, p1[1]], [p2[0] + 1.0, 0.0, p2[1]]])
        segments.append([[p1[0] + 1.0, length, p1[1]], [p2[0] + 1.0, length, p2[1]]])

    # Vertical edges
    for i in range(n):
        p = poly[i]
        segments.append(
            [[p[0] + 1.0, 0.0, p[1]], [p[0] + 1.0, length, p[1]]]
        )

    return np.array(segments, dtype=float)




# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

def run_gui(show_mesh: bool, beam_section: str) -> None:  # pragma: no cover - requires GUI env
    """Launch the PySide6 viewer with OpenGL rendering."""

    from PySide6 import QtWidgets
    from PySide6.QtCore import Qt
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from OpenGL.GL import (
        glBegin,
        glClear,
        glClearColor,
        glColor3f,
        glEnd,
        glEnable,
        glVertex3f,
        glLineWidth,
        glMatrixMode,
        glLoadIdentity,
        glTranslatef,
        glRotatef,
        glViewport,
        glFlush,
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_LINES,
        GL_PROJECTION,
        GL_MODELVIEW,
        GL_TRIANGLES,
    )
    from OpenGL.GLU import gluPerspective

    class GLWidget(QOpenGLWidget):
        """Simple OpenGL widget to render the mesh and beam."""

        def __init__(self, mesh: FEMesh, show_mesh: bool, section: str):
            super().__init__()
            self.mesh = mesh
            self.show_mesh = show_mesh
            self.beam_segments = beam_section_wireframe(section=section)
            self.angle_x = 20.0
            self.angle_y = 20.0
            self.distance = 6.0
            self._last_pos = None

        def initializeGL(self):  # pragma: no cover - requires OpenGL context
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glEnable(GL_DEPTH_TEST)

        def resizeGL(self, w: int, h: int):  # pragma: no cover - requires OpenGL context
            glViewport(0, 0, w, h)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            aspect = w / h if h else 1
            gluPerspective(45.0, aspect, 0.1, 100.0)

        def paintGL(self):  # pragma: no cover - requires OpenGL context
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glTranslatef(0.0, 0.0, -self.distance)
            glRotatef(self.angle_x, 1.0, 0.0, 0.0)
            glRotatef(self.angle_y, 0.0, 1.0, 0.0)

            # Draw plate triangles if requested
            if self.show_mesh:
                glColor3f(0.6, 0.6, 0.8)
                glBegin(GL_TRIANGLES)
                for tri in self.mesh.triangles:
                    for idx in tri:
                        x, y, z = self.mesh.coords[idx]
                        glVertex3f(x, y, z)
                glEnd()

            # Draw beam wireframe
            glColor3f(0.8, 0.1, 0.1)
            glLineWidth(2.0)
            glBegin(GL_LINES)
            for seg in self.beam_segments:
                glVertex3f(*seg[0])
                glVertex3f(*seg[1])
            glEnd()
            glFlush()

        # Basic mouse interaction for rotation and zoom
        def mousePressEvent(self, event):  # pragma: no cover - GUI interaction
            self._last_pos = event.position()

        def mouseMoveEvent(self, event):  # pragma: no cover - GUI interaction
            if self._last_pos and event.buttons() & Qt.LeftButton:
                dx = event.position().x() - self._last_pos.x()
                dy = event.position().y() - self._last_pos.y()
                self.angle_x += dy * 0.5
                self.angle_y += dx * 0.5
                self._last_pos = event.position()
                self.update()

        def wheelEvent(self, event):  # pragma: no cover - GUI interaction
            self.distance -= event.angleDelta().y() / 240.0
            self.distance = max(1.0, self.distance)
            self.update()

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self, mesh: FEMesh):  # pragma: no cover - GUI setup
            super().__init__()
            self.setWindowTitle("FEA Plate with Beam")
            self.setGeometry(100, 100, 800, 600)
            self.widget = GLWidget(mesh, show_mesh, beam_section)
            self.setCentralWidget(self.widget)

    mesh = FEMesh.plate_with_beam()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(mesh)
    window.show()
    sys.exit(app.exec())


def run_test() -> None:
    """Create the mesh and print basic information."""
    mesh = FEMesh.plate_with_beam()
    print(f"Nodes: {len(mesh.coords)}")
    print(f"Triangles: {len(mesh.triangles)}")
    print(f"Beam lines: {len(mesh.lines)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plate with central beam demo")
    parser.add_argument(
        "--test",
        action="store_true",
        help="only generate the mesh and print information",
    )
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="visualize mesh triangles in the viewer",
    )
    parser.add_argument(
        "--beam-section",
        default="T300x12_100x10",
        help="beam section as TfwxfT_whxwt in mm",
    )
    args = parser.parse_args()

    if args.test:
        run_test()
    else:
        run_gui(args.mesh, args.beam_section)
