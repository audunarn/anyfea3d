"""Simple PySide6 OpenGL viewer with gmsh-generated plate and beam mesh."""
from __future__ import annotations

import sys
import argparse

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

        # Plate corner points and beam endpoints
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




# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

def run_gui() -> None:  # pragma: no cover - requires GUI environment
    """Launch the PySide6 viewer with OpenGL rendering."""

    from PySide6 import QtWidgets
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
        """Simple OpenGL widget to render the mesh."""

        def __init__(self, mesh: FEMesh):
            super().__init__()
            self.mesh = mesh
            self.angle = 20.0

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
            glTranslatef(0.0, 0.0, -6.0)
            glRotatef(self.angle, 1.0, 0.0, 0.0)
            glRotatef(self.angle, 0.0, 1.0, 0.0)

            # Draw plate triangles
            glColor3f(0.6, 0.6, 0.8)
            glBegin(GL_TRIANGLES)
            for tri in self.mesh.triangles:
                for idx in tri:
                    x, y, z = self.mesh.coords[idx]
                    glVertex3f(x, y, z)
            glEnd()

            # Draw beam lines
            glColor3f(0.8, 0.1, 0.1)
            glLineWidth(3.0)
            glBegin(GL_LINES)
            for line in self.mesh.lines:
                for idx in line:
                    x, y, z = self.mesh.coords[idx]
                    glVertex3f(x, y, z)
            glEnd()
            glFlush()

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self, mesh: FEMesh):  # pragma: no cover - GUI setup
            super().__init__()
            self.setWindowTitle("FEA Plate with Beam")
            self.setGeometry(100, 100, 800, 600)
            self.widget = GLWidget(mesh)
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
    args = parser.parse_args()

    if args.test:
        run_test()
    else:
        run_gui()
