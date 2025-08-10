import sys
import numpy as np
import gmsh

from PySide6 import QtWidgets
from pyqtgraph.opengl import GLViewWidget, GLMeshItem, GLLinePlotItem, GLGridItem


# ---------- Gmsh-modell: plate + innebygd bjelkekurve ----------
def build_gmsh_plate_with_embedded_beam(Lx=1.0, Ly=1.0, lc=0.05):
    """
    Lager rektangulær plate (plane surface) og en sentral linje (bjelke) som Embedded Curve.
    lc = mål på elementstørrelse.
    Returnerer:
      nodes      : (N,3) float, ordnet etter intern reindeksering (0..N-1)
      tri_faces  : (M,3) int, trekanter for rendering
      beam_node_idx_sorted : (K,) int, indeksene (i nodes) til bjelkens noder i riktig rekkefølge
    """
    gmsh.initialize()
    gmsh.model.add("plate_beam")

    # Geometri
    x0, x1 = 0.0, Lx
    y0, y1 = 0.0, Ly

    p1 = gmsh.model.geo.addPoint(x0, y0, 0, lc)
    p2 = gmsh.model.geo.addPoint(x1, y0, 0, lc)
    p3 = gmsh.model.geo.addPoint(x1, y1, 0, lc)
    p4 = gmsh.model.geo.addPoint(x0, y1, 0, lc)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s  = gmsh.model.geo.addPlaneSurface([cl])

    # Bjelke som midtlinje i y = Ly/2
    pb1 = gmsh.model.geo.addPoint(x0, Ly/2, 0, lc*0.5)
    pb2 = gmsh.model.geo.addPoint(x1, Ly/2, 0, lc*0.5)
    beam_line = gmsh.model.geo.addLine(pb1, pb2)

    # Embed bjelken i plateflaten → felles noder
    gmsh.model.geo.mesh.embed(1, [beam_line], 2, s)

    gmsh.model.geo.synchronize()

    # Mesh
    gmsh.model.mesh.generate(2)  # 2D (trenger ikke eksplisitt 1D når embedded)
    gmsh.model.mesh.optimize("Netgen")

    # Hent noder
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    nodeCoords = nodeCoords.reshape(-1, 3)

    # Reindekser nodetag -> 0..N-1
    tag_to_idx = {int(t): i for i, t in enumerate(nodeTags.astype(int))}

    # Hent trekant-elementer for plate
    etypes, eTags, eNodeTags = gmsh.model.mesh.getElements(dim=2, tag=s)
    tri_faces = []
    for etype, enodes in zip(etypes, eNodeTags):
        # 2 = 3-node triangle, 9 = 6-node triangle, etc.
        if gmsh.model.mesh.getElementProperties(etype)[0] in ("Triangle", "Tria"):
            en = np.array(enodes, dtype=int).reshape(-1, gmsh.model.mesh.getElementProperties(etype)[3])
            if en.shape[1] >= 3:
                # Ta kun de tre første for rendering hvis høyere ordens
                tri = en[:, :3]
                tri_idx = np.vectorize(tag_to_idx.get)(tri)
                tri_faces.append(tri_idx)
    if not tri_faces:
        raise RuntimeError("Fant ingen tri-elementer på platen.")
    tri_faces = np.vstack(tri_faces)

    # Hent bjelkens noder (fra 1D elementer langs embedded line)
    etypes1, eTags1, eNodeTags1 = gmsh.model.mesh.getElements(dim=1, tag=beam_line)
    if len(eNodeTags1) == 0 or len(eNodeTags1[0]) == 0:
        raise RuntimeError("Fant ingen 1D elementer på bjelkelinjen.")
    line_conn = np.array(eNodeTags1[0], dtype=int).reshape(-1, gmsh.model.mesh.getElementProperties(etypes1[0])[3])

    # Unike noder på linjen
    beam_node_tags = np.unique(line_conn.flatten())
    beam_pts = nodeCoords[np.vectorize(tag_to_idx.get)(beam_node_tags)]
    # Sorter langs x (ok for rett linje); alternativ: bygg kjede via konnektivitet
    order = np.argsort(beam_pts[:, 0])
    beam_node_tags_sorted = beam_node_tags[order]
    beam_node_idx_sorted = np.vectorize(tag_to_idx.get)(beam_node_tags_sorted)

    gmsh.finalize()

    return nodeCoords, tri_faces, beam_node_idx_sorted


# ---------- GUI ----------
class GmshPlateBeamWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gmsh-basert skall+bjelke (embedded curve, delt mesh)")
        self.resize(1100, 750)

        self.view = GLViewWidget()
        self.view.setCameraPosition(distance=2.4, azimuth=45, elevation=25)
        self.view.opts['center'] = np.array([0.5, 0.5, 0.0])
        self.setCentralWidget(self.view)

        grid = GLGridItem()
        grid.scale(0.1, 0.1, 0.1)
        grid.setSize(1.2, 1.2, 1.2)
        self.view.addItem(grid)

        # Parametre
        self.Lx = 1.0
        self.Ly = 1.0
        self.lc = 0.06  # mål på elementstørrelse

        # Bygg første mesh
        self._rebuild_mesh()

        # Toolbar
        tb = QtWidgets.QToolBar("Verktøy", self)
        self.addToolBar(tb)
        self.lc_spin = QtWidgets.QDoubleSpinBox()
        self.lc_spin.setRange(0.005, 0.5)
        self.lc_spin.setDecimals(3)
        self.lc_spin.setSingleStep(0.005)
        self.lc_spin.setValue(self.lc)
        self.lc_spin.setPrefix("lc=")

        regen = QtWidgets.QPushButton("Regenerer mesh")
        regen.clicked.connect(self._on_regen)

        deform = QtWidgets.QPushButton("Demo-deformasjon")
        deform.clicked.connect(self._on_deform)

        reset = QtWidgets.QPushButton("Nullstill")
        reset.clicked.connect(self._on_reset)

        tb.addWidget(self.lc_spin)
        tb.addWidget(regen)
        tb.addSeparator()
        tb.addWidget(deform)
        tb.addWidget(reset)

        self.statusBar().showMessage("Embedded Curve i Gmsh: bjelke deler noder med plate.")

    def _rebuild_mesh(self):
        nodes, faces, beam_idx = build_gmsh_plate_with_embedded_beam(self.Lx, self.Ly, self.lc)
        self.nodes0 = nodes.copy()
        self.nodes = nodes.copy()
        self.faces = faces
        self.beam_idx = beam_idx

        # Platevisning
        if hasattr(self, "plate_item"):
            self.view.removeItem(self.plate_item)
        self.plate_item = GLMeshItem(
            vertexes=self.nodes,
            faces=self.faces,
            smooth=False,
            drawFaces=True,
            drawEdges=True,
            edgeColor=(0.1, 0.1, 0.1, 0.6),
            faceColor=(0.2, 0.6, 1.0, 0.35),
            computeNormals=False,
        )
        self.view.addItem(self.plate_item)

        # Bjelkevisning (polyline gjennom bjelkens noder i rekkefølge)
        if hasattr(self, "beam_item"):
            self.view.removeItem(self.beam_item)
        self.beam_item = GLLinePlotItem(pos=self.nodes[self.beam_idx], width=3.0, antialias=True, mode='line_strip')
        self.view.addItem(self.beam_item)

    def _on_regen(self):
        self.lc = float(self.lc_spin.value())
        self._rebuild_mesh()

    def _on_reset(self):
        self.nodes[:] = self.nodes0
        self.plate_item.setMeshData(vertexes=self.nodes, faces=self.faces)
        self.beam_item.setData(pos=self.nodes[self.beam_idx])
        self.statusBar().showMessage("Nullstilt posisjoner.")

    def _on_deform(self):
        # En glatt demo-deformasjon som flytter noder (ikke FEA, kun illustrasjon)
        X = self.nodes0[:, 0]
        Y = self.nodes0[:, 1]
        xc, yc = self.Lx * 0.5, self.Ly * 0.5
        r2 = (X - xc) ** 2 + (Y - yc) ** 2

        ux = 0.01 * (Y - yc)
        uy = -0.01 * (X - xc)
        wz = 0.02 * np.exp(-50 * r2)

        disp = np.zeros_like(self.nodes0)
        disp[:, 0] = ux
        disp[:, 1] = uy
        disp[:, 2] = wz

        self.nodes = self.nodes0 + disp
        # Oppdater plate
        self.plate_item.setMeshData(vertexes=self.nodes, faces=self.faces)
        # Oppdater bjelke med de samme delte nodene → følger automatisk
        self.beam_item.setData(pos=self.nodes[self.beam_idx])

        self.statusBar().showMessage("Demo-deformasjon brukt: bjelke følger platen (felles noder i mesh).")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = GmshPlateBeamWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
