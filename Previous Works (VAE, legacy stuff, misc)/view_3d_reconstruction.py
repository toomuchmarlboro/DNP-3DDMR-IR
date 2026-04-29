"""
Interactive 3D viewer for reconstructed breast meshes.

Default input layout:
  data/reconstruction_output/per_patient/Patient_*/reconstruction_mesh.ply

Features:
- Cycles through patient meshes with Previous/Next buttons
- Colors mesh by stored temperature values when present
- Dependency-light fallback using matplotlib if pyvista/open3d are unavailable
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.colors import Normalize


try:
    import pyvista as pv  # type: ignore
    HAS_PYVISTA = True
except Exception:
    HAS_PYVISTA = False


try:
    import open3d as o3d  # type: ignore
    HAS_OPEN3D = True
except Exception:
    HAS_OPEN3D = False


class PlyMesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, temperatures: np.ndarray | None):
        self.vertices = vertices
        self.faces = faces
        self.temperatures = temperatures


def discover_meshes(per_patient_root: Path) -> List[Tuple[str, Path]]:
    meshes: List[Tuple[str, Path]] = []
    for patient_dir in sorted(per_patient_root.glob("Patient_*")):
        mesh_path = patient_dir / "reconstruction_mesh.ply"
        if mesh_path.exists():
            meshes.append((patient_dir.name, mesh_path))
    return meshes


def load_ascii_ply(path: Path) -> PlyMesh:
    """Load the ASCII PLY mesh written by reconstruct_3d_breast.py."""
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"Not a PLY file: {path}")

    vertex_count = 0
    face_count = 0
    header_end = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("element vertex"):
            vertex_count = int(stripped.split()[-1])
        elif stripped.startswith("element face"):
            face_count = int(stripped.split()[-1])
        elif stripped == "end_header":
            header_end = idx + 1
            break

    if header_end is None:
        raise ValueError(f"PLY header not terminated: {path}")

    body = lines[header_end:]
    if len(body) < vertex_count + face_count:
        raise ValueError(f"PLY body is shorter than expected: {path}")

    vertices = []
    temperatures = []
    for line in body[:vertex_count]:
        parts = line.strip().split()
        if len(parts) < 3:
            raise ValueError(f"Malformed vertex line in {path}: {line}")
        x, y, z = map(float, parts[:3])
        vertices.append((x, y, z))
        if len(parts) >= 4:
            temperatures.append(float(parts[3]))

    faces = []
    for line in body[vertex_count:vertex_count + face_count]:
        parts = line.strip().split()
        if len(parts) < 4:
            raise ValueError(f"Malformed face line in {path}: {line}")
        n = int(parts[0])
        if n != 3:
            continue
        i0, i1, i2 = map(int, parts[1:4])
        faces.append((i0, i1, i2))

    temp_array = np.asarray(temperatures, dtype=np.float32) if temperatures else None
    return PlyMesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        temperatures=temp_array,
    )


def load_open3d_mesh(path: Path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_triangle_normals():
        mesh.compute_vertex_normals()
    return mesh


def mesh_faces_as_vertex_triangles(mesh: PlyMesh) -> List[np.ndarray]:
    return [mesh.vertices[face] for face in mesh.faces]


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


class MeshViewer:
    def __init__(self, meshes: List[Tuple[str, Path]], use_pyvista: bool = False):
        self.meshes = meshes
        self.use_pyvista = use_pyvista and HAS_PYVISTA
        self.index = 0

        if not self.meshes:
            raise FileNotFoundError("No reconstruction_mesh.ply files found")

        if self.use_pyvista:
            self._show_pyvista()
            return

        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection="3d")
        plt.subplots_adjust(bottom=0.18)

        ax_prev = plt.axes([0.30, 0.05, 0.12, 0.06])
        ax_next = plt.axes([0.58, 0.05, 0.12, 0.06])
        self.btn_prev = Button(ax_prev, "Previous")
        self.btn_next = Button(ax_next, "Next")
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)

        self._draw_current()
        plt.show()

    def _show_pyvista(self):
        name, path = self.meshes[self.index]
        mesh = pv.read(str(path))
        plotter = pv.Plotter()
        plotter.add_text(f"{name}", font_size=12)
        plotter.add_mesh(mesh, scalars="temperature" if "temperature" in mesh.array_names else None,
                         cmap="inferno", show_scalar_bar=True)
        plotter.show()

    def _draw_current(self):
        patient_id, mesh_path = self.meshes[self.index]
        mesh = load_ascii_ply(mesh_path)

        self.ax.clear()
        tris = mesh_faces_as_vertex_triangles(mesh)

        if mesh.temperatures is not None and len(mesh.temperatures) == len(mesh.vertices):
            face_temps = mesh.temperatures[mesh.faces].mean(axis=1)
            norm = Normalize(vmin=float(np.min(face_temps)), vmax=float(np.max(face_temps)))
            colors = cm.inferno(norm(face_temps))
        else:
            colors = (0.7, 0.7, 0.8, 0.9)

        poly = Poly3DCollection(tris, linewidths=0.03, alpha=0.95)
        poly.set_facecolor(colors)
        poly.set_edgecolor((0.1, 0.1, 0.1, 0.08))
        self.ax.add_collection3d(poly)

        verts = mesh.vertices
        self.ax.set_xlim(float(verts[:, 0].min()), float(verts[:, 0].max()))
        self.ax.set_ylim(float(verts[:, 1].min()), float(verts[:, 1].max()))
        self.ax.set_zlim(float(verts[:, 2].min()), float(verts[:, 2].max()))
        set_axes_equal(self.ax)
        self.ax.view_init(elev=20, azim=35)
        self.ax.set_title(f"{patient_id}  |  {mesh_path.name}")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        plt.draw()

    def _on_prev(self, _event):
        self.index = (self.index - 1) % len(self.meshes)
        self._draw_current()

    def _on_next(self, _event):
        self.index = (self.index + 1) % len(self.meshes)
        self._draw_current()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View reconstructed 3D breast meshes")
    parser.add_argument(
        "--input-root",
        type=str,
        default="data/reconstruction_output/per_patient",
        help="Folder containing Patient_*/reconstruction_mesh.ply",
    )
    parser.add_argument(
        "--use-pyvista",
        action="store_true",
        help="Use pyvista if installed instead of matplotlib",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = script_dir / input_root

    meshes = discover_meshes(input_root)
    if not meshes:
        raise FileNotFoundError(f"No meshes found under: {input_root}")

    MeshViewer(meshes, use_pyvista=args.use_pyvista)


if __name__ == "__main__":
    main()
