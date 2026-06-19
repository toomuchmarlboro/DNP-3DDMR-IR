"""Per-patient 2x2 input/output visualization (anterior skin-surface view).
For each patient: load NeRF geometry + measured IR (input), solve the forward FEM
at the recovered tumour (z_hat, r_hat), and render a 2x2 grid:
  1. IR measured (input)        2. FEM forward (prediction at recovered tumour)
  3. Residual |FEM - IR|        4. Recovered tumour location on the breast
Output: results/patient_grids/<pid>_grid.png

Usage (tmux, 2 GPUs):
  CUDA_VISIBLE_DEVICES=0 python -u viz_patient_grid.py --stride 2 --offset 0
  CUDA_VISIBLE_DEVICES=1 python -u viz_patient_grid.py --stride 2 --offset 1
Or a single patient:
  python -u viz_patient_grid.py --patients Patient_1
"""
import os, argparse
os.environ["PYVISTA_OFF_SCREEN"] = "true"
import numpy as np, pandas as pd
from pathlib import Path
import pyvista as pv
pv.OFF_SCREEN = True
import run_cohort as R
import mukhmetov_recover as M

COH = Path("results/cohort_femfem_ALL122.csv")
OUTDIR = Path("results/patient_grids"); OUTDIR.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--patients", nargs="+", default=None, help="patient ids; default = all in cohort CSV")
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--offset", type=int, default=0)
args = ap.parse_args()

dataset, patients = R.build_dataset()
encoder, mlp = R.load_models()
id2idx = {p['id']: i for i, p in enumerate(patients)}
coh = pd.read_csv(COH).set_index('patient_id')

ids = args.patients if args.patients else list(coh.index)
ids = ids[args.offset::args.stride]
print(f"rendering {len(ids)} patients (stride={args.stride}, offset={args.offset})", flush=True)

def surf_polydata(geo):
    f = geo['faces'].astype(np.int64)
    faces = np.hstack([np.full((len(f), 1), 3, dtype=np.int64), f]).ravel()
    return pv.PolyData(geo['surface_pts'].astype(float), faces)

def front_cam(surf):
    b = surf.bounds
    cx, cy, cz = (b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2
    # Orientation "B": camera on the -Z side, superior (-Y) up, no mirror.
    # The anterior skin faces -Z in this reconstruction (confirmed vs the Plotly
    # reference), so the -Z camera gives the correct frontal view.
    camz = b[4] - 2.2*max(b[1]-b[0], b[3]-b[2])
    return [(cx, cy, camz), (cx, cy, cz), (0, -1, 0)]

for pid in ids:
    if pid not in id2idx or pid not in coh.index:
        print(f"skip {pid} (no idx/result)", flush=True); continue
    try:
        idx = id2idx[pid]; row = coh.loc[pid]
        x0, y0, z, r = float(row.x0_mm), float(row.y0_mm), float(row.z_hat_mm), float(row.r_hat_mm)
        pdir = R.RESULTS_DIR / pid
        geo, msh = M.load_geo_and_mesh(dataset, idx, encoder, mlp, pdir, mesh_suffix='syn')
        T_fem = np.asarray(M.fem_surface(msh, geo, x0, y0, z, r), dtype=float)
        T_ir  = np.asarray(geo['T_measured'], dtype=float)
        res   = np.abs(T_fem - T_ir)
        surf = surf_polydata(geo)
        surf['IR'] = T_ir; surf['FEM'] = T_fem; surf['RES'] = res
        cam = front_cam(surf)
        # clim from SKIN only (exclude the ~37C chest-wall nodes that would
        # otherwise wash out the skin pattern, esp. in the cold FEM field).
        skin = ~R.chest_wall_mask_from_pts(geo['surface_pts'], geo['vertex_normals'])
        clim_ir  = [float(np.percentile(T_ir[skin], 2)),  float(np.percentile(T_ir[skin], 98))]
        clim_fem = [float(np.percentile(T_fem[skin], 2)), float(np.percentile(T_fem[skin], 98))]

        pl = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(1500, 1500), border=True)
        pl.subplot(0, 0)
        pl.add_mesh(surf.copy(), scalars='IR', cmap='inferno', clim=clim_ir,
                    scalar_bar_args=dict(title='IR (°C)', n_labels=3))
        pl.add_text('1. IR terukur (input)', font_size=10); pl.camera_position = cam
        pl.subplot(0, 1)
        pl.add_mesh(surf.copy(), scalars='FEM', cmap='inferno', clim=clim_fem,
                    scalar_bar_args=dict(title='FEM (°C)', n_labels=3))
        pl.add_text('2. FEM forward (prediksi)', font_size=10); pl.camera_position = cam
        pl.subplot(1, 0)
        pl.add_mesh(surf.copy(), scalars='RES', cmap='turbo',
                    scalar_bar_args=dict(title='|FEM-IR| (°C)', n_labels=3))
        pl.add_text('3. Residual |FEM - IR|', font_size=10); pl.camera_position = cam
        pl.subplot(1, 1)
        pl.add_mesh(surf.copy(), scalars='IR', cmap='inferno', clim=clim_ir, opacity=0.5)
        pl.add_mesh(pv.Sphere(radius=max(r, 3.0), center=(x0, y0, z)),
                    color='cyan', opacity=0.6)
        pl.add_text(f'4. Lokasi tumor inverse\nz={z:.0f} mm  r={r:.0f} mm', font_size=10)
        pl.camera_position = cam
        pl.add_text(f'{pid}  ({row.label})  cost={row.final_cost:.2f}',
                    position='lower_edge', font_size=9, color='black')
        out = OUTDIR / f'{pid}_grid.png'
        pl.screenshot(str(out)); pl.close()
        print(f"saved {out}  (IR {T_ir.min():.1f}-{T_ir.max():.1f}, FEM {T_fem.min():.1f}-{T_fem.max():.1f}, res {res.mean():.2f})", flush=True)
    except Exception as e:
        print(f"FAILED {pid}: {repr(e)[:140]}", flush=True)