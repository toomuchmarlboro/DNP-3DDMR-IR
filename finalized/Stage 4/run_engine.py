#!/usr/bin/env python
"""
Levenberg-Marquardt FEM inverse bioheat — cohort runner.

Method: Gonzalez-Hernandez / Kandlikar et al., Infrared Physics & Technology 105 (2020).
Forward: steady Pennes BVP in FEniCSx (3mm tet mesh per patient).
Inverse: LM over β = [x_t, y_t, z_t, d], cost = direct MSE skin temperatures.
Init:    differential hotspot (ΔT = T_target − T_healthy) → tumour XY prior.

Run in tmux (unbuffered, all outputs to lm_results/):
    conda activate bioheat
    cd /mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR

    # Step 1 — synthetic validation (N random placements per patient)
    python -u finalized/run_lm_cohort.py --synthetic --n-scenarios 20

    # Step 2 — real cohort
    python -u finalized/run_lm_cohort.py --real --all

    # Utilities
    python -u finalized/run_lm_cohort.py --list
    python -u finalized/run_lm_cohort.py --synthetic --patients Patient_1,Patient_408
    python -u finalized/run_lm_cohort.py --real --subset 10
"""

import os, sys, time, csv, argparse
from pathlib import Path

import numpy as np
import torch
import pandas as pd

# ── repo paths ────────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / 'TherMAM-NeRF'))

import run_cohort as R   # dataset, geometry, mesh tools

OUT_DIR = HERE / 'lm_results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Kandlikar 2020 paper constants (Table 3) ──────────────────────────────────
PAPER = dict(
    k_h     = 0.42,
    rho_b   = 1060.0,
    c_b     = 3840.0,
    omega_h = 1.8e-4,
    omega_t = 9.0e-3,
    Q_h     = 450.0,
    T_a     = 37.0,
    T_c     = 37.0,
    h       = 13.5,
    T_inf   = 21.0,
)
PERF_H = PAPER['omega_h'] * PAPER['rho_b'] * PAPER['c_b']
PERF_T = PAPER['omega_t'] * PAPER['rho_b'] * PAPER['c_b']

# quadrant literature distribution (Senie 1983)
QUAD_DIST  = {'UO': 0.50, 'C': 0.18, 'UI': 0.15, 'LO': 0.11, 'LI': 0.06}
QUAD_NAMES = {'UO': 'Upper-Outer', 'UI': 'Upper-Inner',
              'LO': 'Lower-Outer', 'LI': 'Lower-Inner', 'C': 'Central'}

# LM step sizes matched to mesh element size (§9.2 justification)
LM_STEP  = {'x_t_mm': 3.0, 'y_t_mm': 3.0, 'z_t_mm': 3.0, 'd_mm': 2.0}
PNAMES   = ['x_t_mm', 'y_t_mm', 'z_t_mm', 'd_mm']

# ── FEM forward solve ─────────────────────────────────────────────────────────

def pennes_forward(msh_path, beta, geo, const=PAPER):
    """One-shot Pennes solve. Returns (T_dolfinx_fn, msh)."""
    from mpi4py import MPI
    import dolfinx, dolfinx.mesh
    from dolfinx.io import gmsh as gmshio
    from dolfinx import fem
    from dolfinx.fem.petsc import LinearProblem
    import ufl

    fea = gmshio.read_from_msh(str(msh_path), MPI.COMM_WORLD, gdim=3)
    msh = fea.mesh
    msh.geometry.x[:] *= 1e-3

    Vs = fem.functionspace(msh, ('Lagrange', 1))
    T  = ufl.TrialFunction(Vs); v = ufl.TestFunction(Vs)
    Xc = ufl.SpatialCoordinate(msh)

    k_h   = float(beta.get('k_h', const['k_h']))
    ph    = const['omega_h'] * const['rho_b'] * const['c_b']
    pt    = const['omega_t'] * const['rho_b'] * const['c_b']
    Ta, Qh, h_, Ta_ = const['T_a'], const['Q_h'], const['h'], const['T_inf']

    xt = beta['x_t_mm']*1e-3; yt = beta['y_t_mm']*1e-3; zt = beta['z_t_mm']*1e-3
    rt = (max(float(beta['d_mm']), 1.0) / 2.0) * 1e-3
    Qt = float(R.tumor_Qm_from_radius(torch.tensor(beta['d_mm'] / 2.0)).item())

    d2  = (Xc[0]-xt)**2 + (Xc[1]-yt)**2 + (Xc[2]-zt)**2
    chi = ufl.exp(-d2 / (rt**2 + 1e-16))
    P   = ph + (pt - ph)*chi
    Q   = Qh + (Qt - Qh)*chi

    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    ext  = dolfinx.mesh.exterior_facet_indices(msh.topology)
    mid  = dolfinx.mesh.compute_midpoints(msh, fdim, ext)
    mz   = mid[:, 2]; mz0, mz1 = float(mz.min()), float(mz.max())
    post = mz > (mz0 + (1.0 - R.CHEST_WALL_FRAC) * (mz1 - mz0))
    bc   = fem.dirichletbc(np.float64(Ta),
                           fem.locate_dofs_topological(Vs, fdim, ext[post]), Vs)

    # Robin BC defined here
    a = (k_h*ufl.inner(ufl.grad(T), ufl.grad(v)) + P*T*v)*ufl.dx + h_*T*v*ufl.ds
    L = (P*Ta + Q)*v*ufl.dx + h_*Ta_*v*ufl.ds
    try:
        prob = LinearProblem(a, L, bcs=[bc], petsc_options_prefix='lm_',
                             petsc_options={'ksp_type': 'cg', 'pc_type': 'gamg'})
    except TypeError:
        prob = LinearProblem(a, L, bcs=[bc],
                             petsc_options={'ksp_type': 'cg', 'pc_type': 'gamg'})
    return prob.solve(), msh


class PennesCtx:
    """Reads 3mm mesh ONCE; re-solves UFL per β call. Amortises mesh I/O."""

    def __init__(self, msh_path, geo, const=PAPER):
        from mpi4py import MPI
        import dolfinx, dolfinx.mesh
        from dolfinx.io import gmsh as gmshio
        from dolfinx import fem
        from scipy.spatial import cKDTree
        import ufl

        fea = gmshio.read_from_msh(str(msh_path), MPI.COMM_WORLD, gdim=3)
        self.msh = fea.mesh; self.msh.geometry.x[:] *= 1e-3
        self.V   = fem.functionspace(self.msh, ('Lagrange', 1))
        self.Xc  = ufl.SpatialCoordinate(self.msh)
        self.const = const

        # chest-wall Dirichlet (posterior Z-slab on facet midpoints)
        fdim = self.msh.topology.dim - 1
        self.msh.topology.create_connectivity(fdim, self.msh.topology.dim)
        ext = dolfinx.mesh.exterior_facet_indices(self.msh.topology)
        mid = dolfinx.mesh.compute_midpoints(self.msh, fdim, ext)
        mz  = mid[:, 2]; mz0, mz1 = float(mz.min()), float(mz.max())
        post = mz > (mz0 + (1.0 - R.CHEST_WALL_FRAC) * (mz1 - mz0))
        self.bc = fem.dirichletbc(
            np.float64(const['T_a']),
            fem.locate_dofs_topological(self.V, fdim, ext[post]), self.V)

        # surface interpolation index (3 nearest tet nodes → mean)
        _, self.sidx = cKDTree(self.msh.geometry.x * 1e3).query(
            geo['surface_pts'], k=3)
        self.n_nodes = self.msh.geometry.x.shape[0]

    def solve(self, beta):
        from dolfinx import fem
        from dolfinx.fem.petsc import LinearProblem
        import ufl

        c   = self.const
        T   = ufl.TrialFunction(self.V); v = ufl.TestFunction(self.V)
        k_h = float(beta.get('k_h', c['k_h']))
        ph  = c['omega_h']*c['rho_b']*c['c_b']
        pt  = c['omega_t']*c['rho_b']*c['c_b']
        Ta, Qh, h_, Ta_ = c['T_a'], c['Q_h'], c['h'], c['T_inf']

        xt = beta['x_t_mm']*1e-3; yt = beta['y_t_mm']*1e-3; zt = beta['z_t_mm']*1e-3
        dm = max(float(beta['d_mm']), 1.0); rt = (dm / 2.0) * 1e-3
        Qt = float(R.tumor_Qm_from_radius(torch.tensor(dm / 2.0)).item())

        d2  = (self.Xc[0]-xt)**2 + (self.Xc[1]-yt)**2 + (self.Xc[2]-zt)**2
        chi = ufl.exp(-d2 / (rt**2 + 1e-16))
        P   = ph + (pt - ph)*chi; Q = Qh + (Qt - Qh)*chi

        a = (k_h*ufl.inner(ufl.grad(T), ufl.grad(v)) + P*T*v)*ufl.dx + h_*T*v*ufl.ds
        L = (P*Ta + Q)*v*ufl.dx + h_*Ta_*v*ufl.ds
        try:
            prob = LinearProblem(a, L, bcs=[self.bc], petsc_options_prefix='inv_',
                                 petsc_options={'ksp_type': 'cg', 'pc_type': 'gamg'})
        except TypeError:
            prob = LinearProblem(a, L, bcs=[self.bc],
                                 petsc_options={'ksp_type': 'cg', 'pc_type': 'gamg'})
        sol = prob.solve()
        return sol.x.array[self.sidx].mean(axis=1)


# ── geometry helpers ──────────────────────────────────────────────────────────

def skin_mask(geo):
    z = geo['surface_pts'][:, 2]
    z0, z1 = float(z.min()), float(z.max())
    return z < (z0 + (1.0 - R.CHEST_WALL_FRAC) * (z1 - z0))


def quadrants(geo):
    sp  = geo['surface_pts']
    skn = skin_mask(geo)
    Tm  = geo['T_measured']
    x_mid = float(sp[:, 0].mean())
    hot   = int(np.argmax(np.where(skn, Tm, -1e9)))
    side_pos = bool(sp[hot, 0] > x_mid)
    side   = (sp[:, 0] > x_mid) if side_pos else (sp[:, 0] < x_mid)
    breast = skn & side
    c      = sp[breast].mean(0)
    outer  = (sp[:, 0] > c[0]) if side_pos else (sp[:, 0] < c[0])
    upper  = sp[:, 1] < c[1]
    bsp    = sp[breast]
    nipple = bsp[np.argmin(bsp[:, 2])]
    dist_n = np.linalg.norm(sp - nipple, axis=1)
    central = breast & (dist_n < 25.0)
    code = np.full(len(sp), 'other', dtype=object)
    code[breast &  upper &  outer & ~central] = 'UO'
    code[breast &  upper & ~outer & ~central] = 'UI'
    code[breast & ~upper &  outer & ~central] = 'LO'
    code[breast & ~upper & ~outer & ~central] = 'LI'
    code[central] = 'C'
    return code, c, nipple, x_mid, side_pos


def plant_tumor(geo, qcode, c_aff, rng):
    sp  = geo['surface_pts']
    nrm = geo['vertex_normals']
    labels = list(QUAD_DIST.keys())
    probs  = [QUAD_DIST[k] for k in labels]
    quad   = rng.choice(labels, p=probs)
    mask   = (qcode == quad)
    if mask.sum() < 10:
        quad = 'UO'; mask = (qcode == 'UO')
    patch_pts = sp[mask]; patch_nrm = nrm[mask]
    c_surf    = patch_pts.mean(0)
    inward    = -patch_nrm.mean(0)
    inward   /= np.linalg.norm(inward) + 1e-9
    d_mm      = float(rng.uniform(10.0, 20.0))
    skin_clr  = float(rng.uniform(5.0, 20.0))
    depth     = d_mm / 2.0 + skin_clr
    pt        = c_surf + depth * inward
    beta      = dict(x_t_mm=float(pt[0]), y_t_mm=float(pt[1]),
                     z_t_mm=float(pt[2]), d_mm=d_mm, k_h=PAPER['k_h'])
    return beta, quad, skin_clr


# ── LM inverse ────────────────────────────────────────────────────────────────

def run_lm_full(ctx, T_target, skn, surface_pts_skin, bbmin, bbmax,
                max_iter=25, mu0=1e-2, verbose=True, mean_sub=False):
    """Full LM inverse with differential-hotspot init. Returns result dict."""

    # Healthy baseline (tumour outside bbox → χ ≈ 0)
    beta_hlth = dict(x_t_mm=float(bbmax[0])+1000.0,
                     y_t_mm=0.0, z_t_mm=0.0, d_mm=1.0, k_h=PAPER['k_h'])
    T_hlth   = ctx.solve(beta_hlth)[skn]
    n_solves  = 1

    dT       = T_target - T_hlth
    hot_idx  = int(np.argmax(dT))
    z_init   = float(0.5 * (bbmin[2] + bbmax[2]))
    x        = np.array([surface_pts_skin[hot_idx, 0],
                         surface_pts_skin[hot_idx, 1],
                         z_init, 12.0])

    def vec2beta(xv):
        b = {k: float(v) for k, v in zip(PNAMES, xv)}
        b['d_mm'] = max(b['d_mm'], 1.0); b['k_h'] = PAPER['k_h']
        return b

    def resid(xv):
        T_m = ctx.solve(vec2beta(xv))[skn]
        if mean_sub:
            return (T_target - T_target.mean()) - (T_m - T_m.mean())
        return T_target - T_m

    r = resid(x); S = float(np.mean(r**2)); n_solves += 1
    mu = mu0

    hist = {k: [float(x[i])] for i, k in enumerate(PNAMES)}
    hist['RMS']  = [float(np.sqrt(S))]
    hist['mu']   = [float(mu)]
    hist['iter'] = [0]

    if verbose:
        print(f'    init  RMS={np.sqrt(S):.4f}°C  '
              f'start=({x[0]:.0f},{x[1]:.0f},{x[2]:.0f}) d={x[3]:.1f}mm',
              flush=True)

    converged = False
    for it in range(1, max_iter + 1):
        J = np.empty((r.size, len(PNAMES)))
        for j, name in enumerate(PNAMES):
            xp = x.copy(); xp[j] += LM_STEP[name]
            J[:, j] = (resid(xp) - r) / LM_STEP[name]
            n_solves += 1

        JtJ = J.T @ J; Jtr = J.T @ r
        accepted = False
        for _ in range(8):
            A  = JtJ + mu * np.diag(np.diag(JtJ)) + 1e-9 * np.eye(len(x))
            dx = np.linalg.solve(A, -Jtr)
            x_tr = np.array([
                np.clip(x[0]+dx[0], bbmin[0]-15, bbmax[0]+15),
                np.clip(x[1]+dx[1], bbmin[1]-15, bbmax[1]+15),
                np.clip(x[2]+dx[2], bbmin[2]-15, bbmax[2]+15),
                np.clip(x[3]+dx[3], 1.0, 35.0),
            ])
            rn = resid(x_tr); Sn = float(np.mean(rn**2)); n_solves += 1
            if Sn < S:
                x, r, S = x_tr, rn, Sn
                mu = max(mu * 0.3, 1e-9); accepted = True; break
            mu = min(mu * 3.0, 1e8)

        for i, k in enumerate(PNAMES):
            hist[k].append(float(x[i]))
        hist['RMS'].append(float(np.sqrt(S))); hist['mu'].append(float(mu))
        hist['iter'].append(it)

        if verbose:
            tag = '' if accepted else ' [rejected]'
            print(f'    iter {it:2d}  RMS={np.sqrt(S):.4f}°C  mu={mu:.1e}  '
                  f'β=({x[0]:.1f},{x[1]:.1f},{x[2]:.1f},{x[3]:.1f}){tag}',
                  flush=True)

        if not accepted:
            break
        if np.linalg.norm(dx) < 0.05 or S < 1e-6:
            converged = True; break

    beta_hat = vec2beta(x)
    pad = 5.0
    outside = (any(beta_hat[k] < bbmin[i] - pad
                   for i, k in enumerate(['x_t_mm', 'y_t_mm', 'z_t_mm'])) or
               any(beta_hat[k] > bbmax[i] + pad
                   for i, k in enumerate(['x_t_mm', 'y_t_mm', 'z_t_mm'])))
    verdict = 'BENIGN' if outside or beta_hat['d_mm'] < 4.0 else 'MALIGNANT'

    return dict(
        x_t=beta_hat['x_t_mm'], y_t=beta_hat['y_t_mm'],
        z_t=beta_hat['z_t_mm'], d=beta_hat['d_mm'],
        final_rms=float(np.sqrt(S)),
        n_solves=n_solves, converged=converged,
        verdict=verdict, lm_hist=hist,
    )


# ── CSV checkpoint helpers ────────────────────────────────────────────────────

def done_ids(csv_path, key='patient_id'):
    if not Path(csv_path).exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=[key])
        return set(df[key].dropna().unique())
    except Exception:
        return set()


def append_row(csv_path, row):
    path = Path(csv_path)
    write_header = not path.exists()
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


# ── patient selection (mirrors run_cohort.py) ─────────────────────────────────

def select_indices(patients, args):
    if args.patients:
        ids = set(args.patients.split(','))
        return [i for i, p in enumerate(patients) if p['id'] in ids]
    if args.indices:
        return [int(x) for x in args.indices.split(',')]
    if args.subset:
        mal = [i for i, p in enumerate(patients) if p['label'] == 'malignant']
        ben = [i for i, p in enumerate(patients) if p['label'] == 'benign']
        n   = args.subset // 2
        return sorted(mal[:n] + ben[:n])
    idxs = list(range(len(patients)))
    if args.limit:
        idxs = idxs[:args.limit]
    return idxs


# ── per-patient runners ───────────────────────────────────────────────────────

def ensure_mesh(geo, pid, stl_path=None):
    """Generate 3mm mesh if not cached. Returns Path to .msh."""
    msh_3mm = R.STL_DIR / f'{pid}_3mm.msh'
    stl     = R.STL_DIR / f'{pid}.stl'
    if not msh_3mm.exists():
        R.save_stl_binary(str(stl), geo['surface_pts'], geo['faces'])
        print(f'  mesh: generating 3mm...', end=' ', flush=True)
        t0 = time.time()
        R.stl_to_tet_mesh(str(stl), str(msh_3mm), mesh_size_mm=3.0)
        print(f'done ({time.time()-t0:.0f}s)', flush=True)
    return msh_3mm


def run_patient_synthetic(dataset, idx, patients, encoder, mlp, args, rng):
    pid = patients[idx]['id']
    lab = patients[idx]['label']
    print(f'\n  loading geometry...', end=' ', flush=True)
    geo = R.load_patient_geo(
        dataset, idx, encoder, mlp, R.CFG, R.device,
        gaussian_sigma=R.GAUSSIAN_SIGMA, mc_threshold_override=R.MC_THRESHOLD,
        repair_holes=R.REPAIR_HOLES, laplacian_iters=R.LAPLACIAN_ITERS,
        thermal_sigma=R.THERMAL_SIGMA)
    print(f'{len(geo["surface_pts"]):,} verts', flush=True)

    msh_3mm = ensure_mesh(geo, pid)
    print(f'  building PennesCtx...', end=' ', flush=True)
    ctx = PennesCtx(msh_3mm, geo)
    print(f'{ctx.n_nodes:,} nodes', flush=True)

    skn    = skin_mask(geo)
    sp_skn = geo['surface_pts'][skn]
    bbmin  = np.asarray(geo['bbox_min'], float)
    bbmax  = np.asarray(geo['bbox_max'], float)

    qcode, c_aff, nipple_pt, x_mid, side_pos = quadrants(geo)

    rows = []
    for sc in range(args.n_scenarios):
        plant, quad, skin_clr = plant_tumor(geo, qcode, c_aff, rng)
        print(f'\n  sc {sc+1:2d}/{args.n_scenarios}'
              f'  quad={quad}  d={plant["d_mm"]:.1f}mm  skin={skin_clr:.1f}mm'
              f'  plant=({plant["x_t_mm"]:.0f},{plant["y_t_mm"]:.0f},{plant["z_t_mm"]:.0f})',
              flush=True)

        # Self-consistent synthetic target on same mesh → zero floor
        T_target = ctx.solve(plant)[skn]
        t0 = time.time()
        res = run_lm_full(ctx, T_target, skn, sp_skn, bbmin, bbmax,
                          max_iter=25, verbose=args.verbose)
        elapsed = time.time() - t0

        pos_err = float(np.sqrt((res['x_t'] - plant['x_t_mm'])**2 +
                                (res['y_t'] - plant['y_t_mm'])**2 +
                                (res['z_t'] - plant['z_t_mm'])**2))
        d_err   = abs(res['d'] - plant['d_mm'])
        ok      = '✓' if res['verdict'] == 'MALIGNANT' else '✗'
        print(f'    → {ok} {res["verdict"]}  '
              f'pos_err={pos_err:.1f}mm  d_err={d_err:.1f}mm  '
              f'RMS={res["final_rms"]:.2e}  {res["n_solves"]} solves  {elapsed:.0f}s',
              flush=True)

        rows.append(dict(
            patient_id=pid, label=lab, mode='synthetic',
            scenario=sc+1, quad=quad, skin_clr_mm=round(skin_clr, 1),
            planted_x=round(plant['x_t_mm'], 1),
            planted_y=round(plant['y_t_mm'], 1),
            planted_z=round(plant['z_t_mm'], 1),
            planted_d=round(plant['d_mm'], 1),
            recovered_x=round(res['x_t'], 1),
            recovered_y=round(res['y_t'], 1),
            recovered_z=round(res['z_t'], 1),
            recovered_d=round(res['d'], 1),
            pos_err_mm=round(pos_err, 2),
            d_err_mm=round(d_err, 2),
            final_rms=round(res['final_rms'], 6),
            converged=res['converged'],
            verdict=res['verdict'],
            n_solves=res['n_solves'],
            time_s=round(elapsed, 1),
            status='OK',
        ))

    return rows


def run_patient_real(dataset, idx, patients, encoder, mlp, args):
    pid = patients[idx]['id']
    lab = patients[idx]['label']
    print(f'\n  loading geometry...', end=' ', flush=True)
    geo = R.load_patient_geo(
        dataset, idx, encoder, mlp, R.CFG, R.device,
        gaussian_sigma=R.GAUSSIAN_SIGMA, mc_threshold_override=R.MC_THRESHOLD,
        repair_holes=R.REPAIR_HOLES, laplacian_iters=R.LAPLACIAN_ITERS,
        thermal_sigma=R.THERMAL_SIGMA)
    print(f'{len(geo["surface_pts"]):,} verts', flush=True)

    msh_3mm = ensure_mesh(geo, pid)
    print(f'  building PennesCtx...', end=' ', flush=True)
    ctx = PennesCtx(msh_3mm, geo)
    print(f'{ctx.n_nodes:,} nodes', flush=True)

    skn    = skin_mask(geo)
    sp_skn = geo['surface_pts'][skn]
    bbmin  = np.asarray(geo['bbox_min'], float)
    bbmax  = np.asarray(geo['bbox_max'], float)

    T_target = geo['T_measured'][skn].astype(float)
    t0       = time.time()
    res      = run_lm_full(ctx, T_target, skn, sp_skn, bbmin, bbmax,
                           max_iter=25, verbose=args.verbose, mean_sub=True)
    elapsed  = time.time() - t0

    ok = '✓' if res['verdict'] == 'MALIGNANT' else '○'
    print(f'    → {ok} {res["verdict"]}  '
          f'β=({res["x_t"]:.0f},{res["y_t"]:.0f},{res["z_t"]:.0f}) d={res["d"]:.1f}mm  '
          f'RMS={res["final_rms"]:.4f}°C  {res["n_solves"]} solves  {elapsed:.0f}s',
          flush=True)

    return dict(
        patient_id=pid, label=lab, mode='real',
        recovered_x=round(res['x_t'], 1),
        recovered_y=round(res['y_t'], 1),
        recovered_z=round(res['z_t'], 1),
        recovered_d=round(res['d'], 1),
        final_rms=round(res['final_rms'], 6),
        converged=res['converged'],
        verdict=res['verdict'],
        n_solves=res['n_solves'],
        time_s=round(elapsed, 1),
        status='OK',
    )


# ── main runners ──────────────────────────────────────────────────────────────

def _print_banner(mode, n_patients, out_csv, args):
    print('=' * 72)
    print(f'  LM FEM Inverse Bioheat — {mode.upper()} MODE')
    print(f'  Kandlikar 2020 | FEniCSx 3mm mesh | 4-DoF LM')
    print(f'  Patients : {n_patients}')
    if mode == 'synthetic':
        print(f'  Scenarios: {args.n_scenarios} per patient')
    print(f'  Output   : {out_csv}')
    print(f'  Resume   : {args.resume}')
    print('=' * 72, flush=True)


def run_synthetic_cohort(args):
    import datetime
    if getattr(args, 'out', None):
        out_csv = Path(args.out)
    else:
        stamp   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_csv = OUT_DIR / f'lm_synthetic_{stamp}.csv'
    rng     = np.random.default_rng(args.seed)

    dataset, patients = R.build_dataset()
    encoder, mlp      = R.load_models()
    idxs              = select_indices(patients, args)
    done              = done_ids(out_csv, 'patient_id') if args.resume else set()

    _print_banner('synthetic', len(idxs), out_csv, args)

    t0_all = time.time()
    for n, idx in enumerate(idxs, 1):
        pid = patients[idx]['id']
        lab = patients[idx]['label']
        print(f'\n[{n:03d}/{len(idxs)}] {pid} ({lab})', flush=True)

        if args.resume and pid in done:
            print(f'  skipped (already in CSV)', flush=True)
            continue

        ts = time.time()
        try:
            rows = run_patient_synthetic(dataset, idx, patients, encoder, mlp, args, rng)
            for row in rows:
                append_row(out_csv, row)
        except Exception:
            import traceback; traceback.print_exc()
            append_row(out_csv, dict(patient_id=pid, label=lab, mode='synthetic',
                                     status='FAILED'))

        elapsed = time.time() - ts
        print(f'  {pid} done in {elapsed/60:.1f} min', flush=True)

    total = time.time() - t0_all
    print(f'\nFinished {len(idxs)} patients in {total/60:.1f} min.')
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        ok = df[df.status == 'OK']
        print(f'CSV: {len(df)} rows | {len(ok)} successful scenarios')
        if 'pos_err_mm' in ok.columns and len(ok):
            print(f'Position error: mean={ok.pos_err_mm.mean():.1f}mm  '
                  f'max={ok.pos_err_mm.max():.1f}mm  '
                  f'p95={ok.pos_err_mm.quantile(0.95):.1f}mm')
            print(f'Diameter error: mean={ok.d_err_mm.mean():.1f}mm  '
                  f'max={ok.d_err_mm.max():.1f}mm')
            vc = ok.verdict.value_counts()
            print(f'Verdicts (should all be MALIGNANT): {vc.to_dict()}')
    print(f'Output → {out_csv}')


def run_real_cohort(args):
    import datetime
    if getattr(args, 'out', None):
        out_csv = Path(args.out)
    else:
        stamp   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_csv = OUT_DIR / f'lm_real_{stamp}.csv'

    dataset, patients = R.build_dataset()
    encoder, mlp      = R.load_models()
    idxs              = select_indices(patients, args)
    done              = done_ids(out_csv) if args.resume else set()

    _print_banner('real', len(idxs), out_csv, args)

    t0_all = time.time()
    for n, idx in enumerate(idxs, 1):
        pid = patients[idx]['id']
        lab = patients[idx]['label']
        print(f'\n[{n:03d}/{len(idxs)}] {pid} ({lab})', flush=True)

        if args.resume and pid in done:
            print(f'  skipped (already in CSV)', flush=True)
            continue

        ts = time.time()
        try:
            row = run_patient_real(dataset, idx, patients, encoder, mlp, args)
            append_row(out_csv, row)
        except Exception:
            import traceback; traceback.print_exc()
            append_row(out_csv, dict(patient_id=pid, label=lab, mode='real',
                                     status='FAILED'))

        print(f'  {pid} done in {(time.time()-ts)/60:.1f} min', flush=True)

    total = time.time() - t0_all
    print(f'\nFinished {len(idxs)} patients in {total/60:.1f} min.')
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        ok = df[df.status == 'OK']
        print(f'CSV: {len(ok)} rows')
        if len(ok):
            vc = ok.groupby('label')['verdict'].value_counts()
            print('Verdicts by label:\n', vc.to_string())
    print(f'Output → {out_csv}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='LM FEM inverse bioheat cohort runner (tmux-friendly, resumable).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--synthetic', action='store_true',
                      help='synthetic validation: plant known tumours, recover, measure error')
    mode.add_argument('--real',      action='store_true',
                      help='real patient inverse: use measured IR temperatures')
    mode.add_argument('--list',      action='store_true',
                      help='print index→patient map and exit')

    # patient selection
    ap.add_argument('--all',      action='store_true', help='full cohort (default)')
    ap.add_argument('--limit',    type=int,   default=None, help='first N patients')
    ap.add_argument('--subset',   type=int,   default=None, help='balanced N patients')
    ap.add_argument('--patients', type=str,   default=None,
                    help='comma-separated IDs, e.g. Patient_1,Patient_159')
    ap.add_argument('--indices',  type=str,   default=None,
                    help='comma-separated dataset indices')

    # synthetic options
    ap.add_argument('--n-scenarios', type=int, default=10,
                    help='number of random tumour placements per patient (default 10)')
    ap.add_argument('--seed', type=int, default=42,
                    help='global RNG seed for reproducibility (default 42)')

    # run options
    ap.add_argument('--resume',    action='store_true',  default=True,
                    help='skip patients already in the CSV (default on)')
    ap.add_argument('--no-resume', dest='resume', action='store_false',
                    help='reprocess every patient')
    ap.add_argument('--verbose',   action='store_true', default=True,
                    help='print per-LM-iteration log (default on)')
    ap.add_argument('--quiet',     dest='verbose', action='store_false',
                    help='suppress per-iteration log')
    ap.add_argument('--out', type=str, default=None,
                    help='fixed output CSV path (enables resume across runs)')

    args = ap.parse_args()

    if args.list:
        dataset, patients = R.build_dataset()
        for i, p in enumerate(patients):
            print(f'{i:3d} | {p["id"]:25s} | {p["label"]}')
        return

    if args.synthetic:
        run_synthetic_cohort(args)
    else:
        run_real_cohort(args)


if __name__ == '__main__':
    main()