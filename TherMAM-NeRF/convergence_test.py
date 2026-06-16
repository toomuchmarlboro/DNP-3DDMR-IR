"""Rigorous FEM convergence study — mesh-INDEPENDENT metric.
Evaluates the solution at a FIXED set of physical probe points (identical across all
meshes), so re-triangulation of the surface no longer injects noise. Reports the RMS
change between successive meshes (THE convergence indicator: must shrink if converging).
Refines down to 0.5 mm (RAM-gated). Writes incrementally; detached/disconnect-proof.
"""
import sys, math, time, os
import numpy as np
from mpi4py import MPI
import dolfinx, dolfinx.mesh, dolfinx.geometry
from dolfinx.io import gmsh as gmshio
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
import ufl, gmsh

K_TISSUE=0.48; PERFUSION=0.0005*1.0*3600.0; T_ARTERIAL=37.0; Q_METAB=450.0
H_CONV=10.0; T_AIR=20.0; CHEST_WALL_FRAC=0.10
def Qmax_from_r(r_mm):
    D=2.0*r_mm*1e-3; tau=max(50.0+math.log(D/0.01)/0.002134,5.0); return 3.27e6/tau
R_T_MM=15.0; QMAX=Qmax_from_r(R_T_MM)

STL="/mnt/Data1/Peoples/faiz836b/DNP-3DDMR-IR/TherMAM-NeRF/PINNpdeSolver/results/Patient_1/Patient_1_syn.stl"
SIZES=[4.0,3.0,2.0,1.5,1.0,0.7,0.5]
RESULTS="/tmp/convergence_results2.txt"
N_PROBE=8000

def mem_avail_gb():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"): return int(ln.split()[1])/1e6
    return 999

def make_mesh(stl,out,size):
    gmsh.initialize(); gmsh.option.setNumber('General.Verbosity',0)
    try:
        gmsh.model.add('b'); gmsh.merge(stl)
        gmsh.model.mesh.classifySurfaces(np.pi,True,True,2*np.pi)
        gmsh.model.mesh.createGeometry()
        s=gmsh.model.getEntities(2); l=gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
        gmsh.model.geo.addVolume([l]); gmsh.model.geo.synchronize()
        v=gmsh.model.getEntities(3); ss=gmsh.model.getEntities(2)
        gmsh.model.addPhysicalGroup(3,[e[1] for e in v],1)
        gmsh.model.addPhysicalGroup(2,[e[1] for e in ss],2)
        gmsh.option.setNumber('Mesh.CharacteristicLengthMax',size)
        gmsh.option.setNumber('Mesh.CharacteristicLengthMin',size*0.5)
        gmsh.option.setNumber('Mesh.Algorithm3D',1)
        gmsh.model.mesh.generate(3)   # Netgen optimize DISABLED for speed (convergence study)
        gmsh.write(out)
    finally: gmsh.finalize()

def solve_and_probe(msh_path, probe_pts):
    fea=gmshio.read_from_msh(msh_path,MPI.COMM_WORLD,gdim=3)
    msh=fea.mesh; msh.geometry.x[:]*=1e-3
    V=fem.functionspace(msh,('Lagrange',1))
    T=ufl.TrialFunction(V); v=ufl.TestFunction(V); x=ufl.SpatialCoordinate(msh)
    cen=msh.geometry.x.mean(0); x_t,y_t,z_t=cen; r_t=R_T_MM*1e-3
    d2=(x[0]-x_t)**2+(x[1]-y_t)**2+(x[2]-z_t)**2
    Q_tumor=QMAX*ufl.exp(-d2/(r_t**2+1e-16))
    k,perf,Ta,Qm,h,T_air=K_TISSUE,PERFUSION,T_ARTERIAL,Q_METAB,H_CONV,T_AIR
    zc=msh.geometry.x[:,2]; z_lo,z_hi=zc.min(),zc.max(); span=(z_hi-z_lo)+1e-12
    cut=z_lo+CHEST_WALL_FRAC*span
    facets=dolfinx.mesh.locate_entities_boundary(msh,msh.topology.dim-1,lambda p:p[2]<cut)
    dofs=fem.locate_dofs_topological(V,msh.topology.dim-1,facets)
    bc=fem.dirichletbc(np.float64(Ta),dofs,V)
    a=(k*ufl.inner(ufl.grad(T),ufl.grad(v))+perf*T*v)*ufl.dx+h*T*v*ufl.ds
    L=(perf*Ta+Qm+Q_tumor)*v*ufl.dx+h*T_air*v*ufl.ds
    try:
        prob=LinearProblem(a,L,bcs=[bc],petsc_options_prefix='cv_',
                petsc_options={'ksp_type':'cg','pc_type':'gamg','ksp_rtol':1e-10})
    except TypeError:
        prob=LinearProblem(a,L,bcs=[bc],
                petsc_options={'ksp_type':'cg','pc_type':'gamg','ksp_rtol':1e-10})
    Tsol=prob.solve(); ksp=prob.solver
    reason=ksp.getConvergedReason(); iters=ksp.getIterationNumber(); res=ksp.getResidualNorm()
    ndof=V.dofmap.index_map.size_global
    # mesh-independent eval at FIXED probe points (meters)
    tree=dolfinx.geometry.bb_tree(msh,msh.topology.dim)
    cand=dolfinx.geometry.compute_collisions_points(tree,probe_pts)
    coll=dolfinx.geometry.compute_colliding_cells(msh,cand,probe_pts)
    vals=np.full(len(probe_pts),np.nan); cells=[]; idx=[]
    for i in range(len(probe_pts)):
        lk=coll.links(i)
        if len(lk)>0: cells.append(lk[0]); idx.append(i)
    if idx:
        vals[idx]=Tsol.eval(probe_pts[idx],cells).reshape(-1)
    return ndof,reason,iters,res,vals

REASONS={1:'CONV_RTOL_NORMAL',2:'CONV_RTOL',3:'CONV_ATOL',4:'CONV_ITS',
         -3:'DIV_ITS',-4:'DIV_DTOL',-9:'DIV_NANINF'}

def w(s):
    with open(RESULTS,"a") as f: f.write(s+"\n"); f.flush()

# fixed probe points from the existing 1.5mm mesh bbox (meters), seeded -> identical for all
fea0=gmshio.read_from_msh("/tmp/conv_mesh_1.5.msh",MPI.COMM_WORLD,gdim=3)
m0=fea0.mesh; m0.geometry.x[:]*=1e-3
lo=m0.geometry.x.min(0); hi=m0.geometry.x.max(0)
rng=np.random.default_rng(0)
probe=rng.uniform(lo,hi,size=(N_PROBE,3)).astype(np.float64)
del fea0,m0

open(RESULTS,"w").close()
w(f"RIGOROUS convergence study — fixed {N_PROBE} probe points, mesh-independent metric")
w(f"STL: {STL}   tumour r={R_T_MM}mm  Qmax={QMAX:.0f} W/m3")
w(f"{'size':>5} {'DOFs':>10} {'KSP':>16} {'it':>3} {'resid':>10} "
  f"{'probeMeanT':>11} {'probeMaxT':>10} {'nValid':>7} {'RMSvsPrev':>11}")
prev_vals=None
for sz in SIZES:
    need_gb = (742370*(1.5/sz)**3)*1.5e-6
    if need_gb > mem_avail_gb()-4:
        w(f"{sz:5.1f}  SKIPPED — needs ~{need_gb:.0f}GB, only {mem_avail_gb():.0f}GB avail")
        continue
    out=f"/tmp/conv_mesh_{sz}.msh"
    t0=time.time()
    try:
        if not os.path.exists(out): make_mesh(STL,out,sz)
        ndof,reason,iters,res,vals=solve_and_probe(out,probe)
        both = ~np.isnan(vals)
        meanT=np.nanmean(vals); maxT=np.nanmax(vals); nval=int(both.sum())
        if prev_vals is None:
            rms_s="    --"
        else:
            common = both & ~np.isnan(prev_vals)
            rms = float(np.sqrt(np.mean((vals[common]-prev_vals[common])**2)))
            rms_s=f"{rms:.5f}C"
        w(f"{sz:5.1f} {ndof:10d} {REASONS.get(reason,reason):>16} {iters:3d} {res:10.2e} "
          f"{meanT:11.4f} {maxT:10.4f} {nval:7d} {rms_s:>11}  ({time.time()-t0:.0f}s)")
        prev_vals=vals
    except Exception as e:
        w(f"{sz:5.1f}  FAILED: {repr(e)[:100]}")
w("DONE. RMSvsPrev = RMS temp change at the SAME physical points between successive meshes.")
w("Converging <=> RMSvsPrev shrinks toward 0 as size decreases.")
