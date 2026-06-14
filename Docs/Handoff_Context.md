# Handoff Context — for continuing in the Claude app

Upload this file (plus the files listed at the bottom) to a new chat in the Claude app and say:
*"This is the context for my undergraduate thesis. Help me continue — start with [whatever you need]."*

---

## What the project is
An undergraduate thesis: a non-invasive breast-cancer **screening** pipeline that estimates a tumour's **position, depth, and size** from 5-view infrared thermography (DMR-IR dataset), via:
**U-Net segmentation → TherMAM-NeRF (3D geometry + surface temperature) → PINN inverse bioheat solver → FEA cross-check.** "PINN" is in the thesis title and must stay central.

## The key scientific finding (the heart of the thesis)
Tumour **magnitude (Q_max) is unidentifiable** from surface IR (Bezerra 2013 sensitivity analysis; confirmed — it froze at initialisation). Therefore:
- The original PINN (**v1**: 5 learnable params incl. Q_max, Adam→L-BFGS, multi-start) **fails** — magnitude freezes and a free heat source escapes to the boundary. (Documented negative result.)
- The current method (**v3, forward-matching**): PINN is a *forward* solver; magnitude is **fixed by physiology** (Gautherie doubling-time law); lateral position pinned to the IR hot-spot; only **depth × radius** are searched. Source is prescribed, so it cannot escape.

## Honest status of results (do NOT overclaim)
- **FEA residuals are 4.5–5.6 °C**, NOT < 1.5 °C. This is a model-mismatch floor; the tumour's surface signal is *smaller* than this floor.
- **Synthetic recovery (Patient 1):** the forward solver reproduces self-consistent data (J≈0.01), but the depth×radius cost surface is **nearly flat** and the recovered minimum is far from the planted values; the hot-spot also sits ~58 mm from the planted tumour (warmest skin is driven by geometry, not the tumour).
- **Open question still under test:** is the flat surface *fundamental physics* (signal too small) or an *implementation bug*? Two diagnostics implemented in `TherMAM-NeRF/run_cohort.py --diagnose` (a tumour on/off °C sensitivity probe, and a ×100-source recovery) — **not yet run.** Final wording of the validation section depends on their outcome.
- **No depth ground truth exists in DMR-IR**, so real-data depth validation is impossible for any method (Mukhmetov's real patient had unverifiable depth too). This is a citable limitation, not a personal failure.

## Defensible thesis claim
Recoverable: **lateral position + size**. Limited: **depth** (weakly constrained — a *finding*, consistent with Bezerra). Not claimed: **magnitude** (fixed by physiology, not measured).

## Draft review — top priorities (SEMHAS = results seminar, text must match results)
1. Remove `Q_max` as a result/benefit and the "Q_max benign vs malignant (Mann-Whitney U)" plan — Q_max is unidentifiable and (in v3) derived from radius, so that test is invalid and may even reverse.
2. Drop the "FEA residual < 1.5 °C = valid" criterion everywhere (it's contradicted by the 4.5–5.6 °C results) — replace with the model-mismatch-floor framing.
3. Update the method sections from v1 to the v1→v3 narrative.
4. Reconcile perfusion constants (Tabel 2.2: ρ_b=1060, c_b=3840 → ~2035 W/m³K) with the code's value (1.8).
5. Fix: convective skin BC is **Robin**, not Neumann; equation numbering; "ThermalNeRF" vs "TherMAM-NeRF" naming; decoder equation.

## Practical / environment
- GPU RTX 2080 Ti; conda env `bioheat`. Each patient inverse ≈ **1 hour** (so full 122-patient cohort ≈ 8 days = NOT feasible; use a balanced ~20–30 subset, `--subset 20 --no-fea`).
- `run_cohort.py` is a resumable tmux runner (`--synthetic`, `--diagnose`, `--subset`, `--all`).
- Deadline pressure: most of the thesis is writing/editing already-existing material, not new experiments.

## Files to also upload alongside this one
- `SEMHAS DRAFT.pdf` (the thesis draft being revised)
- `Docs/Thesis_Outline.md` (chapter plan + word targets)
- `Docs/Stage3_3DBreastNet_Reconstruction.md` and `Docs/Stage4_PINN.md` (rewritten, honest versions)
- `README.md` (pipeline overview)