# Undergraduate Thesis — Plan & Outline

**You are not behind. You have a complete pipeline. This document is the plan; just follow it top to bottom.**

Target length for an undergrad thesis: ~8,000–15,000 words. That is finite and achievable. Most of it you can write **now**, before any run finishes.

---

## The one-line story of your thesis

> A non-invasive screening pipeline that reconstructs a 3D breast model **and** its surface temperature from multi-view infrared images (U-Net → TherMAM-NeRF), then uses a **physics-informed neural network (PINN)** to solve the inverse Pennes bioheat problem and estimate a tumour's **position, depth, and size** — with the method's limits established honestly on synthetic ground truth.

That is a real, defensible contribution. Keep saying it to yourself.

---

## The narrative arc (this is your strength, not a weakness)

1. **v1 PINN** — naive inverse, learns all 5 parameters including heat magnitude → overconfident, magnitude unidentifiable (a documented **negative result**).
2. **v3 PINN** — forward-matching inverse: magnitude fixed by physiology (Gautherie), lateral position pinned to the IR hot-spot, search only over depth × radius. The source can't escape.
3. **Synthetic validation** — plant known tumours, check what the method can actually recover. This is what makes the thesis rigorous.

Examiners reward this kind of honest methodological progression.

---

## Chapter outline (fill the blanks — most needs no new results)

### 1. Introduction (~1,000–1,500 words) — *writeable now*
- Breast cancer screening; why a non-invasive, low-cost thermography approach matters.
- Problem: surface IR alone is 2D; we want deep-tissue tumour parameters.
- Objective: estimate tumour **position, depth, size** from multi-view IR via 3D reconstruction + PINN inverse bioheat.
- Contributions (bullet list). Thesis structure.

### 2. Background & Related Work (~1,500–2,500 words) — *writeable now*
- Breast thermography & the DMR-IR dataset.
- Pennes bioheat equation (write the equation, explain each term).
- Physics-Informed Neural Networks (what they are, why use one here).
- NeRF / neural fields (one paragraph, conceptual).
- Prior inverse bioheat work: **Mukhmetov 2025/2023** (PINN forward + their inverse), **Bezerra 2013** (sensitivity analysis: magnitude/perfusion unidentifiable). This literature *predicts* your depth limitation — use it.

### 3. Methodology (~2,500–3,500 words) — *writeable now*
- **3.1 Data** — DMR-IR, 5-view protocol (RL/RO/F/LO/LL), FLIR SC620 geometry, 122 complete patients.
- **3.2 Preprocessing + U-Net segmentation** — architecture, hybrid BCE+Dice loss, **Test Dice 0.8935 ± 0.0542**. (Stage2 doc already written — reuse it.)
- **3.3 TherMAM-NeRF reconstruction** — Siamese encoder + MLP → (density, temperature); marching cubes → watertight mesh; surface-temperature sampling. Note: only **surface** temperature is trusted (interior is unconstrained extrapolation).
- **3.4 PINN inverse bioheat** — the v1 → v3 story above; the forward BVP (chest-wall Dirichlet 37 °C + convective Robin skin + interior Pennes); physiology-fixed magnitude; depth × radius search.
- **3.5 FEA forward verification** — FEniCSx/Gmsh independent cross-check.
- **3.6 Synthetic recovery validation** — how the gate works.

### 4. Results (~1,500–2,500 words) — *needs the two runs below*
- **4.1** Segmentation examples + Dice.
- **4.2** Reconstruction examples (3D thermal meshes — you already have HTML viewers; screenshot them).
- **4.3** Synthetic recovery (planted vs recovered depth/radius) — **the key validation figure**.
- **4.4** Cohort estimates: tumour **position** + **size** (+ depth with the validated caveat). Quadrant distribution, radius by benign/malignant.

### 5. Discussion (~1,500 words) — *writeable after results*
- What works: **lateral position and size** are recoverable from the hot-spot and footprint.
- The **depth identifiability** finding — tie directly to Bezerra's sensitivity result. This is a *finding*, not a failure.
- Error sources: NeRF interior not observable, bilateral-frame scale calibration, FEA ~4–6 °C model-mismatch floor.

### 6. Conclusion & Future Work (~500–800 words) — *writeable last*
- Recap contribution. Future: better priors on depth, more BCs, clinical ground truth.

---

## What still needs to run (decided — do NOT run all 122)

1. **Synthetic recovery** (~3 h, the validation gate): `run_cohort.py --synthetic`
2. **A 20-patient subset, no FEA** for the cohort table (~1 day): `run_cohort.py --subset 20 --no-fea`

That's it. Full cohort = ~8 days = not happening, and not needed for an undergrad thesis.

---

## Order of work (do this, in this order)

1. **Tonight:** start the synthetic run, close the laptop, sleep.
2. **Tomorrow AM:** look at `synthetic_recovery.png`. Either outcome gives you a thesis (see Discussion framing). Start the `--subset 20 --no-fea` run.
3. **While that runs:** write Chapters 1, 2, 3 (they need no results).
4. **When runs finish:** drop figures into Chapter 4, write 5 and 6.
5. Polish, references, abstract.

You have enough for every one of these steps. Follow the list.
