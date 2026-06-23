# Stage 3: Metode dan Hasil — Rekonstruksi 2D→3D Termal (TherMAM-NeRF)

Dokumen ini menjelaskan **metode** dan **hasil** rekonstruksi model 3D payudara
berikut distribusi suhu permukaannya dari citra inframerah 2D multi-tampak.
Berkas terkait di `finalized/Stage 3/`.

---

## 1. Tujuan

Mengubah citra IR **2D dari 5 sudut pandang** menjadi **model volumetrik 3D**
payudara dengan suhu per-vertex, yang menjadi geometri masukan bagi *forward/
inverse solver* bioheat (Stage 4).

---

## 2. Metode

### 2.1 Konsep — *Thermally-Modulated Attention NeRF*

TherMAM-NeRF adalah *Neural Radiance Field* yang dimodifikasi untuk
mengeluarkan **dua medan** dari setiap titik 3D: densitas okupansi $\sigma$
(geometri) dan suhu $T$ (termal). Jaringan dilatih agar proyeksi 2D-nya
(via *volume rendering*) cocok dengan masker dan suhu IR pada kelima tampak.

### 2.2 Sudut pandang

5 tampak standar pada sudut $[-90°, -45°, 0°, +45°, +90°]$ —
RL, RO, F (Anterior), LO, LL.

### 2.3 Arsitektur

| Komponen | Spesifikasi |
|---|---|
| Encoder | Siamese (berbagi bobot), 32 kanal fitur per tampak |
| *Positional encoding* | $L=8$ frekuensi, dengan *frequency warm-up* 50 epoch |
| MLP | 4 lapis, 256 unit tersembunyi → keluaran $(\sigma, T)$ |
| Sampling sinar | 256 titik per sinar, 3072 sinar/iterasi |
| *Volume render* | `density_scale = 10`, near/far $= [-1, 1]$ |
| Ekstraksi mesh | *Marching cubes*, ambang 0.3, resolusi $128^3$ |

### 2.4 *Volume rendering*

Untuk tiap sinar, densitas diakumulasi menjadi bobot transmitansi
$w_i = \alpha_i \prod_{j<i}(1-\alpha_j)$ dengan $\alpha_i = 1 - e^{-\sigma_i \delta_i}$.
Masker = $\sum_i w_i$; suhu = $\sum_i w_i T_i / \sum_i w_i$.

> **Trik kunci.** Pada *rendering* suhu, bobot $w_i$ **di-*detach*** sehingga
> *thermal loss* hanya memperbarui medan suhu $T$ dan **tidak merusak geometri**
> $\sigma$. Ini memungkinkan pelatihan geometri + termal **bersama** dari awal
> tanpa saling mengganggu.

### 2.5 Fungsi loss gabungan

| Komponen | Bobot $\lambda$ | Fungsi |
|---|---|---|
| Dice (geometri) | 1.0 | tumpang-tindih masker render vs GT |
| Background ray | 2.0 | menekan densitas di luar siluet |
| Thermal | 20.0 | galat suhu render vs IR (hanya update $T$) |
| Total Variation 3D | 0.01 | kehalusan medan, tepi rapi |
| Entropy | 0.1 | mencegah artefak "comb" di tepi bawah |

### 2.6 *View-specific depth cropping*

Pada tampak **lateral murni (90°)**, payudara jauh secara fisik menonjol di
belakang payudara dekat, sementara masker GT hanya mencakup payudara dekat.
Maka densitas payudara seberang ($|x| > 0.1$) **disembunyikan** saat *rendering*
lateral agar tidak terpenalti — mengatasi keterbatasan tampak lateral pada
Stage 2.

---

## 3. Hasil

Evaluasi pada 122 pasien × 5 tampak = 610 tampak (425 latih / 90 validasi /
95 uji; set uji = 19 pasien *held-out*). Geometri diukur dengan Dice/IoU
(siluet render vs GT), termal dengan MAE/RMSE/akurasi-piksel.

### 3.1 Metrik agregat

| Split | n tampak | Dice | IoU | Thermal MAE | RMSE | Acc@1°C |
|---|---|---|---|---|---|---|
| Latih | 425 | 0.987 | 0.975 | 0.37 °C | 0.91 °C | 94.6 % |
| **Uji** | 95 | **0.951** | **0.912** | **0.66 °C** | 1.41 °C | **90.4 %** |

### 3.2 Interpretasi

- **Geometri sangat akurat** pada set uji (Dice 0.951, IoU 0.912) — siluet 3D
  hasil rekonstruksi nyaris identik dengan masker referensi.
- **Suhu permukaan tepat** dalam < 1 °C untuk **90.4 %** piksel (MAE 0.66 °C),
  cukup untuk analisis bioheat hilir.
- Selisih latih→uji kecil (Dice 0.987→0.951) → generalisasi baik, tanpa
  *overfitting* berarti.

**Gambar audit per-pasien** —
`thermamnerf_outputs_finalized/audit_test_Patient_*.png` menampilkan, untuk 5
tampak: masker GT, masker render (Dice/IoU), dan termal render (MAE, Acc@1°C).

### 3.3 Keluaran untuk tahap berikut

- Mesh permukaan 3D + suhu per-vertex (via *marching cubes*).
- Diekspor ke STL → mesh tetrahedral 3 mm (Gmsh) untuk solusi Pennes Stage 4.

> **Kesimpulan Stage 3.** TherMAM-NeRF berhasil merekonstruksi geometri 3D
> (Dice uji 0.951) dan suhu permukaan (< 1 °C untuk 90 % piksel) dari citra IR
> 2D — memenuhi seluruh ambang dan menjadi fondasi valid bagi inversi bioheat.

---
*Sumber: `finalized/Stage 3/thermamnerf_v3.0.py`,
`thermamnerf_outputs_finalized/thermal_pixel_accuracy_per_view.csv`,
`audit_test_Patient_*.png`, `nerf_run.log`.*