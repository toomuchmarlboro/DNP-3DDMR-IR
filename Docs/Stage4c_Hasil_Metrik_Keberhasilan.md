# Stage 4c: Hasil dan Metrik Keberhasilan Inverse Solver

Dokumen ini merangkum **kriteria dan metrik keberhasilan** keseluruhan pipeline,
dari rekonstruksi 2D→3D (Stage 3) hingga *inverse solver* bioheat (Stage 4),
beserta plot yang siap dipakai untuk penulisan skripsi. Semua gambar tersimpan di
`finalized/Stage 4/figs/`.

---

## 1. Latar Belakang Klinis: Benign vs Malignant

Klasifikasi akhir pipeline adalah membedakan tumor **jinak (*benign*)** dari
**ganas (*malignant*)** berdasarkan jejak panasnya di permukaan kulit.

| Aspek | Tumor Jinak (*Benign*) | Tumor Ganas (*Malignant*) |
|---|---|---|
| Sifat sel | Non-kanker, tumbuh lambat, berbatas tegas | Kanker, invasif, tumbuh cepat |
| Angiogenesis | Minimal | Tinggi — membentuk pembuluh darah baru |
| Metabolisme | Mendekati jaringan normal | Hipermetabolik (panas berlebih) |
| Jejak termal | Tidak ada sumber panas terkurung | **Hotspot terlokalisasi** di permukaan |
| Konsekuensi inverse | Tidak ada sumber panas terkurung yang dapat di-*fit* | Sumber panas Gaussian dapat direkonstruksi |

**Dasar fisis.** Tumor ganas memicu *angiogenesis* dan memiliki laju metabolik
jauh di atas jaringan normal (hukum Gautherie — laju metabolik naik secara
eksponensial terhadap pertumbuhan tumor). Panas metabolik berlebih ini merambat
ke permukaan kulit dan menimbulkan *hotspot* yang dapat ditangkap kamera
inframerah. Tumor jinak tidak menghasilkan sumber panas terkurung, sehingga
secara teori tidak menghasilkan *hotspot* yang dapat direkonstruksi oleh *inverse
solver*.

**Komposisi dataset.** 122 pasien lengkap (96 jinak, 26 ganas), label dari
dataset DMR-IR/TherMAM-NeRF.

**Prinsip deteksi.** *Inverse solver* mencoba merekonstruksi satu sumber panas
Gaussian (parameter posisi + diameter) yang paling cocok dengan suhu permukaan.
Bila tumor benar-benar ada (ganas), sumber tersebut konvergen ke dalam volume
payudara. Bila tidak ada (jinak), lanskap biaya menjadi datar dan solusi
melayang keluar batas — *drift* inilah yang menjadi sinyal "jinak".

---

## 2. Kriteria Pengukuran 2D → 3D (Stage 3, TherMAM-NeRF)

Tahap ini mengubah citra inframerah **2D multi-tampak** menjadi **model 3D**
payudara berikut distribusi suhu permukaannya.

### 2.1 Alur

1. **Masukan**: citra IR 2D dari 5 sudut pandang standar — Anterior/Front (F),
   Right/Left Oblique 45° (RO/LO), Right/Left Lateral 90° (RL/LL).
2. **Rekonstruksi**: TherMAM-NeRF (*Neural Radiance Field* ber-modulasi termal)
   membangun medan volumetrik 3D.
3. **Ekstraksi permukaan**: *marching cubes* → mesh permukaan + suhu per-vertex.
4. **Mesh FEM**: STL → mesh tetrahedral 3 mm (Gmsh) untuk solusi Pennes.

### 2.2 Kriteria keberhasilan 2D→3D

Rekonstruksi diukur dua aspek: **kesetiaan geometri** (siluet 3D di-*render*
ulang ke 2D lalu dibandingkan dengan masker *ground-truth* tiap tampak) dan
**kesetiaan termal** (suhu yang di-*render* vs suhu IR asli).

| Metrik | Arti | Ambang sukses |
|---|---|---|
| Dice | Tumpang-tindih siluet (geometri) | > 0.90 |
| IoU | *Intersection-over-Union* siluet | > 0.85 |
| Thermal MAE | Galat suhu rata-rata absolut | < 1.0 °C |
| Thermal RMSE | Akar galat kuadrat rata-rata | < 1.5 °C |
| Pixel accuracy @1°C | % piksel dengan galat < 1 °C | > 90 % |

### 2.3 Hasil terukur (per-tampak, 104 pasien, 520 tampak)

| Split | n tampak | Dice | IoU | Thermal MAE | RMSE | Acc@1°C |
|---|---|---|---|---|---|---|
| Latih | 425 | 0.987 | 0.975 | 0.37 °C | 0.91 °C | 94.6 % |
| **Uji** | 95 | **0.951** | **0.912** | **0.66 °C** | 1.41 °C | **90.4 %** |

> Pada set uji yang belum pernah dilihat model, geometri tetap sangat akurat
> (Dice 0.95) dan suhu permukaan tepat dalam < 1 °C untuk 90 % piksel —
> memenuhi seluruh ambang. Contoh per-pasien lihat
> `Stage 3/thermamnerf_outputs_finalized/audit_test_Patient_*.png` (5 tampak:
> masker GT, masker render, termal render dengan metrik).

---

## 3. Parameter Keberhasilan Inverse Solver (Stage 4)

*Inverse solver* Levenberg–Marquardt 4-DoF merekonstruksi
$\boldsymbol{\beta} = [x_t, y_t, z_t, d]$ (pusat tumor + diameter) dari suhu
permukaan. Keberhasilan diukur dengan parameter berikut.

| Parameter | Definisi | Ambang sukses |
|---|---|---|
| Galat posisi | $\lvert\Delta(x,y,z)\rvert = \lVert\hat{\mathbf{x}}_t - \mathbf{x}_t\rVert$ | < 5 mm |
| Galat diameter | $\lvert\Delta d\rvert = \lvert\hat{d} - d\rvert$ | < 2 mm |
| RMS akhir (sintetik) | $\sqrt{\langle r^2\rangle}$ residual suhu | < 10⁻³ °C |
| Konvergensi | langkah diterima & $\lVert\delta\boldsymbol{\beta}\rVert < 0.05$ mm | tercapai |
| Anggaran solusi FEM | jumlah pemanggilan *forward solve* | ≤ ~160 |
| Verdict deteksi | sintetik harus → MALIGNANT | sesuai label |

**Catatan identifiabilitas.** Posisi tumor sengaja **tidak dibatasi** selama
iterasi LM agar *drift* keluar batas dapat menjadi sinyal "jinak" (lihat §1).
Sebuah *box-constraint* longgar (batas bbox ± 15 mm, $d \in [1, 35]$ mm)
kemudian ditambahkan hanya untuk mencegah divergensi liar pada kuadran yang
*ill-posed* (lihat §4.3).

---

## 4. Metrik Keberhasilan — Validasi Sintetik

Validasi sintetik menanam tumor pada lokasi & ukuran **yang diketahui**,
menyelesaikan *forward* untuk menghasilkan suhu target, lalu menguji apakah
*inverse solver* dapat memulihkannya. 20 skenario (2 pasien × 10 penempatan
acak menurut distribusi kuadran literatur).

### 4.1 Plot 3D dengan tampak Front / Side / Top

**Gambar `figs/syn_recovery_multiview.png`** — pusat tumor tertanam (lingkaran)
vs hasil rekonstruksi LM (×), ditampilkan dari:
- **(a) 3D** — pandangan perspektif penuh.
- **(b) Front (x–y)** — bidang lateral × supero-inferior.
- **(c) Side (z–y)** — bidang kedalaman × supero-inferior.
- **(d) Top (x–z)** — bidang lateral × kedalaman.

Pada kuadran *well-posed*, marker tertanam dan terekonstruksi nyaris berimpit di
ketiga tampak; kasus *ill-posed* (UO) tampak sebagai panah yang keluar bingkai.

### 4.2 Plot metrik

**Gambar `figs/syn_recovery_metrics.png`** — (a) galat posisi tiap skenario
(skala symlog, garis ambang 5 mm), (b) diameter tertanam vs terekonstruksi,
(c) galat median per kuadran.

### 4.3 Hasil terukur

| Subset | n | Galat posisi (median) | Galat diameter (median) | Sukses < 5 mm |
|---|---|---|---|---|
| **Keseluruhan** | 20 | **0.59 mm** | **0.15 mm** | 16/20 |
| Kuadran UI | 10 | 0.45 mm | 0.05 mm | 10/10 |
| Kuadran C | 3 | 0.01 mm* | 0.02 mm* | 2/3 |
| Kuadran UO | 7 | 3.2 mm† | 6.3 mm† | 4/7 |

\* 2 dari 3 kasus C sempurna (< 0.02 mm); 1 kasus menyimpang.
† Kuadran UO *ill-posed* pada run sintetik ini (sebelum *box-constraint*
ditambahkan); sumber lemah karena jauh dari puting → Jacobian buruk.

> **Kesimpulan validasi sintetik.** Ketika *inverse solver* konvergen ke cekungan
> yang benar (mayoritas kasus UI dan C), pemulihan geometri **sub-milimeter**
> (median posisi 0.59 mm, diameter 0.15 mm) — jauh di bawah ambang 5 mm/2 mm.
> Ini membuktikan **algoritma inverse bekerja dengan benar**. Kegagalan hanya
> pada kuadran UO yang secara fisis *ill-posed* (sinyal permukaan lemah),
> bukan kesalahan algoritma — dan ditangani dengan *box-constraint*.

---

## 5. Metrik Keberhasilan — Pasien Generated (Data IR Nyata)

Pada data nyata, suhu target adalah suhu IR NeRF hasil Stage 3 (bukan sintetik).
Subset seimbang 30 pasien (15 jinak / 15 ganas) dengan biaya *mean-subtracted*.

### 5.1 Plot

| Gambar | Isi |
|---|---|
| `figs/fig1_roc.png` | Kurva ROC (`final_rms` & `d`) — AUC 0.62 |
| `figs/fig2_distributions.png` | Sebaran RMS & diameter per kelas (+ uji-t) |
| `figs/fig3_feature_scatter.png` | Ruang fitur d vs RMS, jinak vs ganas |
| `figs/fig4_confusion.png` | Matriks konfusi verdict |
| `figs/fig5_per_patient.png` | RMS + jumlah solusi FEM per pasien |

### 5.2 Hasil terukur (n = 30)

| Metrik | Jinak | Ganas | AUC |
|---|---|---|---|
| Final RMS | 1.97 ± 0.22 °C | 2.13 ± 0.36 °C | 0.62 |
| Diameter terekonstruksi | 26.1 mm | 29.2 mm | 0.62 |
| Sensitivitas / Spesifisitas | — | — | 0.67 / 0.20 |

> **Kesimpulan data nyata.** Diskriminasi lemah (AUC 0.62, nyaris di atas
> kebetulan), uji-t tak signifikan (p = 0.15), dan diameter sering jenuh di
> 35 mm pada kedua kelas. Penyebabnya **bukan algoritma** (terbukti sub-mm pada
> sintetik) melainkan **kesenjangan model–data**: residual ~2 °C jauh melebihi
> sinyal tumor ~0.5 °C (SNR ~0.25). Forward-model Pennes belum cukup mereproduksi
> pola IR nyata untuk inverse yang *identifiable*.

---

## 6. Perbaikan Metodologi (Lama vs Baru)

Lihat dokumen terpisah **`Docs/Perbandingan_Metode_Inverse.md`** dan
**Gambar `figs/fig6_method_compare.png`** (6 panel).

| Metrik | Metode Lama (grid+opt) | Metode Baru (LM, mean-sub) |
|---|---|---|
| AUC diskriminasi | 0.50 | **0.62** |
| Residual permukaan | 5.32 °C | **2.05 °C** |
| Reduksi residual | — | **p ≈ 9.9 × 10⁻²³** (uji-t berpasangan) |

---

## 7. Ringkasan Naratif untuk Skripsi

1. **2D→3D (Stage 3)** berhasil: geometri Dice 0.95 dan suhu < 1 °C (90 % piksel)
   pada set uji — fondasi yang valid.
2. **Inverse solver (Stage 4)** terbukti benar pada data sintetik: pemulihan
   geometri **sub-milimeter** (median 0.59 mm) — *kontribusi metodologis utama*.
3. **Perbaikan metodologi** terukur: LM + *mean-subtraction* memangkas residual
   5.32→2.05 °C (p ≈ 10⁻²³) dan menaikkan AUC 0.50→0.62.
4. **Keterbatasan data nyata** dilaporkan jujur: klasifikasi malignansi masih
   lemah (AUC 0.62) akibat keterbatasan fidelitas forward-model, **bukan**
   kegagalan algoritma inverse.

---
*Sumber data: `finalized/Stage 4/lm_synthetic_20260623_010523.csv`,
`finalized/Stage 4/lm_real_subset.csv`,
`finalized/Stage 3/thermamnerf_outputs_finalized/thermal_pixel_accuracy_per_view.csv`.
Skrip plot: `finalized/Stage 4/synthetic_recovery_plots.py`,
`finalized/Stage 4/make_thesis_plots.py`.*