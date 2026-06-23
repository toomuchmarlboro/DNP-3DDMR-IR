# Perbandingan Metode Inverse: FEM Grid–Optimasi (Lama) vs Levenberg–Marquardt 4-DoF (Baru)

## Ringkasan

Dua pendekatan inverse diterapkan pada data termografi NeRF nyata untuk
mengestimasi geometri tumor dari distribusi suhu permukaan. Bagian ini
membandingkan keduanya secara *head-to-head* pada **30 pasien yang sama**
(15 jinak / 15 ganas), yang merupakan irisan pasien yang diproses oleh kedua
metode.

| Aspek | Metode Lama | Metode Baru |
|---|---|---|
| Strategi | Pencarian grid + optimasi biaya FEM | Levenberg–Marquardt 4-DoF (Kandlikar 2020) |
| Parameter ukuran | Jari-jari `r_hat` (clamp 40 mm) | Diameter `d` (clamp 35 mm) |
| Fungsi biaya | Residual langsung suhu permukaan | Residual **mean-subtracted** `(T−T̄)` |
| Berkas hasil | `cohort_BCfix_ALL.csv` (122 pasien) | `lm_real_subset.csv` (30 pasien) |

Hasil lengkap divisualisasikan pada **Gambar (fig6_method_compare.png)** dengan
enam panel: (a) kurva ROC, (b) AUC per fitur, (c) reduksi residual berpasangan,
(d) saturasi parameter ukuran, (e) kesesuaian estimasi ukuran, dan (f) residual
per kelas.

## Hasil Kuantitatif (n = 30)

| Metrik | Metode Lama | Metode Baru |
|---|---|---|
| AUC (skor ukuran) | 0.50 (`r_hat`) | **0.62** (`d`) |
| AUC (residual/biaya) | 0.49 (`cost`) / 0.33 (`resid`) | **0.62** (`RMS`) |
| Residual permukaan rata-rata | 5.32 °C | **2.05 °C** |
| Reduksi residual (uji-t berpasangan) | — | **p ≈ 9.9 × 10⁻²³** |
| Saturasi ukuran — jinak | 13/15 mentok 40 mm | 7/15 mentok 35 mm |
| Saturasi ukuran — ganas | 13/15 mentok 40 mm | 11/15 mentok 35 mm |
| Korelasi ukuran antar-metode | r = 0.35 | r = 0.35 |
| Kasus geometri rusak (kohort penuh) | 11/122 (residual 30–75 °C) | — |

## Pembahasan

**1. Mean-subtraction menghilangkan bias sistematis.** Model bioheat Pennes
menghasilkan suhu permukaan rata-rata ~32 °C, sedangkan citra IR NeRF
rata-rata ~29 °C — selisih sistematis ~3.4 °C. Metode lama membandingkan suhu
absolut sehingga residualnya (rata-rata 5.32 °C) didominasi *offset* ini, bukan
sinyal tumor. Dengan mengurangkan rata-rata pada kedua sisi
(`r = (T_target − T̄_target) − (T_model − T̄_model)`), metode baru memangkas
residual menjadi 2.05 °C — reduksi lebih dari setengah, sangat signifikan
secara statistik (uji-t berpasangan pada 30 pasien, p ≈ 10⁻²³; Panel c).

**2. Kemampuan diskriminasi naik dari kebetulan ke lemah.** Pada metode lama,
seluruh fitur berada di sekitar AUC 0.50 — bahkan residual FEM memberi AUC 0.33
(anti-korelasi, akibat *offset* sistematis yang mendominasi arah sinyal). Setelah
mean-subtraction, AUC naik ke 0.62 baik untuk diameter maupun RMS (Panel a, b).

**3. Parameter ukuran tetap mengalami saturasi.** Metode lama merailkan jari-jari
ke 40 mm untuk hampir semua pasien (13/15 pada kedua kelas) — radius praktis
tidak teridentifikasi. Metode baru lebih baik (jinak 7/15 mentok), tetapi
diameter masih sering jenuh di 35 mm, menunjukkan inverse cenderung mem-*fit*
gradien termal anatomis dengan satu sumber panas besar alih-alih mendeteksi
tumor (Panel d).

**4. Kedua metode tidak sepakat soal ukuran.** Korelasi estimasi diameter antar
metode hanya r = 0.35 (Panel e) — menegaskan bahwa estimasi ukuran absolut pada
data nyata belum dapat diandalkan oleh metode manapun.

## Kesimpulan

Metode Levenberg–Marquardt 4-DoF dengan biaya *mean-subtracted* merupakan
**perbaikan metodologis yang terukur** atas pendekatan grid–optimasi: residual
turun 5.32 → 2.05 °C (p ≈ 10⁻²³) dan AUC diskriminasi naik 0.50 → 0.62.
Namun AUC 0.62 masih lemah secara klinis, dan saturasi diameter serta korelasi
ukuran yang rendah menunjukkan keterbatasan mendasar: **fidelitas model
*forward* Pennes belum cukup mereproduksi pola IR nyata** sehingga inverse-nya
belum *identifiable* untuk klasifikasi malignansi. Validasi metode tetap kuat
pada data sintetis (galat posisi ~1 mm), sehingga keterbatasan ini bersumber
pada kesenjangan model–data, bukan pada algoritma inverse itu sendiri.

---
*Sumber data: `finalized/Stage 4/lm_real_subset.csv`,
`TherMAM-NeRF/results/cohort_BCfix_ALL.csv`.
Gambar: `finalized/Stage 4/figs/fig6_method_compare.png`.
Skrip: `finalized/Stage 4/make_thesis_plots.py`.*