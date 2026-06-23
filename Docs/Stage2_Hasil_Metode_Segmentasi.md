# Stage 2: Metode dan Hasil — Segmentasi Region Payudara (Attention U-Net)

Dokumen ini menjelaskan **metode** dan **hasil** segmentasi otomatis region
payudara dari citra termal inframerah, beserta plot siap-pakai untuk skripsi.
Berkas terkait di `finalized/Stage 2/`.

---

## 1. Tujuan

Melatih model **segmentasi semantik** untuk memisahkan region payudara dari
citra termal satu-kanal. Masker biner keluaran dipakai untuk dua hal di tahap
berikutnya:

1. Mendefinisikan **kontur siluet** yang menjadi masukan rekonstruksi 3D
   (Stage 3).
2. Mengisolasi **region of interest (ROI)** untuk analisis termal dan inversi
   bioheat (Stage 4), menghilangkan derau latar belakang.

---

## 2. Metode

### 2.1 Praproses data

Setiap akuisisi tersimpan sebagai **TIFF 16-bit** berisi nilai suhu per-piksel
$P_{i,j}$. Sebelum masuk jaringan:

1. **Resize** ke resolusi kanonik $256\times256$ piksel (`cv2.INTER_AREA`,
   menjaga densitas energi termal).
2. **Normalisasi min–max** ke rentang $[0,1]$:
   $$ m_{i,j} = \frac{P_{i,j} - \min(P)}{\max(P) - \min(P) + \varepsilon} $$

### 2.2 Anotasi *ground-truth*

Masker biner dibuat manual melalui anotasi poligon interaktif dengan prior
anatomis ketat: batas **superior** mengecualikan leher/torso atas, batas
**lateral** dipotong di lipatan lateral (cegah kebocoran aksila), batas
**inferior** mengikuti lipatan inframammary (IMF). Poligon dirasterisasi
(`cv2.fillPoly`) menjadi masker biner $256\times256$.

### 2.3 Arsitektur — Attention U-Net

Encoder–decoder berbentuk-U dengan **attention gate** pada *skip connection*,
yang menekan respons di region tak-relevan (latar) dan mempertajam batas
payudara. Keluaran satu kanal dilewatkan sigmoid → peta probabilitas masker.

### 2.4 Fungsi loss

**Dice loss** (langsung mengoptimalkan tumpang-tindih, tahan terhadap
ketidakseimbangan kelas latar/objek):
$$ \mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum p_i g_i + \epsilon}{\sum p_i + \sum g_i + \epsilon} $$
dengan $p_i$ prediksi dan $g_i$ *ground-truth*.

### 2.5 Protokol pelatihan dan evaluasi

| Aspek | Nilai |
|---|---|
| Total citra berlabel | 162 |
| Split | 78 % latih (126) / 22 % uji (36), `SEED=42` |
| Validasi silang | **5-fold** pada pool latih |
| *Early stopping* | berdasarkan val Dice |
| Set uji | *held-out*, hanya untuk evaluasi akhir tanpa bias |

Set uji **tidak pernah** dipakai saat seleksi model, penyetelan, atau
*early stopping*.

---

## 3. Hasil

### 3.1 Validasi silang 5-fold

| Fold | Epoch terbaik | Val Dice terbaik |
|---|---|---|
| 1 | 51 | 0.864 |
| 2 | 52 | 0.885 |
| 3 | 78 | 0.911 |
| 4 | 55 | 0.891 |
| 5 | 53 | 0.923 |
| **Rerata** | — | **0.895 ± 0.021** |

**Gambar `unet_5fold_training_curves.png`** — kurva *train loss* dan *val Dice*
tiap fold; konvergensi stabil, *early stopping* mencegah *overfitting*.

### 3.2 Evaluasi set uji (Tabel 4.4)

Dice per **sudut pandang** pada 36 citra uji *held-out*:

| Sudut Pandang | n | Dice (mean ± std) |
|---|---|---|
| Anterior (Front) | 7 | 0.923 ± 0.037 |
| Right Oblique (45°) | 8 | 0.939 ± 0.022 |
| Left Oblique (45°) | 7 | 0.921 ± 0.022 |
| Right Lateral (90°) | 8 | 0.868 ± 0.083 |
| Left Lateral (90°) | 6 | 0.878 ± 0.084 |
| **Keseluruhan** | **36** | **0.906** (rata-rata terbobot) |

**Gambar `unet_test_metrics_and_table44.png`** — metrik uji + Tabel 4.4.

### 3.3 Interpretasi

- **Dice uji 0.906** menunjukkan segmentasi yang andal pada data tak-terlihat,
  konsisten dengan val 5-fold (0.895) — tidak ada *overfitting* berarti.
- Tampak **oblique (45°)** dan **frontal** paling akurat (Dice > 0.92) karena
  siluet payudara paling jelas.
- Tampak **lateral (90°)** sedikit lebih rendah & ber-variansi besar
  (Dice ~0.87, std ~0.08): pada sudut samping kedua payudara saling tumpang
  tindih dan batas posterior ambigu — keterbatasan yang ditangani secara
  geometris di Stage 3 (*view-specific depth cropping*).

> **Kesimpulan Stage 2.** Attention U-Net menghasilkan masker region payudara
> berkualitas tinggi (Dice uji 0.906) yang menjadi fondasi geometris andal bagi
> rekonstruksi 3D — memenuhi ambang Dice > 0.90.

---
*Sumber: `finalized/Stage 2/2_unetsegmentation_fixed.ipynb`,
`unet_training_history/fold_*_history.json`,
`Tabel_4_4_Dice_Score_UNet_Set_Uji.csv`.
Gambar: `unet_5fold_training_curves.png`, `unet_test_metrics_and_table44.png`.*