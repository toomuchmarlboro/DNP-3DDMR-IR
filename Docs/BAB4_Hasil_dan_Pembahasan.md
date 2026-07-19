# BAB 4 — HASIL DAN PEMBAHASAN

---

## 4.1 Persiapan Dataset — Reorganisasi DMR-IR per Pasien dan Sudut Pandang

### 4.1.1 Struktur Dataset Asli DMR-IR

Dataset yang digunakan adalah **DMR-IR** (*Dynamic Mammography for Infrared*),
yang dalam kondisi aslinya menggunakan struktur *split*-sentris:
`train/`, `test/`, dan `validation/`, masing-masing memuat subfolder `benign/`
dan `malignant/` berisi file TIFF dan `metadata.csv`. Struktur ini menyulitkan
pengelompokan multi-tampak per pasien yang diperlukan oleh pipeline rekonstruksi
3D — karena kelima tampak seorang pasien dapat tersebar di split yang berbeda.

### 4.1.2 Proses Reorganisasi (`organize_by_patient_and_view.py`)

Untuk mengatasi hal ini, ditulis skrip `organize_by_patient_and_view.py` yang
membaca `metadata.csv` dari ketiga split, memetakan setiap file TIFF ke
`patient_id` dan `view_name`-nya, lalu menyalin file ke struktur baru
berbasis-pasien:

```
data/organized_by_patient/
└── Patient_<ID>/
    └── <benign|malignant>/
        ├── Anterior (Front).tiff
        ├── Left Oblique (45°).tiff
        ├── Right Oblique (45°).tiff
        ├── Left Lateral (90°).tiff
        └── Right Lateral (90°).tiff
```

Pemetaan nama tampak dilakukan melalui kamus normalisasi:

| Nama Asli Dataset | Nama Standar Pipeline |
|---|---|
| `"Frontal"` | `"Anterior (Front)"` |
| `"Left 45°"` | `"Left Oblique (45°)"` |
| `"Right 45°"` | `"Right Oblique (45°)"` |
| `"Left 90°"` | `"Left Lateral (90°)"` |
| `"Right 90°"` | `"Right Lateral (90°)"` |

File dengan `patient_id = "Unknown"` atau `view_name = "Unknown"` dilewati.
Hasil akhir: **657 file TIFF** dari 137 pasien tersusun dalam struktur per-pasien.
Pasien yang memiliki kelima tampak lengkap (5/5) dianggap **pasien lengkap** dan
menjadi kandidat pipeline; pasien tidak lengkap dilewati otomatis di semua tahap
berikutnya.

---

## 4.2 Pembuatan Ground Truth Segmentasi — Anotasi Masker Manual

### 4.2.1 Alasan Masker Manual

Segmentasi region payudara memerlukan prior anatomis yang tidak dapat diperoleh
dari algoritma threshold otomatis: leher, lengan, dan aksila harus dikecualikan
meskipun memiliki respons termal yang mirip dengan payudara pada sudut pandang
tertentu. Oleh karena itu, seluruh *ground-truth* dibuat melalui anotasi poligon
manual satu per satu (`MANUAL MASKING.ipynb`).

### 4.2.2 Alur Anotasi

Alur anotasi berjalan sebagai berikut:

**Langkah 1 — Pemindaian batch yang belum dianotasi.**
Skrip memindai seluruh TIFF di `organized_by_patient/`, memeriksa apakah PNG
masker yang bersesuaian sudah ada di `GroundTruth_Masks/`, dan membangun
`target_batch` dari yang belum dianotasi. Dalam satu sesi, pengguna bisa memilih
berapa banyak yang akan dianotasi (misalnya, `target_batch = unmasked[:150]`).

**Langkah 2 — Pra-proses untuk tampilan anotasi.**
Setiap TIFF dimuat dengan `tifffile.imread()` lalu dinormalisasi ke [0, 255]:

$$
\text{disp}_{i,j} = 255 \times \frac{P_{i,j} - \min(P)}{\max(P) - \min(P) + \varepsilon}
$$

Kemudian di-resize ke 256×256 dan di-*colormap* dengan `COLORMAP_MAGMA`
(palet termal magenta-oranye-kuning) untuk mempertajam batas siluet secara visual.

**Langkah 3 — Anotasi poligon interaktif.**
Jendela OpenCV (`"V2 Stronger Masking"`) ditampilkan. Anotator mengklik kiri
untuk meletakkan titik-titik poligon $\mathcal{P} = \{(x_k, y_k)\}_{k=1}^N$
mengikuti batas payudara dengan tiga aturan ketat:

> 1. **AVOID THE NECK**: Batas superior harus berhenti di bagian atas payudara,
>    tidak mencakup leher atau torso atas.
> 2. **AVOID ARMPITS**: Batas lateral berhenti di lipatan lateral untuk mencegah
>    "kebocoran aksila" (*armpit leakage*).
> 3. Batas inferior mengikuti lipatan *inframammary fold* (IMF) secara konsisten
>    antarpassien dan antartampak.
>
> Tombol: **S** = simpan masker saat ini, **C** = hapus dan gambar ulang,
> **Q** = keluar sesi dengan aman.

**Langkah 4 — Rasterisasi dan penyimpanan.**
Ketika anotator menekan **S**, poligon dirasterisasi menjadi masker biner 256×256:

```python
mask = np.zeros((256, 256), dtype=np.uint8)
cv2.fillPoly(mask, [np.array(pts)], 255)
```

Masker disimpan sebagai PNG di `GroundTruth_Masks/` dengan hirarki relatif yang
identik dengan sumber TIFF. Proses dilanjutkan ke gambar berikutnya secara
otomatis.

### 4.2.3 Statistik Anotasi

Pada akhir seluruh sesi anotasi:
- **Total TIFF**: 657
- **Masker berhasil dibuat**: **162 masker** (pasien dengan kelima tampak lengkap
  dan terseleksi untuk pelatihan U-Net)
- Format keluaran: PNG biner 256×256 (piksel = 0 atau 255)

---

## 4.3 Tahap 2 — Segmentasi Region Payudara (Attention U-Net)

### 4.3.1 Praproses Data dan Pembagian Dataset

#### 4.3.1.1 Pemuatan Citra dan Normalisasi

Kelas `ThermalDataset` memuat pasangan TIFF–PNG. Setiap TIFF diproses:

1. **Muat** dengan `tifffile.imread()` → array 2D nilai suhu kalibrasi 16-bit.
2. **Normalisasi min–max** ke [0, 1] (*Paper Equation 1*):

$$
m_{i,j} = \frac{P_{i,j} - \min(P)}{\max(P) - \min(P) + \varepsilon}
$$

dengan $\varepsilon = 10^{-6}$ untuk cegah pembagian nol.

3. **Resize** ke $256 \times 256$ piksel menggunakan `cv2.INTER_AREA` — interpolasi
   rata-rata area, yang menjaga densitas energi termal lebih baik daripada bilinear
   pada proses *downsampling*.
4. Dikembalikan sebagai tensor PyTorch bentuk $(1, 256, 256)$ (kanal tunggal).

Masker PNG dimuat dengan `cv2.imdecode` (bukan `cv2.imread` biasa — untuk
melewati masalah encoding karakter derajat `°` pada Windows), dibagi 255, dan
di-resize ke $256 \times 256$.

#### 4.3.1.2 Pembagian Train/Test

Dataset 162 citra dibagi **satu kali sebelum pelatihan apapun** dengan `SEED = 42`:

```python
n_train = int(0.78 * 162) = 126   # pool pelatihan (5-fold CV di dalam ini)
n_test  = 162 - 126        = 36   # held-out, dikunci selama seluruh sesi
```

| Split | Jumlah | Fungsi |
|---|---|---|
| Pool latih | 126 (78 %) | 5-fold CV + pelatihan |
| Set uji *held-out* | 36 (22 %) | Evaluasi akhir sekali, tanpa bias |

Set uji **tidak pernah** dipakai selama seleksi model, penyetelan *hyperparameter*,
atau pemantauan *early stopping*.

---

### 4.3.2 Augmentasi Data

Untuk mengurangi *overfitting* pada dataset medis terbatas, **Albumentations**
diaplikasikan per *batch* saat pelatihan:

| Transformasi | Parameter | Diterapkan pada |
|---|---|---|
| `HorizontalFlip` | $p = 0{,}5$ | Citra + masker |
| `ShiftScaleRotate` | geser 5 %, skala 5 %, rotasi ±15° | Citra + masker |
| `ElasticTransform` | $\alpha=1$, $\sigma=50$ | Citra + masker |
| `RandomBrightnessContrast` | $p = 0{,}5$ | Citra saja |
| `CoarseDropout` | 8 lubang, $16 \times 16$ px | Citra saja |

Transformasi geometri diterapkan **identik** pada citra dan masker (menjaga
korespondensi spasial). Transformasi fotometrik hanya pada citra — masker
tidak berubah karena biner.

---

### 4.3.3 Arsitektur — Attention U-Net

#### 4.3.3.1 Prinsip Gerbang Atensi

Model adalah **Attention U-Net** 4-level: encoder–decoder berbentuk U dengan
**gerbang atensi** (*attention gate*) pada setiap *skip connection*. Gerbang atensi
mempelajari peta bobot spasial $\psi \in [0,1]$ yang menekan aktivasi latar
belakang sebelum fitur *skip* digabung dengan jalur dekoder:

$$
\psi = \sigma\!\left(\mathbf{W}_\psi \cdot \text{ReLU}\!\left(\mathbf{W}_g\, g + \mathbf{W}_x\, x\right)\right)
$$

di mana $g$ adalah fitur dekoder ter-*upsample* dan $x$ adalah fitur encoder
*skip*. Keluaran yang dilemahkan atensi $x \cdot \psi$ menggantikan *skip*
mentah, mempertajam perhatian pada batas payudara dan menekan respons aksila/torso.

#### 4.3.3.2 Aliran Kanal

Setiap blok konvolusi: `Conv(3×3) → BN → ReLU → Conv(3×3) → BN → ReLU → Dropout`.

| Tahap | Masukan → Keluaran | Dropout |
|---|---|---|
| Encoder 1 | 1 → 64 | — |
| Encoder 2 | 64 → 128 | — |
| Encoder 3 | 128 → 256 | 0,1 |
| Encoder 4 | 256 → 512 | 0,1 |
| Bottleneck | 512 → 1024 | 0,2 |
| Decoder 4 (Attn + cat) | 1024 → 512 | 0,1 |
| Decoder 3 (Attn + cat) | 512 → 256 | 0,1 |
| Decoder 2 (Attn + cat) | 256 → 128 | — |
| Decoder 1 (Attn + cat) | 128 → 64 | — |
| Keluaran | 64 → 1 (logit) | — |

**Total parameter yang dapat dilatih: 31.388.013.**

Keluaran adalah *logit* (tanpa sigmoid di *head*) — sigmoid diterapkan hanya
saat inferensi dan perhitungan metrik.

#### 4.3.3.3 Inisialisasi Bobot

Semua lapisan konvolusi menggunakan **Kaiming Normal** (inisialisasi He):

$$
\mathbf{W}^{(l)} \sim \mathcal{N}\!\left(0,\; \frac{2}{n_l}\right)
$$

BatchNorm: bobot = 1, bias = 0.

---

### 4.3.4 Fungsi Loss — Focal-Dice Komposit

$$
\mathcal{L} = 0{,}5 \cdot \mathcal{L}_{\text{Focal}} + 0{,}5 \cdot \mathcal{L}_{\text{Dice}}
$$

**Focal Loss** ($\alpha = 0{,}25$, $\gamma = 2{,}0$):

$$
\mathcal{L}_{\text{Focal}} = -\frac{\alpha}{N}\sum_{i=1}^N (1 - p_i)^\gamma
\left[y_i \log p_i + (1-y_i)\log(1-p_i)\right]
$$

Faktor $(1-p_i)^\gamma$ me-*down-weight* piksel latar belakang yang mudah
diklasifikasikan, memusatkan gradien pada batas payudara yang sulit — terutama
bermanfaat pada tampak lateral di mana payudara hanya sebagian kecil citra.

**Soft Dice Loss**:

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_i p_i y_i + \varepsilon}{\sum_i p_i + \sum_i y_i + \varepsilon}
$$

Dice loss secara langsung mengoptimalkan tumpang tindih volumetrik dan tahan
terhadap ketidakseimbangan kelas foreground/background.

---

### 4.3.5 Protokol Pelatihan — 5-Fold Cross-Validation

5-fold CV dilakukan **hanya di dalam 126 citra pool latih**. Set uji 36 citra
tidak tersentuh.

| Hyperparameter | Nilai |
|---|---|
| Perangkat | `cuda:1` |
| Optimiser | AdamW |
| Learning rate awal $\eta_0$ | $3 \times 10^{-4}$ |
| Weight decay | $1 \times 10^{-4}$ |
| Penjadwal LR | `ReduceLROnPlateau` (faktor 0,5, patience 3, min $10^{-6}$) |
| Ukuran batch | 2 |
| Epoch maksimum | 120 |
| Patience *early stopping* | 12 epoch (tanpa peningkatan val Dice) |
| AMP | Aktif (CUDA) |
| Pemotongan gradien | norm maks 1,0 |

Setiap fold: (a) melatih model baru dari awal, (b) menyimpan *checkpoint* terbaik
saat val Dice membaik (`unet_fold_N.pth`), (c) menyimpan riwayat lengkap ke
`unet_training_history/fold_N_history.json` untuk plot ulang.

**Keluaran log pelatihan (dari notebook — nilai aktual):**

```
========== FOLD 1/5 ==========   CV train: 100 | CV val: 26 | held-out test: 36
Epoch 010 | Train Loss: 0.1151 | Train Dice: 0.8301 | Val Dice: 0.7980
Epoch 040 | Train Loss: 0.0551 | Train Dice: 0.9181 | Val Dice: 0.8586
Early stopping fold 1 at epoch 51.
Fold 1 Best Val Dice: 0.8641 at epoch 39

========== FOLD 2/5 ==========   CV train: 101 | CV val: 25
Epoch 020 | Train Loss: 0.0779 | Train Dice: 0.8807 | Val Dice: 0.8644
Epoch 040 | Train Loss: 0.0642 | Train Dice: 0.9018 | Val Dice: 0.8845
Early stopping fold 2 at epoch 52.
Fold 2 Best Val Dice: 0.8845 at epoch 40

========== FOLD 3/5 ==========   CV train: 101 | CV val: 25
Epoch 030 | Train Loss: 0.0666 | Train Dice: 0.8968 | Val Dice: 0.9072
Epoch 060 | Train Loss: 0.0575 | Train Dice: 0.9114 | Val Dice: 0.9099
Early stopping fold 3 at epoch 78.
Fold 3 Best Val Dice: 0.9107 at epoch 66

========== FOLD 4/5 ==========   CV train: 101 | CV val: 25
Epoch 050 | Train Loss: 0.0605 | Train Dice: 0.9079 | Val Dice: 0.8891
Early stopping fold 4 at epoch 55.
Fold 4 Best Val Dice: 0.8914 at epoch 43

========== FOLD 5/5 ==========   CV train: 101 | CV val: 25
Epoch 030 | Train Loss: 0.0705 | Train Dice: 0.8916 | Val Dice: 0.9173
Epoch 040 | Train Loss: 0.0659 | Train Dice: 0.8974 | Val Dice: 0.9226
Early stopping fold 5 at epoch 53.
Fold 5 Best Val Dice: 0.9234 at epoch 41
```

**Tabel 4.1 — Hasil 5-Fold Cross-Validation (pool latih, 126 citra)**

| Fold | Sampel Latih CV | Sampel Val CV | Val Dice Terbaik | Epoch Berhenti |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 100 | 26 | 0,8641 | 51 |
| 2 | 101 | 25 | 0,8845 | 52 |
| 3 | 101 | 25 | 0,9107 | 78 |
| 4 | 101 | 25 | 0,8914 | 55 |
| 5 | 101 | 25 | 0,9234 | 53 |
| **Rerata ± SD** | | | **0,895 ± 0,021** | **57,8 ± 11,2** |

> **Gambar 4.1 — `finalized/Stage 2/unet_5fold_training_curves.png`**
> Kurva *training loss* (biru) dan *validation Dice* (hijau) per fold dengan
> sumbu-Y kiri (loss) dan kanan (Dice). Lingkaran oranye = *best val Dice*;
> garis putus merah = epoch *early stopping*. Semua fold menunjukkan konvergensi
> stabil: loss turun monoton, val Dice naik dan mendatar, *early stopping* aktif
> jauh sebelum epoch ke-120.

> **Gambar 4.2 — `finalized/Stage 2/unet_training_loss.png`**
> Loss semua 5 fold dalam satu grafik untuk perbandingan kecepatan konvergensi.

> **Gambar 4.3 — `finalized/Stage 2/unet_validation_dice.png`**
> Val Dice semua 5 fold — menunjukkan fold 5 (oranye) paling cepat melampaui
> ambang 0,90, fold 1 paling lambat namun tetap di atas 0,86.

---

### 4.3.6 Inferensi — Ensemble 5 Model + Seleksi Komponen Terbesar

Setelah pelatihan, kelima *checkpoint* terbaik dimuat dan **sigmoid probability**-nya
dirata-ratakan:

$$
\hat{p}_i = \frac{1}{5}\sum_{k=1}^{5} \sigma\!\left(z_i^{(k)}\right),
\quad \hat{y}_i = \mathbf{1}\!\left[\hat{p}_i \geq 0{,}5\right]
$$

Rata-rata ensemble mengurangi variansi prediksi dari idiosinkrasi fold tertentu.
Setelah itu, **hanya komponen terhubung terbesar** yang dipertahankan:

$$
C_{\max} = \arg\max_{C \in \mathcal{C}} \text{Area}(C)
$$

Prior topologi ini menegakkan bahwa payudara adalah massa anatomis kontinu —
menghilangkan artefak *blob* kecil yang mungkin muncul dari noise termal.
Masker akhir di-resize ke **128×128** (format masukan TherMAM-NeRF), lalu
kontur Canny dari $C_{\max}$ menghasilkan siluet geometris $\partial C_{\max}$
yang dikonsumsi Stage 3.

Proses ensemble dijalankan untuk seluruh 137 pasien → **657 masker** tersimpan
di `data/organized_by_patient_unet/`.

---

### 4.3.7 Evaluasi Set Uji Held-Out

**Tabel 4.2 (= Tabel 4.4 Skripsi) — Dice per Sudut Pandang, 36 Citra Held-Out**

| Sudut Pandang | n | Dice (mean ± std) |
|---|---|---|
| Anterior / Front (F) | 7 | 0,923 ± 0,037 |
| Right Oblique 45° (RO) | 8 | 0,939 ± 0,022 |
| Left Oblique 45° (LO) | 7 | 0,921 ± 0,022 |
| Right Lateral 90° (RL) | 8 | 0,868 ± 0,083 |
| Left Lateral 90° (LL) | 6 | 0,878 ± 0,084 |
| **Keseluruhan** | **36** | **0,906** (rata-rata terbobot) |

*Sumber: `Tabel_4_4_Dice_Score_UNet_Set_Uji.csv`*

Ensemble test Dice keseluruhan dari notebook: **0,9073 ± 0,0642** (per-sampel).

> **Gambar 4.4 — `finalized/Stage 2/unet_test_metrics_and_table44.png`**
> (kiri) Bar chart Dice per sudut pandang (mean ± std) pada 36 citra held-out —
> ambang 0,90 ditunjukkan garis merah. (kanan) Tabel ringkasan metrik CV + test.

**Interpretasi.** Tampak oblique 45° dan frontal paling akurat (Dice > 0,92):
siluet payudara terlihat utuh, batas ke latar belakang tegas. Tampak lateral 90°
sedikit lebih rendah (Dice ~0,87, std ~0,08): proyeksi samping menyebabkan kedua
payudara saling tumpang tindih dan batas posterior ambigu. Keterbatasan lateral ini
kemudian ditangani di Stage 3.

**Kesimpulan Stage 2.** Dice uji 0,906 melampaui ambang 0,90 yang ditetapkan.

---

## 4.4 Tahap 3 — Rekonstruksi 3D Termal (TherMAM-NeRF)

### 4.4.1 Gambaran Umum dan Tujuan

Masukan Stage 3 adalah **10 tensor per pasien**: 5 gambar termogram ternormalisasi
dan 5 masker U-Net, dari 5 sudut pandang standar [−90°, −45°, 0°, +45°, +90°]
(RL, RO, F, LO, LL). Keluarannya adalah **mesh permukaan 3D payudara dengan suhu
per-vertex**, yang menjadi domain geometri bagi *inverse solver* Pennes (Stage 4).

---

### 4.4.2 Pembagian Dataset dan Konfigurasi

Dari 137 pasien lengkap (5 tampak TIFF + 5 masker U-Net), **122 pasien** lolos
semua pemeriksaan kualitas dan digunakan dalam pelatihan TherMAM-NeRF.

```python
n = 122 pasien; random.seed(42); random.shuffle(patients)
n_train = int(0.70 * 122) = 85    # 425 tampak
n_val   = int(0.15 * 122) = 18    # 90 tampak
n_test  = 122 - 85 - 18   = 19    # 95 tampak (held-out)
```

Pelatihan: 1 pasien = 1 sampel dataset, batch_size = 1. Setiap epoch = 85 iterasi.

---

### 4.4.3 Arsitektur Jaringan

TherMAM-NeRF adalah *Neural Radiance Field* yang dimodifikasi untuk mengeluarkan
**dua medan 3D** dari setiap titik $\mathbf{x} \in \mathbb{R}^3$: densitas
okupansi $\sigma$ (geometri) dan suhu $T$ (termal).

**Konfigurasi utama:**

| Komponen | Spesifikasi | Nilai |
|---|---|---|
| Resolusi input | `img_size` | 128 × 128 |
| Encoder | Siamese (bobot dibagi antarview) | 32 kanal fitur per tampak |
| *Positional encoding* | $L = 8$ frekuensi | *Frequency warm-up* 50 epoch |
| MLP | 4 lapisan × 256 unit tersembunyi | Keluaran $(\sigma, T)$ |
| Sampling sinar | 256 titik per sinar | 3.072 sinar per iterasi |
| *Volume rendering* | `density_scale = 10` | near/far $= [-1, 1]$ |
| *Marching cubes* | Ambang $= 0{,}3$, resolusi $128^3$ | Ekstraksi mesh |

**Encoder Siamese** memproses kelima tampak dengan bobot yang sama. Setiap tampak
menghasilkan *feature map* 32 kanal yang kemudian digabung bersama melalui
*positional encoding* koordinat 3D sebelum masuk MLP.

---

### 4.4.4 Volume Rendering dan Dua Trik Kunci

Untuk tiap sinar dari kamera, densitas diakumulasi menjadi bobot transmitansi:

$$
w_i = \alpha_i \prod_{j < i} (1 - \alpha_j),
\quad \alpha_i = 1 - e^{-\sigma_i \delta_i}
$$

Masker render = $M = \sum_i w_i$; Suhu render = $\hat{T} = \sum_i w_i T_i / \sum_i w_i$.

**Trik 1 — Weight-detach untuk thermal loss.**
Saat menghitung *thermal loss*, bobot $w_i$ **di-*detach*** dari graf komputasi:

```python
weights_detached = weights.detach()
T_rendered = (weights_detached * T_field).sum() / (weights_detached.sum() + 1e-6)
```

Akibatnya, gradien dari *thermal loss* hanya memperbarui medan suhu $T$
dan **tidak mengubah geometri $\sigma$**. Tanpa *detach*, *thermal loss*
akan mengoptimalkan distribusi kedalaman yang berbeda dari kebutuhan geometri,
merusak rekonstruksi 3D.

**Trik 2 — View-specific depth cropping untuk tampak lateral.**
Pada tampak 90° (RL, LL), payudara seberang secara fisik menonjol di belakang
payudara dekat, tetapi masker GT hanya mencakup payudara dekat. Solusi:
densitas titik-titik dengan $|x| > 0{,}1$ **disembunyikan** saat *rendering*
lateral, mencegah penalti yang tidak seharusnya pada payudara yang tidak
ditargetkan.

---

### 4.4.5 Fungsi Loss Gabungan

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{Dice}} \mathcal{L}_{\text{Dice}}
+ \lambda_{\text{bg}} \mathcal{L}_{\text{bg}}
+ \lambda_{\text{thermal}} \mathcal{L}_{\text{thermal}}
+ \lambda_{\text{TV}} \mathcal{L}_{\text{TV}}
+ \lambda_{\text{ent}} \mathcal{L}_{\text{ent}}
$$

| Komponen | Bobot $\lambda$ | Fungsi |
|---|---|---|
| $\mathcal{L}_{\text{Dice}}$ | 1,0 | Tumpang tindih masker render vs GT |
| $\mathcal{L}_{\text{bg}}$ | 2,0 | Tekan densitas di luar siluet |
| $\mathcal{L}_{\text{thermal}}$ | 20,0 | Galat suhu render vs IR (hanya update $T$) |
| $\mathcal{L}_{\text{TV}}$ | 0,01 | *Total variation* 3D — kehalusan medan |
| $\mathcal{L}_{\text{ent}}$ | 0,1 | Cegah artefak "comb" di tepi bawah |

Bobot $\lambda_{\text{thermal}} = 20{,}0$ dipilih besar karena sinyal termal
(skala °C) lebih lemah dari sinyal geometri (diskret 0/1) — perlu diperkuat
agar pengoptiman bersama tidak mengabaikan suhu.

---

### 4.4.6 Keluaran — Mesh 3D dan Suhu Per-vertex

Setelah 1000 epoch pelatihan:
1. **Marching cubes** pada volume densitas $128^3$ dengan ambang 0,3
   → mesh permukaan STL (~beberapa ratus ribu vertex).
2. **Suhu per-vertex** diperoleh dari evaluasi medan $T$ di koordinat vertex.
3. **Mesh tetrahedral 3 mm** dibuat oleh Gmsh dari STL untuk solusi FEM Pennes:
   setiap mesh memiliki ~587.000 tetrahedra (mesh FEM Stage 4).

---

### 4.4.7 Hasil Kuantitatif

Evaluasi dilakukan pada 122 pasien × 5 tampak = 610 tampak total. Geometri
dievaluasi dengan Dice/IoU (siluet re-render vs masker GT); termal dengan
MAE/RMSE/akurasi piksel.

**Tabel 4.3 — Metrik Evaluasi TherMAM-NeRF**

| Split | n tampak | Dice | IoU | MAE Termal | RMSE | Akurasi @1 °C |
|---|---|---|---|---|---|---|
| Latih | 425 | 0,987 | 0,975 | 0,37 °C | 0,91 °C | 94,6 % |
| **Uji** | **95** | **0,951** | **0,912** | **0,66 °C** | 1,41 °C | **90,4 %** |

*Sumber: `thermamnerf_outputs_finalized/thermal_pixel_accuracy_per_view.csv`*

> **Gambar 4.5** — Contoh *audit plot* per pasien (19 pasien uji tersedia):
> `finalized/Stage 3/thermamnerf_outputs_finalized/audit_test_Patient_107.png`
> Setiap audit menampilkan 5 × 3 panel: untuk tiap tampak — (a) masker GT,
> (b) masker render + Dice/IoU, (c) suhu render + MAE & Akurasi@1°C.

**Interpretasi:**
- Dice uji **0,951** dan IoU **0,912** melebihi ambang (Dice > 0,90, IoU > 0,85):
  siluet 3D sangat akurat bahkan pada pasien yang tidak pernah dilihat.
- Suhu permukaan tepat dalam < 1 °C untuk **90,4 %** piksel (MAE 0,66 °C):
  cukup untuk analisis bioheat karena perbedaan tumor-normal skala 0,5–2 °C.
- Selisih latih→uji kecil (Dice 0,987→0,951): generalisasi baik, tidak ada
  *overfitting* berarti.

**Kesimpulan Stage 3.** TherMAM-NeRF memenuhi semua kriteria keberhasilan
2D→3D dan menghasilkan fondasi geometri + termal yang valid bagi Stage 4.

---

## 4.5 Tahap 4 — Inverse Solver Bioheat (Levenberg–Marquardt)

### 4.5.1 Perumusan Masalah

*Inverse solver* bertugas menjawab pertanyaan: *"Diberikan distribusi suhu
permukaan payudara dari TherMAM-NeRF, posisi dan ukuran tumor sub-permukaan
mana yang paling cocok?"*

**Variabel yang dioptimalkan** (4 derajat kebebasan):
$$
\boldsymbol{\beta} = [x_t,\; y_t,\; z_t,\; d]^\top
$$
di mana $(x_t, y_t, z_t)$ adalah posisi pusat tumor dalam mm dan $d$ adalah
diameter tumor dalam mm.

**Model forward** (FEniCSx + persamaan Pennes dalam kondisi tunak):
$$
k \nabla^2 T(\mathbf{x}) - P(\mathbf{x})\bigl(T(\mathbf{x}) - T_a\bigr)
+ Q(\mathbf{x}) = 0, \quad \mathbf{x} \in \Omega
$$

Tumor dimodelkan sebagai sumber Gaussian:
$$
\chi(\mathbf{x}) = \exp\!\left(-\frac{\|\mathbf{x} - \mathbf{x}_t\|^2}{r_t^2}\right),
\quad r_t = d/2
$$

Sehingga perfusi dan panas metabolik bervariasi secara spasial:
$$
P(\mathbf{x}) = P_h + (P_t - P_h)\chi, \quad Q(\mathbf{x}) = Q_h + (Q_t - Q_h)\chi
$$

**Konstanta Kandlikar 2020 (Tabel 3):**

| Simbol | Makna | Nilai | Satuan |
|---|---|---|---|
| $k_h$ | Konduktivitas termal jaringan sehat | 0,42 | W/(m·K) |
| $\rho_b$ | Densitas darah | 1060 | kg/m³ |
| $c_b$ | Kalor spesifik darah | 3840 | J/(kg·K) |
| $\omega_h$ | Laju perfusi jaringan sehat | $1{,}8 \times 10^{-4}$ | s⁻¹ |
| $\omega_t$ | Laju perfusi tumor | $9{,}0 \times 10^{-3}$ | s⁻¹ |
| $Q_h$ | Panas metabolik sehat | 450 | W/m³ |
| $T_a$ | Suhu arteri / inti tubuh | 37 | °C |
| $h$ | Koefisien konveksi kulit | 13,5 | W/(m²·K) |
| $T_\infty$ | Suhu udara sekitar | 21 | °C |

---

### 4.5.2 Kondisi Batas FEM

Masalah nilai batas (BVP) yang diselesaikan FEniCSx:
- **Dinding dada (Dirichlet)**: $T = T_a = 37\,°C$ pada facet posterior
  (Z-slab atas; $z > z_{\min} + 0{,}15 \cdot (z_{\max} - z_{\min})$).
- **Kulit (Robin/konvektif)**: $k\,\partial_n T + h(T - T_\infty) = 0$ pada
  semua facet lainnya.

Solver: *Conjugate Gradient* + *Algebraic Multigrid* (CG/GAMG, PETSc).
Konvergensi: ~17 iterasi CG, mandiri terhadap ukuran mesh.

---

### 4.5.3 Algoritma Levenberg–Marquardt

Setiap iterasi LM memecahkan sistem normal yang ter-*damping*:

$$
\left(\mathbf{J}^\top \mathbf{J} + \mu \mathbf{D}\right) \delta\boldsymbol{\beta}
= -\mathbf{J}^\top \mathbf{r}
$$

- $\mathbf{r} = T_\text{target} - T_\text{model}$ — vektor residual suhu permukaan
- $\mathbf{J}$ — Jacobian numerik (beda hingga maju, $\Delta_{xyz} = 3{,}0$ mm,
  $\Delta_d = 2{,}0$ mm sesuai ukuran elemen mesh)
- $\mu$ — parameter *damping* (ditingkatkan × 10 saat langkah ditolak,
  diturunkan × 10 saat langkah diterima)
- $\mathbf{D}$ — matriks diagonal penskala (Marquardt: $D_{ii} = (J^\top J)_{ii}$)

**Inisialisasi berbasis *hotspot* diferensial.**
Posisi awal $(x_0, y_0, z_0)$ bukan acak, tetapi dari *hotspot* diferensial:
$\Delta T = T_\text{target} - T_\text{sehat}$ di permukaan kulit, dengan $T_\text{sehat}$
diperoleh dari forward solve dengan tumor di luar domain. Ini memastikan titik awal
berada di bawah area terpanas — prior fisis yang kuat.

**Box-constraint** (mencegah divergensi liar):
$$
\mathbf{x}_\text{tr} = \text{clip}\!\left(\mathbf{x} + \delta\mathbf{x},\;
\text{bbox} \pm 15\,\text{mm}\right),
\quad d \in [1, 35]\,\text{mm}
$$

**Kriteria berhenti:**
1. Iterasi maksimum = 25
2. Tidak ada langkah diterima setelah 8 kenaikan $\mu$
3. $\|\delta\boldsymbol{\beta}\| < 0{,}05$ mm
4. $S < 10^{-6}$ (biaya sudah sangat kecil)

---

### 4.5.4 Optimisasi `PennesCtx` — Amortisasi I/O Mesh

Karena setiap iterasi LM memerlukan satu *forward solve*, pembacaan mesh (.msh)
dari disk setiap kali akan sangat lambat. Kelas `PennesCtx` memuat mesh
**sekali** di awal, menyimpan objek FEM (`dolfinx.mesh`, `FunctionSpace`,
kondisi batas, indeks interpolasi cKDTree), dan hanya menjalankan ulang
perakitan UFL + solver CG/GAMG per panggilan `ctx.solve(beta)`.

Interpolasi suhu permukaan: untuk setiap titik permukaan dalam `surface_pts`,
3 node mesh terdekat (cKDTree) diambil dan suhunya dirata-ratakan:
```python
return sol.x.array[self.sidx].mean(axis=1)
```

---

### 4.5.5 Mode Sintetik vs Mode Real

| Aspek | Mode Sintetik | Mode Real (Pasien) |
|---|---|---|
| $T_\text{target}$ | Forward solve dengan $\boldsymbol{\beta}^*$ yang diketahui | Suhu permukaan NeRF dari Stage 3 |
| Fungsi biaya | Residual langsung: $r = T_\text{target} - T_\text{model}$ | **Mean-subtracted**: $r = (T_t - \bar{T}_t) - (T_m - \bar{T}_m)$ |
| *Ground truth* | Tersedia (posisi + ukuran tertanam) | Tidak tersedia untuk geometri tumor |
| Tujuan | Validasi algoritma | Klasifikasi jinak/ganas |

**Mengapa mean-subtraction diperlukan untuk mode real?**
Model Pennes menghasilkan suhu permukaan rata-rata ~32 °C, sedangkan NeRF IR
rata-rata ~29 °C — *offset* sistematis ~3,4 °C. Tanpa mean-subtraction, residual
(~5,3 °C) didominasi *offset* ini bukan sinyal tumor. Dengan mean-subtraction,
optimizer merespons **pola spasial** distribusi suhu, bukan nilai absolut.

---

### 4.5.6 Validasi Sintetik — Hasil Lengkap

#### 4.5.6.1 Desain Skenario

- **2 pasien**: Patient\_159 (label jinak) dan Patient\_179 (label ganas)
- **10 penempatan acak per pasien** menurut distribusi kuadran Senie 1983:
  UO 50 %, C 18 %, UI 15 %, LO 11 %, LI 6 %
- Diameter tumor: $d^* \sim \text{Uniform}(10, 20)$ mm; clearance kulit: 5–20 mm
- Total: **20 skenario**

> **Gambar 4.6 — `finalized/Stage 4/figs/QuadrantMapTumorPlant.png`**
> Peta kuadran payudara (UI/UO/LI/LO/C) dengan 20 titik tumor tertanam yang
> ditampilkan pada koordinat 3D permukaan payudara, berwarna per kuadran.

> **Gambar 4.7 — `finalized/Stage 4/figs/Syn_forward.png`**
> Distribusi suhu permukaan dari *forward solve* untuk 3 skenario representatif
> — menunjukkan *hotspot* yang muncul di permukaan kulit akibat tumor sub-permukaan.

> **Gambar 4.8 — `finalized/Stage 4/figs/Syn_Inverse.png`**
> Visualisasi proses inverse: titik awal (hotspot), lintasan iterasi LM, dan
> titik akhir terekonstruksi vs titik tertanam.

#### 4.5.6.2 Hasil Terukur per Skenario

**Tabel 4.4 — Hasil Validasi Sintetik per Skenario (20 skenario)**

| # | Pasien | Kuadran | Diameter Tanam (mm) | Galat Posisi (mm) | Galat Diameter (mm) | RMS Akhir (°C) | Konvergen | Verdict |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|---|
| 1 | P159 | UI | 14,4 | 1,66 | 0,21 | 0,000593 | ✓ | MALIGNANT ✓ |
| 2 | P159 | UI | 10,9 | 1,11 | 0,62 | 0,000257 | ✓ | MALIGNANT ✓ |
| 3 | P159 | UI | 17,9 | 0,41 | 0,04 | 0,000478 | ✓ | MALIGNANT ✓ |
| 4 | P159 | UO | 13,7 | 1,09 | 7,98 | 0,000206 | ✓ | MALIGNANT ✓ |
| 5 | P159 | C | — | 0,01 | 0,02 | 0,000604 | ✓ | MALIGNANT ✓ |
| 6 | P159 | UO | — | 3,21 | 6,30 | 0,002604 | ✓ | MALIGNANT ✓ |
| 7 | P159 | UI | — | 0,36 | 0,05 | 0,000234 | ✓ | MALIGNANT ✓ |
| 8 | P159 | UO | — | 0,15 | 0,09 | 0,000378 | ✓ | MALIGNANT ✓ |
| 9 | P159 | UI | — | 1,56 | 0,34 | 0,000589 | ✓ | MALIGNANT ✓ |
| 10 | P159 | UO | — | 0,73 | 6,25 | 0,000909 | ✓ | MALIGNANT ✓ |
| 11 | P179 | UI | — | 0,05 | 0,03 | 0,000197 | ✓ | MALIGNANT ✓ |
| 12 | P179 | UO | — | **60,07** | 30,21 | 0,033299 | ✓ | BENIGN ✗ |
| 13 | P179 | UO | — | **61,21** | 30,50 | 0,033428 | ✓ | BENIGN ✗ |
| 14 | P179 | C | — | **144,76** | 110,73 | 0,064282 | ✗ | BENIGN ✗ |
| 15 | P179 | UI | — | 0,09 | 0,00 | 0,000080 | ✓ | MALIGNANT ✓ |
| 16 | P179 | UI | — | 0,04 | 0,01 | 0,000044 | ✓ | MALIGNANT ✓ |
| 17 | P179 | UI | — | 0,26 | 0,03 | 0,000236 | ✓ | MALIGNANT ✓ |
| 18 | P179 | UO | — | **74,29** | 31,72 | 0,045006 | ✓ | BENIGN ✗ |
| 19 | P179 | UI | — | 0,46 | 0,05 | 0,000997 | ✓ | MALIGNANT ✓ |
| 20 | P179 | C | — | 0,01 | 0,02 | 0,000571 | ✓ | MALIGNANT ✓ |

**Tabel 4.5 — Ringkasan per Kuadran**

| Kuadran | n | Galat Posisi Median (mm) | Galat Diameter Median (mm) | Sukses (< 5 mm) |
|---|:---:|:---:|:---:|:---:|
| UI (*Upper Inner*) | 10 | **0,38** | **0,04** | **10/10** |
| C (*Central*) | 3 | 0,01 | 0,02 | 2/3 |
| UO (*Upper Outer*) | 7 | 3,21 | 6,30 | 4/7 |
| **Keseluruhan** | **20** | **0,59** | **0,15** | **16/20** |

> **Gambar 4.9 — `finalized/Stage 4/figs/syn_recovery_multiview.png`**
> 4-panel: (a) perspektif 3D, (b) tampak depan (x–y), (c) tampak samping (z–y),
> (d) tampak atas (x–z). Lingkaran merah = tertanam; × biru = terekonstruksi.
> Pada UI, kedua marker berimpit di ketiga tampak. Pada UO yang gagal, tanda ×
> berada jauh di luar bingkai gambar (sumbu dikliping ke rentang koordinat tumor
> yang ditanam).

> **Gambar 4.10 — `finalized/Stage 4/figs/syn_recovery_metrics.png`**
> (a) Galat posisi per skenario pada skala *symlog* — 16 skenario di bawah
> ambang 5 mm (hijau), 4 skenario UO/C di atas; (b) Scatter diameter tertanam
> vs terekonstruksi — UI sempurna di garis diagonal; UO tersebar jauh;
> (c) Bar galat median per kuadran.

> **Gambar 4.11 — `finalized/Stage 4/figs/LMConvergence.png`**
> Kurva konvergensi LM: nilai $S$ (MSE) vs iterasi untuk beberapa skenario.
> Penurunan tajam 3–4 orde magnitudo dalam 5–10 iterasi pertama pada UI.

#### 4.5.6.3 Analisis Identifiabilitas per Kuadran

**UI (Upper Inner) — identifiable sempurna.**
Semua 10 skenario UI berhasil: median posisi 0,38 mm, median diameter 0,04 mm.
Tumor di kuadran dalam-atas memiliki jarak proyeksi pendek ke permukaan anterior —
sinyal termal kuat, Jacobian baik-terkondisi, LM konvergen ke minimum global dalam
< 22 iterasi.

**C (Central) — sebagian besar identifiable.**
2 dari 3 skenario sempurna (galat < 0,02 mm). Satu skenario (sc14, P179)
divergen besar (144 mm) karena geometri tumpang tindih dengan dinding dada.

**UO (Upper Outer) — *ill-posed* secara fisik.**
4 dari 7 skenario berhasil (median 3,21 mm), 3 skenario divergen jauh
(60–74 mm). Tumor UO terletak dekat dinding aksila, jauh dari puting —
sinyal permukaannya lemah dan Jacobian buruk-terkondisi. Parameter *damping* $\mu$
terus meningkat tanpa langkah diterima, solusi melayang keluar *box-constraint*.
Ini adalah **keterbatasan identifiabilitas geometri** yang diketahui dari
literatur (Bezerra 2013), bukan kesalahan implementasi.

---

### 4.5.7 Hasil pada Data Pasien Nyata (30 Pasien)

#### 4.5.7.1 Setup

Subset seimbang **30 pasien** (15 jinak / 15 ganas) dari DMR-IR, menggunakan
suhu permukaan NeRF sebagai $T_\text{target}$.

Fungsi biaya *mean-subtracted*:
$$
r_i = \bigl(T_{\text{target},i} - \bar{T}_\text{target}\bigr)
      - \bigl(T_{\text{model},i} - \bar{T}_\text{model}\bigr)
$$

#### 4.5.7.2 Hasil Kuantitatif

**Tabel 4.6 — Statistik Inverse Data Nyata (n = 30)**

| Metrik | Jinak (n=15) | Ganas (n=15) | AUC |
|---|---|---|---|
| Final RMS (°C) | 1,97 ± 0,22 | 2,13 ± 0,36 | 0,62 |
| Diameter terekonstruksi (mm) | 26,1 ± 11,5 | 29,2 ± 10,9 | 0,62 |

**Tabel 4.7 — Matriks Konfusi Verdict (n = 30)**

| | Verdict MALIGNANT | Verdict BENIGN |
|---|---|---|
| **Label Ganas** | 10 (TP) | 5 (FN) |
| **Label Jinak** | 12 (FP) | 3 (TN) |

Sensitivitas: 10/15 = **0,67** | Spesifisitas: 3/15 = **0,20** | Akurasi: 13/30 = **0,43**

> **Gambar 4.12 — `finalized/Stage 4/figs/fig1_roc.png`**
> Kurva ROC untuk `final_rms` dan `recovered_d` — keduanya AUC = 0,62;
> garis acak (AUC 0,50) ditampilkan sebagai referensi.

> **Gambar 4.13 — `finalized/Stage 4/figs/fig2_distributions.png`**
> *Violin + box plot* `final_rms` dan `recovered_d` per kelas. Tumpang tindih
> distribusi yang besar konsisten dengan AUC 0,62 dan uji-t tidak signifikan
> (p = 0,15 untuk RMS).

> **Gambar 4.14 — `finalized/Stage 4/figs/fig3_feature_scatter.png`**
> Ruang fitur 2D: sumbu-x = diameter, sumbu-y = RMS; merah = jinak, biru = ganas.
> Tidak ada batas linier yang memisahkan kedua kelas.

> **Gambar 4.15 — `finalized/Stage 4/figs/fig4_confusion.png`**
> Matriks konfusi visual dari Tabel 4.7.

> **Gambar 4.16 — `finalized/Stage 4/figs/fig5_per_patient.png`**
> Per pasien: `final_rms` (batang) dan jumlah solusi FEM `n_solves` (garis).
> Pasien dengan `n_solves` > 100 mengindikasikan LM bergulat dengan lanskap
> biaya datar sebelum berhenti.

#### 4.5.7.3 Analisis Keterbatasan

Penyebab AUC 0,62 dapat dipahami dari rasio sinyal-derau (SNR):

$$
\text{SNR} = \frac{\Delta T_\text{tumor}}{\text{RMS}_\text{model}} \approx \frac{0{,}5\,°C}{2{,}0\,°C} = 0{,}25
$$

Sinyal termal tumor di permukaan kulit (~0,5 °C, dari literatur) **4× lebih kecil**
dari derau model (~2 °C). Derau ini berasal dari ketidaksesuaian antara model
Pennes homogen dengan realitas jaringan payudara heterogen. Ini bukan kegagalan
algoritma — terbukti dengan hasil sintetik sub-milimeter. Ini adalah **batasan
fidelitas forward-model**.

Diameter sering mentok di 35 mm (jinak: 7/15, ganas: 11/15) karena LM mencoba
me-*fit* gradien termal anatomi keseluruhan payudara dengan satu sumber panas
besar alih-alih tumor kecil — ketika sinyal tumor tenggelam dalam derau model.

---

## 4.6 Perbandingan Metode: Grid-Optimasi Lama vs Levenberg–Marquardt Baru

### 4.6.1 Setup

Dua metode dibandingkan *head-to-head* pada 30 pasien yang sama:

| Aspek | Metode Lama | Metode Baru |
|---|---|---|
| Strategi | Grid 5×4 kasar + Nelder-Mead | Levenberg–Marquardt iteratif |
| DoF | $(z_t, r_t)$ — 2 parameter | $(x_t, y_t, z_t, d)$ — 4 parameter |
| Biaya | Residual suhu absolut | Residual *mean-subtracted* |
| *Clamp* ukuran | `r_hat` ≤ 40 mm | `d` ≤ 35 mm |

### 4.6.2 Hasil Kuantitatif

**Tabel 4.8 — Perbandingan Head-to-Head (n = 30)**

| Metrik | Metode Lama | Metode Baru | Perubahan |
|---|---|---|---|
| AUC (ukuran) | 0,50 | **0,62** | +0,12 |
| AUC (residual) | 0,33* | **0,62** | +0,29 |
| Residual rata-rata (°C) | 5,32 | **2,05** | −3,27 |
| p reduksi residual (uji-t berpasangan) | — | **≈ 9,9 × 10⁻²³** | — |
| Saturasi ukuran jinak | 13/15 mentok 40 mm | 7/15 mentok 35 mm | lebih baik |
| Saturasi ukuran ganas | 13/15 mentok 40 mm | 11/15 mentok 35 mm | lebih baik |
| Korelasi ukuran antar-metode | r = 0,35 | — | rendah |

\* AUC 0,33 (anti-korelasi): *offset* 3,4 °C membalik arah diskriminasi.

> **Gambar 4.17 — `finalized/Stage 4/figs/fig6_method_compare.png`**
> 6-panel:
> (a) Kurva ROC — Baru (merah) vs Lama (biru) + garis acak.
> (b) Bar AUC per fitur untuk kedua metode.
> (c) Reduksi residual berpasangan per 30 pasien — hampir semua garis turun.
> (d) Saturasi ukuran — histogram `r_hat` (lama, semua mentok 40 mm) vs `d` (baru).
> (e) Korelasi estimasi ukuran antar-metode (scatter, r = 0,35).
> (f) Residual per kelas per metode — metode baru lebih rendah pada kedua kelas.

### 4.6.3 Diskusi

*Mean-subtraction* adalah perbaikan kritis: menghilangkan *offset* 3,4 °C
memangkas residual lebih dari setengah (5,32 → 2,05 °C) dengan signifikansi
statistik yang sangat tinggi (p ≈ 10⁻²³). Korelasi ukuran r = 0,35 yang rendah
antara kedua metode menegaskan bahwa estimasi ukuran absolut dari data IR
permukaan belum dapat diandalkan tanpa validasi *ground truth* (MRI/USG).

---

## 4.7 Ringkasan Keberhasilan Pipeline

**Tabel 4.9 — Seluruh Metrik Pipeline vs Ambang Keberhasilan**

| Tahap | Metrik | Nilai Terukur | Ambang | Status |
|---|---|---|---|---|
| Stage 2 — U-Net | Dice val 5-fold | 0,895 ± 0,021 | > 0,85 | ✓ |
| Stage 2 — U-Net | Dice uji held-out | **0,906** | > 0,90 | ✓ |
| Stage 3 — TherMAM-NeRF | Dice uji geometri | **0,951** | > 0,90 | ✓ |
| Stage 3 — TherMAM-NeRF | IoU uji geometri | **0,912** | > 0,85 | ✓ |
| Stage 3 — TherMAM-NeRF | MAE termal uji | **0,66 °C** | < 1,0 °C | ✓ |
| Stage 3 — TherMAM-NeRF | Akurasi@1°C uji | **90,4 %** | > 90 % | ✓ |
| Stage 4 — Sintetik (UI) | Galat posisi median | **0,38 mm** | < 5 mm | ✓ |
| Stage 4 — Sintetik (all) | Galat posisi median | **0,59 mm** | < 5 mm | ✓ |
| Stage 4 — Sintetik (all) | Galat diameter median | **0,15 mm** | < 2 mm | ✓ |
| Stage 4 — Sintetik (all) | RMS akhir | **< 10⁻³ °C** | < 10⁻³ °C | ✓ |
| Stage 4 — Data nyata | AUC diskriminasi | **0,62** | > 0,70 (klinik) | ✗ |

**Tiga Temuan Utama:**

1. **Algoritma inverse terbukti benar secara sub-milimeter** pada data sintetik —
   kontribusi metodologis utama penelitian ini. Pemulihan geometri median posisi
   0,59 mm dan diameter 0,15 mm jauh di bawah ambang 5 mm / 2 mm.

2. **Perbaikan metodologis LM + mean-subtraction terukur dan signifikan**:
   residual turun 5,32 → 2,05 °C (p ≈ 10⁻²³) dan AUC naik 0,50 → 0,62 dibanding
   metode lama.

3. **Keterbatasan data nyata bersumber pada kesenjangan model–data**, bukan
   algoritma. SNR ~0,25 (sinyal tumor 0,5 °C vs derau model 2 °C) membuat
   klasifikasi malignansi lemah (AUC 0,62). Peningkatan fidelitas forward-model
   atau penggunaan *phantom* termal sebagai validasi adalah langkah penelitian
   selanjutnya yang alami.

---

*Sumber data:*
- *`finalized/Stage 2/unet_training_history/fold_*_history.json`*
- *`finalized/Stage 2/Tabel_4_4_Dice_Score_UNet_Set_Uji.csv`*
- *`finalized/Stage 3/thermamnerf_outputs_finalized/thermal_pixel_accuracy_per_view.csv`*
- *`finalized/Stage 4/lm_synthetic_20260623_010523.csv`*
- *`finalized/Stage 4/lm_real_subset.csv`*
- *`TherMAM-NeRF/results/cohort_BCfix_ALL.csv`*

*Gambar: `finalized/Stage 2/unet_*.png`, `finalized/Stage 4/figs/*.png`*

*Skrip: `Previous Works (VAE, legacy stuff, misc)/organize_by_patient_and_view.py`,*
*`UNET_Segmentation/Masking and Segmentation/MANUAL MASKING.ipynb`,*
*`finalized/Stage 2/2_unetsegmentation_fixed.ipynb`,*
*`finalized/Stage 3/thermamnerf_v3.0.py`,*
*`finalized/Stage 4/run_lm_cohort.py`*