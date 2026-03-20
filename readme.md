Tentu, ini adalah dokumen **`README.md`** yang sangat komprehensif, teknis, dan *beginner-friendly*. Dokumen ini dirancang untuk memenuhi permintaan Anda: mencakup seluruh perjalanan proyek (Epic 1 hingga Epic 5), membahas arsitektur, tantangan teknis, alasan pemilihan library, hingga troubleshooting.

Panjang dokumen ini dirancang ekstensif untuk memberikan dokumentasi setingkat proyek profesional.

```markdown
# 📦 Sistem AI Forecasting & Optimasi Produksi UMKM Patchwork

![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red)

> **Repository Resmi Proyek Akhir:** Sistem Perencanaan Produksi Cerdas Berbasis Deep Learning (LSTM) dan Optimasi Stokastik (TSSP).

---

## 📑 Daftar Isi
1.  [Gambaran Umum Proyek](#1-gambaran-umum-proyek)
2.  [Latar Belakang & Pernyataan Masalah](#2-latar-belakang--pernyyataan-masalah)
3.  [Arsitektur Sistem](#3-arsitektur-sistem)
4.  [Struktur Repositori](#4-struktur-repositori)
5.  [EPIC 1: Data Engineering & Synthetic Generation](#5-epic-1-data-engineering--synthetic-generation)
    *   [Tantangan & Solusi](#tantangan--solusi-epic-1)
    *   [Spesifikasi Library](#spesifikasi-library-epic-1)
6.  [EPIC 2: AI Model Development (LSTM)](#6-epic-2-ai-model-development-lstm)
    *   [Arsitektur Model](#arsitektur-model)
    *   [Evaluasi & Metrik](#evaluasi--metrik)
7.  [EPIC 3: Optimization Engine (TSSP)](#7-epic-3-optimization-engine-tssp)
    *   [Logika Skenario P10/P50/P90](#logika-skenario-p10p50p90)
8.  [EPIC 4: Dashboard & Visualization (Streamlit)](#8-epic-4-dashboard--visualization-streamlit)
    *   [Fitur Utama Aplikasi](#fitur-utama-aplikasi)
    *   [Spesifikasi Library UI](#spesifikasi-library-ui)
9.  [EPIC 5: Deployment & Production](#9-epic-5-deployment--production)
    *   [Panduan Instalasi Lengkap](#panduan-instalasi-lengkap)
    *   [Troubleshooting](#troubleshooting)
10. [Kesimpulan](#10-kesimpulan)

---

## 1. Gambaran Umum Proyek

Proyek ini adalah implementasi *end-to-end* dari sistem perencanaan produksi cerdas untuk UMKM di bidang fashion/tekstil (Patchwork). Sistem ini mengatasi masalah klasik manajemen rantai pasok: **Ketidakpastian Permintaan (Demand Uncertainty)**.

Dengan mengintegrasikan kekuatan **Deep Learning (LSTM)** untuk peramalan permintaan (*forecasting*) dan model **Two-Stage Stochastic Programming (TSSP)** untuk pengambilan keputusan optimasi, sistem ini mampu memberikan rekomendasi produksi yang presisi, mengurangi risiko *lost sales* (kehabisan stok) dan *overstock* (penumpukan stok).

### Tujuan Utama
1.  **Generatif:** Mampu menghasilkan data sintetis yang valid dari data parameter terbatas.
2.  **Prediktif:** Memprediksi permintaan di masa depan menggunakan pola historis dan musiman.
3.  **Optimatif:** Memberikan rekomendasi keputusan produksi (jumlah produksi, kebutuhan lembur, estimasi biaya).
4.  **User-Friendly:** Menyajikan kompleksitas algoritma dalam antarmuka dashboard yang mudah dipahami.

---

## 2. Latar Belakang & Pernyataan Masalah

### Konteks Bisnis
UMKM Patchwork menghadapi tantangan dalam perencanaan produksi karena dua faktor utama:
1.  **Pola Fluktuatif:** Permintaan tidak datang secara konstan. Ada bulan-bulan tertentu (Ramadhan, Natal, Tahun Ajaran Baru) di mana permintaan melonjak drastis, sementara bulan lain sepi.
2.  **Keterbatasan Data:** Sebagian besar UMKM tidak memiliki data penjualan historis dalam format *time-series* yang rapi. Data yang ada seringkali hanya berupa profil produk (biaya, waktu produksi) atau hasil survei yang statis.

### Masalah Teknis
*   **Data Scarcity:** Model AI seperti LSTM membutuhkan ratusan hingga ribuan data poin untuk belajar. Data survei yang hanya 142 baris tidak cukup.
*   **Kalender Dinamis:** Event keagamaan seperti Ramadhan berpindah tanggal setiap tahunnya (Kalender Hijriah). Model statis tidak akan akurat memprediksi musim Ramadhan di tahun berikutnya.
*   **Optimasi Multi-Objektif:** UMKM ingin memaksimalkan profit tetapi memiliki keterbatasan kapasitas produksi (tenaga kerja, jam lembur maksimal).

---

## 3. Arsitektur Sistem

Sistem ini dibangun dengan arsitektur **Pipeline Data-Driven** yang modular.

```mermaid
graph TD
    %% BLOK 1: DATA INPUT
    A[Data Survei 142 Responden] --> B(Epic 1: Data Engineering)
    
    %% BLOK 2: PREPARATION
    B --> C{Synthetic Generator}
    C -->|Logika Musiman| D[Dataset Sintetis 6000+ Baris]
    
    %% BLOK 3: AI ENGINE
    D --> E(Epic 2: LSTM Training)
    E --> F[Model Trained (model.h5)]
    
    %% BLOK 4: OPTIMIZATION
    F --> G(Epic 3: TSSP Logic)
    G --> H[Rekomendasi Produksi]
    
    %% BLOK 5: USER INTERFACE
    H --> I(Epic 4: Streamlit Dashboard)
    I --> J((User: Pemilik UMKM))
```

---

## 4. Struktur Repositori

Berikut adalah struktur folder standar proyek ini:

```
project_patchwork/
│
├── data/                          # Folder data mentah dan hasil generate
│   ├── Dataset.csv                # Data survei 142 responden (Input)
│   └── dataset_synthetic_umkm.csv # Dataset sintetis final (Output Epic 1)
│
├── models/                        # Penyimpanan artefak model
│   ├── model_lstm.h5              # Bobot model Keras
│   ├── scaler_X.pkl               # Scaler fitur input (Penting!)
│   └── scaler_y.pkl               # Scaler target output (Penting!)
│
├── notebooks/                     # Jupyter/Colab Notebooks untuk eksperimen
│   └── epic_1_2_3_experiment.ipynb
│
├── src/                           # Source code utama
│   └── app.py                     # Script dashboard Streamlit
│
├── README.md                      # Dokumentasi ini
└── requirements.txt               # Daftar library dependencies
```

---

## 5. EPIC 1: Data Engineering & Synthetic Generation

### Deskripsi Teknis
Tahap ini adalah fondasi terpenting. Karena data historis tidak ada, kita menggunakan strategi **Synthetic Data Generation**. Kita tidak membuat data acak, melainkan **mensimulasikan realitas bisnis** berdasarkan "aturan main" yang diekstrak dari data survei.

Prosesnya adalah **Inverse Decomposition**: kita membangun data dari komponen-komponen penyusunnya (Trend, Seasonality, Noise) secara terbalik.

### Tantangan & Solusi (Epic 1)

| Tantangan (Challenge) | Solusi Teknis | Alasan Teknis |
| :--- | :--- | :--- |
| **Format Data Salah** | Data survei bersifat *cross-sectional* (profil), sedangkan LSTM butuh *time-series*. | Menggunakan `pandas.date_range` untuk membuat kerangka waktu, lalu mengisi nilainya dengan parameter dari survei. |
| **Kalender Dinamis** | Tanggal Ramadhan bergeser setiap tahun. | Membuat *dictionary* rentang tanggal dinamis untuk 3 tahun (2022-2024) dan logika *mapping* ke dataset. |
| **Class Imbalance** | Event musiman jarang terjadi dibanding hari biasa. | Penerapan *Multiplier Logic* (Demand x 3.5) pada hari musiman dan penambahan *Jitter/Noise* pada hari biasa. |
| **Kehilangan Fitur Penting** | Kolom produk (Produk) hilang saat One-Hot Encoding. | Rekonstruksi kolom produk dari nama kolom One-Hot (`Prod_Sajadah` -> `Sajadah`) menggunakan fungsi string parsing. |

### Spesifikasi Library (Epic 1)

*   **`pandas`**: Standar industri untuk manipulasi data tabular.
    *   *Fungsi Utama:* `groupby` untuk agregasi parameter, `date_range` untuk generasi waktu.
*   **`numpy`**: Komputasi numerik tingkat rendah.
    *   *Fungsi Utama:* `random.normal` untuk membuat *noise* (variasi) agar data sintetis terlihat natural, bukan angka bulat sempurna.
*   **`random`**: Modul bawaan Python.
    *   *Fungsi Utama:* Menentukan baseline demand acak.

---

## 6. EPIC 2: AI Model Development (LSTM)

### Deskripsi Teknis
Model jantung sistem ini menggunakan **Long Short-Term Memory (LSTM)**, jenis Recurrent Neural Network (RNN) yang mampu "mengingat" informasi dari masa lalu (context) untuk prediksi masa depan.

### Arsitektur Model
Model dirancang untuk mencegah *overfitting* pada data sintetis:
1.  **Input Layer:** Menerima bentuk data `(Samples, 7 Timesteps, 16 Features)`.
2.  **LSTM Layer 1 (64 Units):** Menangkap pola urutan waktu. `return_sequences=True` diperlukan untuk stacking layer LSTM.
3.  **Dropout (0.2):** "Mematikan" 20% neuron secara acak setiap epoch. Ini memaksa model untuk tidak menghafal data (robust).
4.  **LSTM Layer 2 (64 Units):** Layer pemrosesan akhir.
5.  **Dense Layer (1 Unit):** Output angka prediksi tunggal.

### Evaluasi & Metrik
Penggunaan **Huber Loss** sebagai fungsi loss dipilih karena:
*   MSE (Mean Squared Error) sangat sensitif terhadap outlier (angka besar dikuadratkan).
*   Huber Loss linear untuk error besar, sehingga lebih stabil menghadapi lonjakan musiman ekstrem.

**Hasil Evaluasi Model:**
*   **RMSE: 3.79** -> Rata-rata error prediksi hanya sekitar 4 unit.
*   **RMAE: 0.8072** -> Model 19.3% lebih akurat dibanding metode Naive Forecast (tebakan sederhana).

---

## 7. EPIC 3: Optimization Engine (TSSP)

### Deskripsi Teknis
AI hanya mengeluarkan angka prediksi. Bisnis butuh keputusan. Modul ini mengubah prediksi menjadi skenario bisnis.

### Logika Skenario P10/P50/P90
Kita menggunakan Standar Deviasi Error (RMSE) untuk membuat interval keyakinan:

1.  **P50 (Realistis):** Nilai prediksi mentah. Target utama.
2.  **P10 (Pesimis):** `Prediksi - (1.28 * Std_Dev)`. Batas bawah. Di sini kita terapkan **Floor Logic (Min 1 Unit)** agar sistem tidak memprediksi produksi 0 (nol), yang akan menyebabkan *lost sales*.
3.  **P90 (Optimis):** `Prediksi + (1.28 * Std_Dev)`. Batas atas. Antisipasi *overstock*.

### Alur Keputusan TSSP
1.  Cek kapasitas normal vs P50.
2.  Jika P50 > Normal -> Hitung Jam Lembur.
3.  Jika Jam Lembur > Max Lembur -> Status **OVERLOAD** (Sarankan cari tenaga borongan/tolak order).
4.  Jika Jam Lembur < Max Lembur -> Status **BUTUH LEMBUR**.

---

## 8. EPIC 4: Dashboard & Visualization (Streamlit)

### Deskripsi Teknis
Antarmuka pengguna dibangun dengan **Streamlit**. Fokus utama adalah **Interpretability**.

### Fitur Utama Aplikasi
*   **Multi-Select Produk:** Mendukung prediksi simultan untuk banyak produk.
*   **Kalender Dinamis:** Terintegrasi dengan library `hijridate` untuk deteksi otomatis musim Ramadhan/Idul Fitri tanpa hardcode manual.
*   **Dual Charts:** Menampilkan Bar Chart (Perbandingan P10/P50/P90) dan Gauge Chart (Utilisasi Kapasitas).
*   **PDF Report Generator:** Fitur download laporan otomatis yang memastikan karakter Unicode (Emoji) dihapus (`clean_text_for_pdf`) agar kompatibel dengan library `fpdf`.

### Spesifikasi Library UI
*   **`streamlit`**: Framework tercepat untuk membangun aplikasi data science.
*   **`plotly`**: Visualisasi interaktif (zoom, hover) yang lebih baik dari Matplotlib statis.
*   **`fpdf`**: Membuat file PDF on-the-fly di memori tanpa perlu save file fisik.

---

## 9. EPIC 5: Deployment & Production

### Panduan Instalasi Lengkap (Step-by-Step)

#### 1. Prasyarat Sistem
*   **OS:** Windows 10/11, macOS, Linux.
*   **Python:** Versi **3.10.x** (Disarankan). *Hindari 3.12 untuk kompatibilitas TensorFlow yang stabil.*
*   **RAM:** Min 4GB.

#### 2. Setup Environment
```bash
# 1. Clone repositori
git clone https://github.com/username/project_patchwork.git
cd project_patchwork

# 2. Buat Virtual Environment (venv)
python -m venv venv

# 3. Aktifkan venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install semua dependencies
pip install -r requirements.txt
```

#### 3. Menjalankan Aplikasi
Pastikan file `model_lstm.h5`, `scaler_X.pkl`, `scaler_y.pkl`, dan `dataset_synthetic_umkm.csv` ada di folder yang sama dengan `app.py`.

```bash
streamlit run app.py
```

### Troubleshooting (Masalah Umum)

#### Q: Error `StreamlitDuplicateElementId`?
**A:** Terjadi karena membuat grafik di dalam *loop* tanpa ID unik.
*   *Solusi:* Pastikan `st.plotly_chart(fig, key=f"unik_key_{i}")` memiliki argumen `key`.

#### Q: Error `UnicodeEncodeError` saat Download PDF?
**A:** Library FPDF standar hanya mendukung karakter Latin-1. Emoji (🚧, ✅) menyebabkan error.
*   *Solusi:* Gunakan fungsi regex `re.sub(r'[^\x00-\xff]', '', text)` untuk membersihkan teks sebelum dimasukkan ke PDF.

#### Q: Hasil prediksi di app berbeda jauh?
**A:** Periksa urutan kolom (Feature Order). Scaler yang di-load harus diinisialisasi dengan urutan kolom yang sama persis saat training. Gunakan `scaler_X.feature_names_in_` untuk memvalidasi.

---

## 10. Kesimpulan

Sistem ini berhasil mendemonstrasikan bagaimana teknologi AI canggih (LSTM) dapat diimplementasikan pada skala UMKM dengan keterbatasan data. Melalui teknik **Synthetic Data Generation** yang terstruktur dan **Optimasi Stokastik**, sistem ini memberikan nilai tambah berupa efisiensi biaya dan ketenangan pikiran bagi pelaku usaha dalam menghadapi ketidakpastian permintaan.

---

*Dibuat dengan ❤️ untuk kemajuan UMKM Indonesia.*
```