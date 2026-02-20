# 🤟 SIBI Sign Language Recognition

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-00BCD4?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![Keras](https://img.shields.io/badge/Keras-2.15-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Sistem pengenalan Bahasa Isyarat SIBI (Sistem Isyarat Bahasa Indonesia) berbasis Deep Learning**  
menggunakan LSTM Neural Network dengan integrasi AI generatif untuk menghasilkan kalimat yang bermakna.

> 📄 **Tugas Akhir** — Ahmad Alfaruqi Haqinullah

</div>

---

## 📌 Tentang Proyek

**SIBI Sign Language Recognition** adalah sistem cerdas yang mampu mendeteksi dan menerjemahkan gerakan bahasa isyarat SIBI secara **real-time** menggunakan webcam. Sistem ini memanfaatkan kekuatan **MediaPipe** untuk ekstraksi landmark tubuh dan tangan, kemudian mengklasifikasikannya menggunakan **LSTM (Long Short-Term Memory)** Neural Network.

Hasil prediksi berupa kata-kata kemudian dirangkai menjadi kalimat yang natural dan bermakna menggunakan **Ollama + DeepSeek**, lalu disuarakan melalui **Text-to-Speech (gTTS)**.

---

## ✨ Fitur Unggulan

| Fitur | Deskripsi |
|-------|-----------|
| 🎯 **Real-time Detection** | Deteksi isyarat langsung dari webcam dengan latensi rendah |
| 🧠 **LSTM Neural Network** | Model sekuensial yang memahami urutan gerakan (30 frame/sampel) |
| 🤖 **AI Sentence Generator** | Ollama + DeepSeek mengubah urutan kata menjadi kalimat bermakna |
| 🗣️ **Text-to-Speech** | Konversi otomatis hasil prediksi ke audio menggunakan gTTS |
| 🎨 **GUI Interaktif** | Antarmuka pengguna modern berbasis PyQt5 / PyQt6 |
| 📹 **Dataset Recorder** | Tools lengkap untuk merekam dan membuat dataset SIBI baru |
| 📊 **High Confidence** | Threshold kepercayaan tinggi (95%) untuk prediksi akurat |
| ⚡ **Multi-threading** | Performa optimal dengan pemrosesan paralel |

---

## 🏗️ Arsitektur Sistem

```
┌──────────────┐    ┌──────────────────────┐    ┌──────────────────┐
│  Webcam      │───▶│  MediaPipe           │───▶│  Landmark        │
│  (Input)     │    │  Pose + Hand Track   │    │  Extraction      │
└──────────────┘    └──────────────────────┘    └────────┬─────────┘
                                                         │
                                              1662 features / frame
                                                         │
                                                         ▼
┌──────────────┐    ┌──────────────────────┐    ┌──────────────────┐
│  🔊 gTTS     │◀───│  Ollama + DeepSeek   │◀───│  LSTM Model      │
│  Audio Play  │    │  Sentence Generation │    │  (10 Kata)       │
└──────────────┘    └──────────────────────┘    └──────────────────┘
```

**Flow Data:**
```
Webcam → MediaPipe → Landmarks (1662 fitur) → Sequence (30 frame) → LSTM → Kata → AI → Kalimat → TTS
```

---

## 📁 Struktur Repository

```
SIBI-SIgnLanguage-TA/
│
├── 📂 SIBI-RecordVideo/          # 🎥 Tools untuk merekam dataset
│   ├── main.py                   # ▶ Entry point — jalankan ini!
│   ├── gui.py                    # Antarmuka grafis perekaman
│   ├── pose.py                   # Deteksi pose & tangan (MediaPipe)
│   ├── recorder.py               # Modul perekaman & penyimpanan video
│   ├── utils.py                  # Fungsi utilitas & helper
│   └── LOG/                      # Log aktivitas aplikasi
│
├── 📂 SIBI-TrainSignLanguage/    # 🧠 Training & evaluasi model
│   └── sign-language-LSTM.ipynb  # Jupyter notebook full pipeline
│
└── 📂 SIBI-App/                  # 🚀 Aplikasi pengenalan bahasa isyarat
    ├── gui.py                    # Antarmuka utama aplikasi
    └── sign_v5_10kata.keras      # Model LSTM terlatih (10 kata)
```

---

## 💻 Persyaratan Sistem

### Hardware
- **CPU**: Intel Core i5 / AMD Ryzen 5 atau lebih tinggi
- **RAM**: Minimum 8 GB (rekomendasi 16 GB)
- **GPU**: Opsional — CUDA-compatible untuk training lebih cepat
- **Kamera**: Webcam minimal 720p (30fps)

### Software
- **Python**: 3.8 atau lebih tinggi
- **Conda**: Untuk manajemen environment (rekomendasi Anaconda/Miniconda)
- **Ollama**: Untuk fitur AI sentence generation (opsional)

---

## 🚀 Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/madeeryu/SIBI-SIgnLanguage-TA.git
cd SIBI-SIgnLanguage-TA
```

### 2. Buat Conda Environment

```bash
# Untuk SIBI-App dan SIBI-TrainSignLanguage
conda create -n sibi-main python=3.10
conda activate sibi-main

# Untuk SIBI-RecordVideo
conda create -n sibi-record python=3.10
conda activate sibi-record
```

### 3. Install Dependencies

**Environment `sibi-main` (App + Training):**
```bash
pip install tensorflow==2.15.0 keras==2.15.0
pip install opencv-python==4.10.0.84 opencv-contrib-python==4.11.0.86
pip install mediapipe==0.10.9
pip install numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.2
pip install PyQt6==6.10.0 PySide6==6.10.0
pip install gtts pygame pyttsx3
pip install jupyterlab notebook
```

**Environment `sibi-record` (Recording Tools):**
```bash
pip install opencv-python==4.8.1.78 opencv-contrib-python==4.11.0.86
pip install mediapipe==0.10.9
pip install PyQt5==5.15.10
pip install moviepy==2.2.1
pip install numpy pandas sounddevice
```

### 4. Install Ollama (Opsional — Fitur AI)

```bash
# Kunjungi https://ollama.com untuk download installer
# Setelah terinstall, pull model DeepSeek:
ollama pull deepseek-r1:1.7b

# Jalankan Ollama server (di terminal terpisah):
ollama serve
```

---

## 🎮 Cara Penggunaan

### 🎥 1. Merekam Dataset Baru

Gunakan modul **SIBI-RecordVideo** untuk membuat dataset gerakan isyarat baru.

```bash
conda activate sibi-record
cd SIBI-RecordVideo
python main.py
```

**Langkah-langkah dalam aplikasi:**
1. Masukkan nama kata yang ingin direkam (misal: `halo`)
2. Pilih jumlah repetisi (rekomendasi: 30 video per kata)
3. Posisikan diri di depan kamera
4. Tekan tombol **"Mulai Rekam"** dan lakukan gerakan isyarat
5. Dataset otomatis tersimpan dalam format landmark `.npy`

> 💡 **Tips**: Pastikan pencahayaan cukup dan seluruh tangan terlihat oleh kamera.

---

### 🧠 2. Training Model LSTM

Latih model dengan dataset yang telah dikumpulkan menggunakan notebook Jupyter.

```bash
conda activate sibi-main
cd SIBI-TrainSignLanguage
jupyter notebook sign-language-LSTM.ipynb
```

**Pipeline Training:**
1. **Load Data** — Muat landmark `.npy` dari hasil perekaman
2. **Preprocessing** — Normalisasi & reshape data menjadi `(n_samples, 30, 1662)`
3. **Split Data** — Bagi data train/validation (80:20)
4. **Build Model** — Definisikan arsitektur LSTM
5. **Training** — Latih model (rekomendasi: 50 epochs)
6. **Evaluasi** — Lihat confusion matrix & accuracy
7. **Save Model** — Ekspor sebagai `.keras` file

---

### 🚀 3. Menjalankan Aplikasi Utama

Jalankan **SIBI-App** untuk memulai pengenalan bahasa isyarat secara real-time.

```bash
conda activate sibi-main
cd SIBI-App
python gui.py
```

**Cara pakai:**
1. Aplikasi membuka webcam secara otomatis
2. Lakukan gerakan isyarat SIBI di depan kamera
3. Sistem mendeteksi & menampilkan prediksi kata secara real-time
4. Kumpulkan beberapa kata → tekan tombol **"Buat Kalimat"**
5. AI (DeepSeek) merangkai kata menjadi kalimat lengkap
6. Kalimat disuarakan otomatis melalui speaker

---

## 📊 Dataset & Model

### Spesifikasi Dataset

| Parameter | Nilai |
|-----------|-------|
| **Pose Landmarks** | 33 titik × 4 koordinat (x, y, z, visibility) |
| **Hand Landmarks** | 21 titik × 3 koordinat × 2 tangan |
| **Total Fitur/Frame** | **1662 fitur** |
| **Sequence Length** | **30 frame** per sampel |
| **Format Data** | `(n_samples, 30, 1662)` |

### Arsitektur LSTM

```
Input Shape: (30, 1662)
     │
     ▼
┌─────────────────────────────────┐
│ LSTM Layer 1 │ 64 units │ return_sequences=True  │
├─────────────────────────────────┤
│ LSTM Layer 2 │ 128 units│ return_sequences=True  │
├─────────────────────────────────┤
│ LSTM Layer 3 │ 64 units │ return_sequences=False │
├─────────────────────────────────┤
│ Dense Layer 1│ 64 units │ ReLU                   │
├─────────────────────────────────┤
│ Dense Layer 2│ 32 units │ ReLU                   │
├─────────────────────────────────┤
│ Output Layer │ 10 units │ Softmax                │
└─────────────────────────────────┘

Optimizer : Adam
Loss      : Categorical Crossentropy
Metric    : Categorical Accuracy
```

---

## 🗣️ Kata yang Didukung (10 Kata)

| # | Kata Isyarat | Arti |
|---|--------------|------|
| 1 | **saya** | I / Me |
| 2 | **kamu** | You |
| 3 | **mau** | Want |
| 4 | **makan** | Eat |
| 5 | **diam** | Silent / Quiet |
| 6 | **siapa** | Who |
| 7 | **tolong** | Please / Help |
| 8 | **apa** | What |
| 9 | **kenalkan** | Introduce |
| 10 | **nama** | Name |

---

## 🤖 Integrasi AI (Ollama + DeepSeek)

Sistem menggunakan **Ollama** sebagai platform inferensi AI lokal dengan model **DeepSeek-R1 1.7B** untuk menghasilkan kalimat yang grammatikal dan bermakna dari urutan kata isyarat.

**Contoh alur:**
```
Isyarat     : [saya] [mau] [makan]
         ↓
LSTM Output : ["saya", "mau", "makan"]
         ↓
DeepSeek AI : "Saya mau makan."
         ↓
gTTS + Pygame → 🔊 Audio diputar
```

**Kemampuan AI:**
- ✅ Sentence Generation dari urutan kata
- ✅ Grammar & konteks bahasa Indonesia
- ✅ Berjalan **lokal** tanpa koneksi internet

---

## 🛠️ Teknologi yang Digunakan

| Teknologi | Versi | Fungsi |
|-----------|-------|--------|
| **Python** | 3.10 | Bahasa pemrograman utama |
| **TensorFlow / Keras** | 2.15 | Framework deep learning |
| **MediaPipe** | 0.10.9 | Deteksi pose & landmark tangan |
| **OpenCV** | 4.10 / 4.11 | Pemrosesan video & kamera |
| **NumPy** | 1.26.4 / 2.x | Komputasi numerik |
| **PyQt5 / PyQt6** | 5.15 / 6.10 | GUI Framework |
| **gTTS** | 2.5.4 | Text-to-Speech (Google) |
| **Pygame** | 2.6.1 | Pemutaran audio |
| **Ollama + DeepSeek** | r1:1.7b | AI sentence generation lokal |
| **MoviePy** | 2.2.1 | Pemrosesan video dataset |
| **Conda** | Latest | Manajemen environment |

---

## 📦 Environment Summary

| Environment | Digunakan Untuk | Package Utama |
|-------------|-----------------|---------------|
| `sibi-main` | App + Training | TensorFlow, Keras, PyQt6, gTTS |
| `sibi-record` | Dataset Recording | PyQt5, MoviePy, MediaPipe |

---

## 🤝 Kontribusi

Kontribusi sangat diterima! Ikuti langkah berikut:

```bash
# 1. Fork repository ini
# 2. Buat branch fitur baru
git checkout -b feature/fitur-baru

# 3. Commit perubahan
git commit -m "feat: menambahkan fitur baru"

# 4. Push ke branch
git push origin feature/fitur-baru

# 5. Buat Pull Request di GitHub
```

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE). Bebas digunakan untuk keperluan akademik dan non-komersial.

---

## 👤 Kontak

**Ahmad Alfaruqi Haqinullah**

[![GitHub](https://img.shields.io/badge/GitHub-@madeeryu-181717?style=flat-square&logo=github)](https://github.com/madeeryu)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=flat-square&logo=gmail)](mailto:email@example.com)

---

## 🙏 Ucapan Terima Kasih

- [MediaPipe](https://mediapipe.dev/) — Solusi deteksi pose & tangan yang luar biasa
- [TensorFlow](https://tensorflow.org/) — Framework deep learning yang powerful
- [Ollama](https://ollama.com/) — Platform AI inferensi lokal yang mudah digunakan
- [DeepSeek](https://www.deepseek.com/) — Model bahasa yang efisien dan cerdas

---

## 📚 Referensi

1. [MediaPipe Documentation](https://mediapipe.dev/)
2. [TensorFlow — Keras RNN Guide](https://www.tensorflow.org/guide/keras/rnn)
3. [SIBI — Sistem Isyarat Bahasa Indonesia](https://kbbi.kemdikbud.go.id/)
4. [Ollama Documentation](https://ollama.com/docs)
5. [DeepSeek R1 Model](https://ollama.com/library/deepseek-r1)

---

<div align="center">

**[⬆ Kembali ke Atas](#-sibi-sign-language-recognition)**

Made with ❤️ and 🤟 in Indonesia

</div>
