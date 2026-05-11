# 🌤️ ANALISIS KOMPARATIF 5 ALGORITMA MACHINE LEARNING UNTUK PREDIKSI SUHU MAKSIMUM DI SEATTLE

**UTS Praktikum Kecerdasan Buatan – Semester 4**

---

## 📋 Informasi Proyek

| Keterangan | Detail |
|------------|--------|
| **Nama** | Fahriansyah Putra |
| **NIM** | 301240024 |
| **Program Studi** | Teknik Informatika |
| **Fakultas** | Fakultas Teknologi Informasi |
| **Universitas** | Universitas Bale Bandung |
| **Mata Kuliah** | Praktikum Kecerdasan Buatan |
| **Semester** | 4 (Genap) |
| **Tahun Akademik** | 2025/2026 |

---

## 📝 Deskripsi Proyek

Proyek ini bertujuan untuk melakukan **analisis komparatif lima algoritma machine learning** dalam memprediksi suhu maksimum (TMAX) harian di Seattle, Amerika Serikat. Dataset yang digunakan adalah **Seattle Weather 1948–2017** dari Kaggle.

Lima algoritma yang diimplementasikan:
1. **Linear Regression**
2. **Artificial Neural Network (ANN)**
3. **Long Short-Term Memory (LSTM)**
4. **K-Means Clustering**
5. **Backpropagation Model (Custom Training Loop)**

Hasil evaluasi model diintegrasikan ke dalam **aplikasi web** yang dapat diakses publik untuk prediksi suhu secara real-time.

---

## 🔗 Link Penting

| Item | URL |
|------|-----|
| 🌐 **Aplikasi Web (Deploy)** | [https://huggingface.co/spaces/Fahputr/predik-cuaca-bandara-seattle] |
| 🎥 **Video Demo YouTube** | [https://youtu.be/NKCmE1XOF4g] |

---

## 📊 Hasil Evaluasi Model

| Model | MAE (°F) | RMSE (°F) | R² |
|-------|----------|-----------|-----|
| Linear Regression | 3.44 | 4.37 | 0.887 |
| ANN | 3.47 | 4.36 | 0.888 |
| LSTM | 4.07 | 5.22 | 0.839 |
| **Backpropagation** | **3.19** | **4.08** | **0.902** |

🏆 **Model Terbaik:** Backpropagation Model (R² = 0.902)

🔵 **K-Means Clustering:** Optimal k = 3, Silhouette Score = 0.314

---

## 🛠️ Teknologi yang Digunakan

- **Bahasa Pemrograman:** Python 3.10+
- **Library ML:** Scikit-learn, TensorFlow, Keras
- **Data Analysis:** Pandas, NumPy
- **Visualisasi:** Matplotlib, Seaborn
- **Web Framework:** Flask
- **Frontend:** Bootstrap 5, Chart.js
- **Deployment:** Railway, Gunicorn

---

## 🚀 Cara Menjalankan Aplikasi Secara Lokal

### 1. Clone Repository
```bash
git clone https://github.com/yuuji005/Prediksi_Cuaca_Bandara_seattle
cd [repo-name]