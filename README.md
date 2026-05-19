# Stock Price Prediction — ML Comparative Analysis

Aplikasi prediksi harga saham berbasis Machine Learning menggunakan
5 algoritma AI untuk analisis komparatif.

## Identitas
- **Nama**  : Asep Sunandar
- **NIM**   : 301240039
- **Prodi** : Teknik Informatika
- **Dosen** : Mohammad Bayu Anggara S.Kom, M.Kom


## Deskripsi
Proyek ini melakukan analisis perbandingan 5 algoritma ML untuk
memprediksi harga penutupan saham AAPL menggunakan dataset
S&P 500 All Stocks 5 Years dari Kaggle.

## Algoritma yang Digunakan
| No | Algoritma | Library | Metrik |
|----|-----------|---------|--------|
| 1 | Linear Regression | scikit-learn | MAE, RMSE, R² |
| 2 | ANN (Artificial Neural Network) | TensorFlow/Keras | MAE, RMSE, R² |
| 3 | LSTM / RNN | TensorFlow/Keras | MAE, RMSE, MAPE |
| 4 | K-Means Clustering | scikit-learn | Inertia, Silhouette |
| 5 | Backpropagation Manual | NumPy | MAE, RMSE, R² |

## Struktur Folder
stock-prediction/
├── data/                   # Dataset
├── notebooks/              # Jupyter Notebook EDA & modeling
├── models/                 # Model tersimpan (.pkl, .keras)
├── app/                    # Aplikasi Flask
│   ├── static/             # CSS, JS, gambar
│   ├── templates/          # HTML templates
│   └── app.py              # File utama Flask
├── docs/                   # Grafik & laporan
├── requirements.txt
├── Procfile
└── README.md

## Cara Instalasi & Menjalankan

```bash
# Clone repository
git clone https://github.com/USERNAME/stock-prediction.git
cd stock-prediction

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
cd app
python app.py
```

Buka browser: `http://localhost:5000`

## Link Penting
- **Demo Aplikasi** : https://stockai.my.id
- **Laporan PDF**   : [link laporan]
- **Video YouTube** : [link video]

## Dataset
- **Nama**    : S&P 500 All Stocks 5 Years
- **Sumber**  : [Kaggle](https://www.kaggle.com/datasets/camnugent/sandp500)
- **Lisensi** : CC0 Public Domain
- **Ukuran**  : 619,040+ baris, 7 kolom