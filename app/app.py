# ── FILE: app/app.py ──────────────────────────────────────
import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
from flask import Flask, render_template, request, jsonify

# ── Tambah path agar bisa import dari root ────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── TensorFlow / Keras ────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF warning
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ──────────────────────────────────────────────────────────
# LOAD SEMUA MODEL & SCALER (saat server start)
# ──────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
DOCS_DIR    = os.path.join(BASE_DIR, 'docs')

def load_all_models():
    models = {}
    try:
        models['scaler_X']  = joblib.load(os.path.join(MODELS_DIR, 'scaler_X.pkl'))
        models['scaler_y']  = joblib.load(os.path.join(MODELS_DIR, 'scaler_y.pkl'))
        models['scaler_cluster'] = joblib.load(os.path.join(MODELS_DIR, 'scaler_cluster.pkl'))
        models['lr']        = joblib.load(os.path.join(MODELS_DIR, 'linear_regression.pkl'))
        models['ann']       = load_model(os.path.join(MODELS_DIR, 'ann_model.keras'))
        models['lstm']      = load_model(os.path.join(MODELS_DIR, 'lstm_model.keras'))
        models['kmeans']    = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
        models['label_map'] = joblib.load(os.path.join(MODELS_DIR, 'kmeans_label_map.pkl'))
        models['cluster_features'] = joblib.load(os.path.join(MODELS_DIR, 'cluster_features.pkl'))
        models['seq_len']   = joblib.load(os.path.join(MODELS_DIR, 'seq_len.pkl'))
        models['bp']        = joblib.load(os.path.join(MODELS_DIR, 'backprop_model.pkl'))
        models['k_optimal'] = joblib.load(os.path.join(MODELS_DIR, 'k_optimal.pkl'))
        print("✅ Semua model berhasil dimuat")
    except Exception as e:
        print(f"❌ Error memuat model: {e}")
    return models

M = load_all_models()

# ──────────────────────────────────────────────────────────
# LOAD BACKPROP CLASS (harus didefinisikan ulang untuk load)
# ──────────────────────────────────────────────────────────
class NeuralNetworkManual:
    def __init__(self, layer_sizes, learning_rate=0.001,
                 lr_decay=0.995, seed=42):
        self.layer_sizes = layer_sizes
        self.lr          = learning_rate
        self.lr_init     = learning_rate
        self.lr_decay    = lr_decay
        self.weights     = []
        self.biases      = []

    def leaky_relu(self, z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)

    def forward(self, X):
        self.A = [X]
        self.Z = []
        current = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ w + b
            self.Z.append(z)
            if i == len(self.weights) - 1:
                a = z
            else:
                a = self.leaky_relu(z)
            self.A.append(a)
            current = a
        return current

    def predict(self, X):
        return self.forward(X).ravel()

    @staticmethod
    def load(path):
        payload = joblib.load(path)
        model   = NeuralNetworkManual(
            payload['layer_sizes'],
            payload['learning_rate'],
            payload.get('lr_decay', 0.995)
        )
        model.weights = payload['weights']
        model.biases  = payload['biases']
        return model

# Load ulang backprop dengan class yang sudah didefinisikan
M['bp'] = NeuralNetworkManual.load(os.path.join(MODELS_DIR, 'backprop_model.pkl'))

# ──────────────────────────────────────────────────────────
# HELPER: FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────
def compute_cluster_features(close_history, rsi, volatility, price_range):
    close_arr = np.array(close_history, dtype=float)

    # return_5d, return_10d, return_20d
    def safe_return(arr, n):
        if len(arr) > n:
            return (arr[-1] - arr[-n-1]) / arr[-n-1] if arr[-n-1] != 0 else 0.0
        return 0.0

    return_5d  = safe_return(close_arr, 5)
    return_10d = safe_return(close_arr, 10)
    return_20d = safe_return(close_arr, 20)

    # vol_ratio: volatility 5 hari / volatility 20 hari
    vol5  = np.std(close_arr[-5:])  if len(close_arr) >= 5  else 0.0
    vol20 = np.std(close_arr[-20:]) if len(close_arr) >= 20 else 1e-10
    vol_ratio = vol5 / vol20 if vol20 != 0 else 0.0

    # range_ratio: price_range / close terakhir
    range_ratio = price_range / close_arr[-1] if close_arr[-1] != 0 else 0.0

    # ma_gap_5_20: (MA5 - MA20) / MA20
    ma5  = np.mean(close_arr[-5:])  if len(close_arr) >= 5  else close_arr[-1]
    ma20 = np.mean(close_arr[-20:]) if len(close_arr) >= 20 else close_arr[-1]
    ma_gap_5_20 = (ma5 - ma20) / ma20 if ma20 != 0 else 0.0

    # Urutan HARUS sama persis dengan CLUSTER_FEATURES di training
    return np.array([
        rsi,          # RSI
        return_5d,    # return_5d
        return_10d,   # return_10d
        return_20d,   # return_20d
        vol_ratio,    # vol_ratio
        range_ratio,  # range_ratio
        ma_gap_5_20   # ma_gap_5_20
    ], dtype=float)

    return features


# app.py — letakkan di bagian atas, sebelum @app.route

import numpy as np

def compute_features(open_, high, low, close_series, volume):
    close_arr = np.array(close_series, dtype=float)
    close_now = close_arr[-1]

    lag1 = close_arr[-2] if len(close_arr) >= 2 else close_now
    lag2 = close_arr[-3] if len(close_arr) >= 3 else close_now
    lag3 = close_arr[-4] if len(close_arr) >= 4 else close_now
    lag5 = close_arr[-6] if len(close_arr) >= 6 else close_now

    ma5  = np.mean(close_arr[-5:])  if len(close_arr) >= 5  else close_now
    ma10 = np.mean(close_arr[-10:]) if len(close_arr) >= 10 else close_now
    ma20 = np.mean(close_arr[-20:]) if len(close_arr) >= 20 else close_now

    volatility  = np.std(close_arr[-10:]) if len(close_arr) >= 10 else 0.0
    price_range = float(high) - float(low)
    daily_return = (close_now - lag1) / lag1 if lag1 != 0 else 0.0

    if len(close_arr) >= 15:
        deltas = np.diff(close_arr[-15:])
        gain = np.mean(deltas[deltas > 0]) if any(deltas > 0) else 0
        loss = np.mean(-deltas[deltas < 0]) if any(deltas < 0) else 1e-10
        rsi  = 100 - (100 / (1 + gain / loss))
    else:
        rsi = 50.0

    return np.array([
        float(open_), float(high), float(low), float(volume),
        lag1, lag2, lag3, lag5,
        ma5, ma10, ma20,
        volatility, price_range, daily_return, rsi
    ], dtype=float)


def compute_features(open_, high, low, close_series, volume):
    close_arr = np.array(close_series, dtype=float)
    close_now = close_arr[-1]

    lag1 = close_arr[-2] if len(close_arr) >= 2 else close_now
    lag2 = close_arr[-3] if len(close_arr) >= 3 else close_now
    lag3 = close_arr[-4] if len(close_arr) >= 4 else close_now
    lag5 = close_arr[-6] if len(close_arr) >= 6 else close_now

    ma5  = np.mean(close_arr[-5:])  if len(close_arr) >= 5  else close_now
    ma10 = np.mean(close_arr[-10:]) if len(close_arr) >= 10 else close_now
    ma20 = np.mean(close_arr[-20:]) if len(close_arr) >= 20 else close_now

    volatility   = np.std(close_arr[-10:], ddof=1) if len(close_arr) >= 10 else 0.0
    price_range  = float(high) - float(low)
    daily_return = (close_now - lag1) / lag1 if lag1 != 0 else 0.0

    # ✅ RSI versi Wilder's (sama dengan rolling(14).mean() di training)
    if len(close_arr) >= 15:
        delta = np.diff(close_arr[-15:])          # 14 perubahan
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        avg_gain = gain.mean()                     # simple mean of 14 periods
        avg_loss = loss.mean()
        rs  = avg_gain / avg_loss if avg_loss != 0 else 1e10
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50.0

    return np.array([
        float(open_), float(high), float(low), float(volume),
        lag1, lag2, lag3, lag5,
        ma5, ma10, ma20,
        volatility, price_range, daily_return, rsi
    ], dtype=float)


# ──────────────────────────────────────────────────────────
# LOAD DATA HASIL MODELING (untuk halaman comparison)
# ──────────────────────────────────────────────────────────
def load_comparison_data():
    csv_path = os.path.join(DOCS_DIR, 'model_comparison.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path).to_dict(orient='records')
    # Fallback jika CSV tidak ada
    return []


# ──────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────

# ── Halaman Beranda ───────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


# ── Halaman Prediksi ──────────────────────────────────────
@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')


# ── API Prediksi (POST) ───────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # ── Validasi input ────────────────────────────────
        required = ['open', 'high', 'low', 'close_history', 'volume']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Field {field} wajib diisi'}), 400

        open_    = float(data['open'])
        high     = float(data['high'])
        low      = float(data['low'])
        volume   = float(data['volume'])
        close_history = [float(x) for x in data['close_history']]

        if len(close_history) < 20:
            return jsonify({
                'error': 'Minimal 20 data harga historis diperlukan'
            }), 400

        if high < low:
            return jsonify({'error': 'High tidak boleh lebih kecil dari Low'}), 400

        # ── Feature engineering ───────────────────────────
        features = compute_features(
            open_, high, low, close_history, volume
        )
        features_sc = M['scaler_X'].transform(features.reshape(1, -1))

        results = {}

        # ── 1. Linear Regression ──────────────────────────
        pred_lr = float(M['lr'].predict(features_sc)[0])
        results['linear_regression'] = round(pred_lr, 4)
                        
        # ── 2. ANN ────────────────────────────────────────
        pred_ann_sc = M['ann'].predict(features_sc, verbose=0).ravel()
        pred_ann    = M['scaler_y'].inverse_transform(
                  pred_ann_sc.reshape(-1, 1)
              ).ravel()[0]
        results['ann'] = round(float(pred_ann), 4)

        # ── 3. LSTM ───────────────────────────────────────
        seq_len = M['seq_len']
        close_arr = np.array(close_history, dtype=float)

        if len(close_arr) >= seq_len:
            # Buat sequence dari SEQ_LEN hari terakhir
            recent = close_arr[-seq_len:]
            # Buat fitur untuk setiap hari dalam sequence
            seq_features = []
            for i in range(seq_len):
                idx   = len(close_arr) - seq_len + i
                hist  = close_arr[max(0, idx-20):idx+1].tolist()
                if len(hist) < 2:
                    hist = [close_arr[0]] * 2 + hist
                f = compute_features(
                    close_arr[idx], close_arr[idx],
                    close_arr[idx], hist, volume
                )
                seq_features.append(f)

            seq_arr    = np.array(seq_features)
            seq_sc     = M['scaler_X'].transform(seq_arr)
            seq_input  = seq_sc.reshape(1, seq_len, -1)

            pred_lstm_sc = M['lstm'].predict(seq_input, verbose=0).ravel()
            pred_lstm    = M['scaler_y'].inverse_transform(
                               pred_lstm_sc.reshape(-1, 1)
                           ).ravel()[0]
            results['lstm'] = round(float(pred_lstm), 4)
        else:
            results['lstm'] = None
            results['lstm_note'] = f'Butuh minimal {seq_len} data historis'

        # ── 4. Backpropagation ────────────────────────────
        pred_bp_sc = M['bp'].predict(features_sc)
        pred_bp    = M['scaler_y'].inverse_transform(
                         pred_bp_sc.reshape(-1, 1)
                     ).ravel()[0]
        results['backpropagation'] = round(float(pred_bp), 4)

        # ── 5. K-Means: klasifikasi kondisi pasar ─────────
        feat_raw = compute_features(
            open_, high, low, close_history, volume
        )
        # Ambil RSI, volatility, price_range dari feat_raw
        rsi        = feat_raw[14]
        volatility = feat_raw[11]
        price_range = feat_raw[12]

        cluster_feat = compute_cluster_features(
            close_history, rsi, volatility, price_range
        )
        cluster_sc   = M['scaler_cluster'].transform(
                           cluster_feat.reshape(1, -1)
                       )
        cluster_id   = int(M['kmeans'].predict(cluster_sc)[0])
        cluster_label = M['label_map'].get(cluster_id, 'Unknown')
        results['kmeans'] = {
            'cluster_id'   : cluster_id,
            'market_condition': cluster_label
        }

        # ── Rata-rata prediksi (exclude LSTM jika None) ───
        numeric_preds = [
            v for k, v in results.items()
            if k not in ['kmeans', 'lstm_note']
            and v is not None
            and isinstance(v, (int, float))
        ]
        results['average_prediction'] = round(
            float(np.mean(numeric_preds)), 4
        )

        return jsonify({'success': True, 'predictions': results})

    except ValueError as ve:
        return jsonify({'error': f'Input tidak valid: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


# ── Halaman Perbandingan Model ────────────────────────────
@app.route('/comparison')
def comparison():
    comparison_data = load_comparison_data()
    return render_template('comparison.html',
                           comparison_data=comparison_data)


# ── Halaman Clustering ────────────────────────────────────
@app.route('/clustering')
def clustering():
    return render_template('clustering.html',
                           k_optimal=M.get('k_optimal', 3))


# ── API: Data untuk Chart.js ──────────────────────────────
@app.route('/api/comparison-data')
def api_comparison_data():
    data = load_comparison_data()
    return jsonify(data)


# ── Health check ──────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({
        'status' : 'ok',
        'models_loaded': list(M.keys())
    })


# ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)