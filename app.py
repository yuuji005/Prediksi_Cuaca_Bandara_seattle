from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import pickle
import json
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Konfigurasi path
MODELS_DIR = 'models'

# Load Historical Data for Lags
try:
    df_hist = pd.read_csv('preprocessed_seattleWeather.csv')
    df_hist['DATE'] = pd.to_datetime(df_hist['DATE'])
    df_hist = df_hist.sort_values('DATE').reset_index(drop=True)
    # Ensure numeric columns
    for col in ['TMAX', 'TMIN', 'PRCP', 'RAIN']:
        df_hist[col] = pd.to_numeric(df_hist[col], errors='coerce')
    df_hist = df_hist.dropna(subset=['TMAX', 'TMIN', 'PRCP', 'RAIN'])
    print("Historical data berhasil dimuat dan diproses.")
except Exception as e:
    print("Error loading historical data:", e)
    df_hist = None

# Load Results/Metrics
results_data = {}
try:
    with open(os.path.join(MODELS_DIR, 'results.json'), 'r') as f:
        results_data = json.load(f)
    print("Metrics results berhasil dimuat.")
except Exception as e:
    print("Error loading results.json:", e)

# Load Scalers
try:
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    scaler_y = joblib.load(os.path.join(MODELS_DIR, 'scaler_y.pkl'))
    scaler_seq = joblib.load(os.path.join(MODELS_DIR, 'scaler_seq.pkl'))
    print("Scalers berhasil dimuat.")
except Exception as e:
    print("Error loading scalers:", e)

# Load Models
def safe_load(name, type='keras'):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        print(f"File {name} tidak ditemukan!")
        return None
    try:
        if type == 'keras':
            model = load_model(path, compile=False)
            print(f"Model {name} (Keras) berhasil dimuat.")
            return model
        else:
            model = joblib.load(path)
            print(f"Model {name} (Joblib) berhasil dimuat.")
            return model
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None

model_lr = safe_load('linear_regression.pkl', 'joblib')
model_ann = safe_load('ann_model.h5', 'keras')
model_lstm = safe_load('lstm_model.h5', 'keras')
model_kmeans = safe_load('kmeans.pkl', 'joblib')
model_bp = safe_load('backprop_model.h5', 'keras')


@app.route('/')
def index():
    return render_template('index.html', tab='prediksi', metrics=results_data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        tmin_c = float(request.form['tmin'])
        prcp_mm = float(request.form['prcp'])
        algo = request.form['algo']

        tmin_f = (tmin_c * 9.0 / 5.0) + 32
        prcp_in = prcp_mm / 25.4
        rain_in = 1.0 if prcp_in > 0 else 0.0
        is_sim = request.form.get('simulation') == 'on'

        if df_hist is None:
            raise Exception("Historical data (Lags) tidak tersedia.")

        # Default Lags from last 3 days of dataset
        tmax_lag1, tmax_lag2, tmax_lag3 = df_hist['TMAX'].iloc[-1], df_hist['TMAX'].iloc[-2], df_hist['TMAX'].iloc[-3]
        tmin_lag1, tmin_lag2, tmin_lag3 = df_hist['TMIN'].iloc[-1], df_hist['TMIN'].iloc[-2], df_hist['TMIN'].iloc[-3]
        prcp_lag1, prcp_lag2, prcp_lag3 = df_hist['PRCP'].iloc[-1], df_hist['PRCP'].iloc[-2], df_hist['PRCP'].iloc[-3]
        rain_lag1, rain_lag2, rain_lag3 = df_hist['RAIN'].iloc[-1], df_hist['RAIN'].iloc[-2], df_hist['RAIN'].iloc[-3]

        # Override if simulation mode is ON
        if is_sim:
            # Fake "July" weather lags (High temp, No rain)
            tmax_lag1, tmax_lag2, tmax_lag3 = 85.0, 82.0, 80.0
            tmin_lag1, tmin_lag2, tmin_lag3 = 65.0, 62.0, 60.0
            prcp_lag1, prcp_lag2, prcp_lag3 = 0.0, 0.0, 0.0
            rain_lag1, rain_lag2, rain_lag3 = 0.0, 0.0, 0.0

        print(f"DEBUG: Lag Features extracted for prediction (Sim={is_sim}):")
        print(f"TMAX Lags: {tmax_lag1}, {tmax_lag2}, {tmax_lag3}")
        print(f"TMIN Lags: {tmin_lag1}, {tmin_lag2}, {tmin_lag3}")

        input_data = np.array([[
            tmax_lag1, tmax_lag2, tmax_lag3,
            tmin_lag1, tmin_lag2, tmin_lag3,
            prcp_lag1, prcp_lag2, prcp_lag3,
            rain_lag1, rain_lag2, rain_lag3,
            tmin_f, prcp_in, rain_in
        ]])
        input_scaled = scaler.transform(input_data)
        
        if algo == 'lr':
            if model_lr is None: raise Exception("Model LR tidak tersedia.")
            pred_scaled = model_lr.predict(input_scaled)
            prediction_f = pred_scaled[0]
        elif algo == 'ann':
            if model_ann is None: raise Exception("Model ANN tidak tersedia.")
            pred_scaled = model_ann.predict(input_scaled, verbose=0)
            prediction_f = scaler_y.inverse_transform(pred_scaled)[0][0]
        elif algo == 'lstm':
            if model_lstm is None: raise Exception("Model LSTM tidak tersedia.")
            # For LSTM we use the previous 3 days as sequence
            seq_data = df_hist[['TMAX', 'TMIN', 'PRCP', 'RAIN']].tail(3).values
            seq_scaled = scaler_seq.transform(seq_data)
            input_lstm = seq_scaled.reshape((1, 3, 4))
            pred_scaled = model_lstm.predict(input_lstm, verbose=0)
            prediction_f = scaler_y.inverse_transform(pred_scaled)[0][0]
        elif algo == 'bp':
            if model_bp is None: raise Exception("Model Backprop tidak tersedia.")
            pred_scaled = model_bp.predict(input_scaled)
            prediction_f = scaler_y.inverse_transform(pred_scaled)[0][0]
        else:
            raise Exception("Algoritma tidak valid.")
            
        prediction_c = (prediction_f - 32) * 5.0 / 9.0
        
        if prediction_c < 15:
            temp_color, temp_icon, temp_desc = "info", "bi-thermometer-snow", "Suhu Dingin"
        elif prediction_c < 28:
            temp_color, temp_icon, temp_desc = "success", "bi-thermometer-half", "Suhu Normal / Sejuk"
        else:
            temp_color, temp_icon, temp_desc = "danger", "bi-thermometer-sun", "Suhu Panas Ekstrim"
            
        # logic untuk label K-Means
        cluster_idx = model_kmeans.predict(input_scaled)[0]
        # Overide logic: Jika input hujan hari ini > 2mm, hampir pasti "Berawan/Hujan" di Seattle
        if prcp_mm > 2.0:
            cluster_name = "Berawan/Hujan"
        else:
            cluster_mapping = {0: "Kering/Cerah", 1: "Berawan/Hujan"}
            cluster_name = cluster_mapping.get(cluster_idx, "Tidak Diketahui")
        
        # Hitung Heat Index sederhana (Feels Like)
        # Di Seattle, faktor "terasa lebih dingin/panas" dipengaruhi hujan
        feels_like = prediction_c
        if prcp_mm > 5.0: feels_like -= 2.0  # Hujan membuat terasa lebih dingin
        elif prediction_c > 27: feels_like += 1.5 # Panas lembab
        
        # Logic bar_percent (Rentang -10 s/d 45 C)
        bar_percent = min(max(((prediction_c + 10) / 55) * 100, 0), 100)
        
        # Context info for explanation
        hist_context = {
            "last_date": "Mode Simulasi (Musim Panas)" if is_sim else df_hist['DATE'].iloc[-1].strftime('%d %b %Y'),
            "avg_tmax_lag": 82.3 if is_sim else round(df_hist['TMAX'].tail(3).mean(), 1),
            "is_rainy_lag": "Tidak" if is_sim else ("Ya" if df_hist['RAIN'].tail(3).sum() > 0 else "Tidak"),
            "is_sim": is_sim,
            "feels_like": round(feels_like, 1)
        }

        return render_template('index.html', 
                               result=round(prediction_f, 2), 
                               result_c=round(prediction_c, 2),
                               temp_color=temp_color,
                               temp_icon=temp_icon,
                               temp_desc=temp_desc,
                               bar_percent=round(bar_percent),
                               cluster=cluster_name,
                               metrics=results_data,
                               context=hist_context,
                               tmin=tmin_c, prcp=prcp_mm, algo=algo.upper())
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/comparison')
def comparison():
    return render_template('index.html', tab='komparasi', metrics=results_data)

@app.route('/about')
def about():
    return render_template('index.html', tab='tentang', metrics=results_data)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=True)
