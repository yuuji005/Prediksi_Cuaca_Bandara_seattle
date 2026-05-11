# train_models.py
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')

from utils import load_and_preprocess_data, evaluate_regression, plot_residuals, plot_comparison

# Create directories if not exist
os.makedirs('models', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load data
X_train, X_test, y_train, y_test, scaler, feature_cols, df = load_and_preprocess_data()

# 0. Scale Target (Y) for Neural Networks
from sklearn.preprocessing import StandardScaler
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
joblib.dump(scaler_y, 'models/scaler_y.pkl')
print("Target scaler (Y) disimpan.")

# Dictionary to store results
results = {}

# 1. Linear Regression
print("Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mae_lr, rmse_lr, r2_lr = evaluate_regression(y_test, y_pred_lr)
results['Linear Regression'] = {'MAE': mae_lr, 'RMSE': rmse_lr, 'R2': r2_lr}
joblib.dump(lr, 'models/linear_regression.pkl')
plot_residuals(y_test, y_pred_lr, 'Linear Regression', 'residuals_lr.png')
print(f"LR - MAE: {mae_lr:.2f}, RMSE: {rmse_lr:.2f}, R2: {r2_lr:.3f}")

# 2. Artificial Neural Network (ANN)
print("Training ANN...")
ann = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
ann.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = ann.fit(X_train, y_train_scaled, epochs=100, batch_size=32,
                  validation_split=0.15, callbacks=[early_stop], verbose=1)
y_pred_ann_scaled = ann.predict(X_test)
y_pred_ann = scaler_y.inverse_transform(y_pred_ann_scaled).flatten()
mae_ann, rmse_ann, r2_ann = evaluate_regression(y_test, y_pred_ann)
results['ANN'] = {'MAE': mae_ann, 'RMSE': rmse_ann, 'R2': r2_ann}
ann.save('models/ann_model.h5')
plot_residuals(y_test, y_pred_ann, 'ANN', 'residuals_ann.png')
print(f"ANN - MAE: {mae_ann:.2f}, RMSE: {rmse_ann:.2f}, R2: {r2_ann:.3f}")

# 3. LSTM (Recurrent Neural Network)
print("Training LSTM...")
# Reshape data for LSTM: (samples, timesteps, features)
# We'll use 3 timesteps (lags) and relevant features
# Recreate sequences from original df (already in utils, but need to reshape)
# Let's create sequence data from the original time series
def create_sequences(data, seq_length=3):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(data[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

# We need to rebuild sequences from scaled data of raw features (TMAX, TMIN, PRCP, RAIN)
df_seq = df[['TMAX', 'TMIN', 'PRCP', 'RAIN']].values
scaler_seq = joblib.load('models/scaler.pkl')  # we'll use separate scaler for simplicity
# Fit a scaler specific to these 4 features
from sklearn.preprocessing import StandardScaler
scaler_seq = StandardScaler()
df_seq_scaled = scaler_seq.fit_transform(df_seq)
joblib.dump(scaler_seq, 'models/scaler_seq.pkl')
X_seq, y_seq = create_sequences(df_seq_scaled, seq_length=3)
y_seq_raw = df['TMAX'].values[3:]
y_seq_scaled = scaler_y.transform(y_seq_raw.reshape(-1, 1)).flatten()

split_idx_seq = int(0.8 * len(X_seq))
X_train_seq, X_test_seq = X_seq[:split_idx_seq], X_seq[split_idx_seq:]
y_train_seq, y_test_seq = y_seq_scaled[:split_idx_seq], y_seq_scaled[split_idx_seq:]
y_test_seq_raw = y_seq_raw[split_idx_seq:]

# Build LSTM
lstm_model = models.Sequential([
    layers.LSTM(64, activation='tanh', return_sequences=True, input_shape=(3, 4)),
    layers.Dropout(0.2),
    layers.LSTM(32, activation='tanh'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])
lstm_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
history_lstm = lstm_model.fit(X_train_seq, y_train_seq, epochs=150, batch_size=32,
                              validation_split=0.15, callbacks=[early_stop], verbose=1)
y_pred_lstm_scaled = lstm_model.predict(X_test_seq)
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled).flatten()
mae_lstm, rmse_lstm, r2_lstm = evaluate_regression(y_test_seq_raw, y_pred_lstm)
results['LSTM'] = {'MAE': mae_lstm, 'RMSE': rmse_lstm, 'R2': r2_lstm}
lstm_model.save('models/lstm_model.h5')
plot_residuals(y_test_seq_raw, y_pred_lstm, 'LSTM', 'residuals_lstm.png')
print(f"LSTM - MAE: {mae_lstm:.2f}, RMSE: {rmse_lstm:.2f}, R2: {r2_lstm:.3f}")

# 4. K-Means Clustering
print("Training K-Means Clustering...")
# Use the same feature set as regression
kmeans_data = X_train  # scaled
inertias = []
silhouette_scores = []
K_range = range(2, 10)
best_score = -1
best_k = 2
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(kmeans_data)
    inertias.append(km.inertia_)
    if k > 1:
        score = silhouette_score(kmeans_data, labels)
        silhouette_scores.append(score)
        if score > best_score:
            best_score = score
            best_k = k
# Train final KMeans with best_k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans.fit(X_train)
joblib.dump(kmeans, 'models/kmeans.pkl')
# Store results (inertia and silhouette)
results['K-Means'] = {'Inertia': kmeans.inertia_, 'Silhouette': best_score, 'Best K': best_k}
print(f"K-Means - Best K: {best_k}, Inertia: {kmeans.inertia_:.2f}, Silhouette: {best_score:.3f}")

# 5. Backpropagation Model (Same architecture as ANN for consistency)
print("Training Backpropagation Model...")
bp_model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
bp_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history_bp = bp_model.fit(X_train, y_train_scaled, epochs=100, batch_size=32,
                          validation_split=0.15, callbacks=[early_stop], verbose=1)
y_pred_bp_scaled = bp_model.predict(X_test)
y_pred_bp = scaler_y.inverse_transform(y_pred_bp_scaled).flatten()
mae_bp, rmse_bp, r2_bp = evaluate_regression(y_test, y_pred_bp)
results['Backpropagation'] = {'MAE': mae_bp, 'RMSE': rmse_bp, 'R2': r2_bp}
bp_model.save('models/backprop_model.h5')
plot_residuals(y_test, y_pred_bp, 'Backpropagation', 'residuals_bp.png')
print(f"Backprop - MAE: {mae_bp:.2f}, RMSE: {rmse_bp:.2f}, R2: {r2_bp:.3f}")


# Save all results
import json
with open('models/results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Plot comparison graph
# For regression models only (exclude K-Means)
reg_results = {k: v for k, v in results.items() if 'MAE' in v}
plot_comparison(reg_results, 'comparison_bar.png')

print("\nAll models trained and saved.")
print("Results:", results)