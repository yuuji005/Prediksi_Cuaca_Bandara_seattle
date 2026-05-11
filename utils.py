# utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_preprocess_data(filepath='seattleWeather_1948-2017.csv'):
    """
    Load dataset, filter to last 20 years, clean anomalies, create lags, and split.
    """
    df = pd.read_csv(filepath)
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # 1. Gunakan data penuh (1948 - 2017)
    df = df.sort_values('DATE').reset_index(drop=True)
    
    # 2. Cleaning: Hapus data anomali (Suhu Max tidak boleh lebih kecil dari Min)
    df = df.dropna(subset=['TMAX', 'TMIN', 'PRCP'])
    df = df[df['TMAX'] >= df['TMIN']]
    
    # Convert RAIN to binary
    df['RAIN'] = df['RAIN'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
    df['RAIN'] = df['RAIN'].fillna(0)
    
    # 3. Create lag features
    for lag in [1, 2, 3]:
        df[f'TMAX_lag{lag}'] = df['TMAX'].shift(lag)
        df[f'TMIN_lag{lag}'] = df['TMIN'].shift(lag)
        df[f'PRCP_lag{lag}'] = df['PRCP'].shift(lag)
        df[f'RAIN_lag{lag}'] = df['RAIN'].shift(lag)
    
    # Drop rows with NaN created by lags
    df = df.dropna()
    
    # Simpan dataset yang sudah dibersihkan untuk digunakan oleh app.py
    df.to_csv('preprocessed_seattleWeather.csv', index=False)
    print(f"Dataset dibersihkan. Jumlah data sekarang: {len(df)} baris (1998-2017).")
    
    # Feature columns
    feature_cols = ['TMAX_lag1', 'TMAX_lag2', 'TMAX_lag3',
                    'TMIN_lag1', 'TMIN_lag2', 'TMIN_lag3',
                    'PRCP_lag1', 'PRCP_lag2', 'PRCP_lag3',
                    'RAIN_lag1', 'RAIN_lag2', 'RAIN_lag3',
                    'TMIN', 'PRCP', 'RAIN']
    
    X = df[feature_cols].values
    y = df['TMAX'].values
    
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols, df

def evaluate_regression(y_true, y_pred):
    """Calculate MAE, RMSE, R2"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def plot_residuals(y_true, y_pred, title, filename):
    """Plot residuals"""
    residuals = y_true - y_pred
    plt.figure(figsize=(8,5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted TMAX')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot - {title}')
    plt.tight_layout()
    plt.savefig(f'static/{filename}')
    plt.close()

def plot_comparison(results_dict, filename):
    """Plot comparison bar chart of metrics"""
    models = list(results_dict.keys())
    mae_vals = [v['MAE'] for v in results_dict.values()]
    rmse_vals = [v['RMSE'] for v in results_dict.values()]
    r2_vals = [v['R2'] for v in results_dict.values()]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.bar(x - width, mae_vals, width, label='MAE', color='#1f77b4')
    ax1.bar(x, rmse_vals, width, label='RMSE', color='#ff7f0e')
    ax1.set_ylabel('Error (°F)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    
    ax2 = ax1.twinx()
    ax2.bar(x + width, r2_vals, width, label='R²', color='#2ca02c')
    ax2.set_ylabel('R² Score')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    fig.tight_layout()
    plt.savefig(f'static/{filename}')
    plt.close()