"""
=======================================================
  Motor Temperature Prediction - Training Script
  Dataset: Electric Motor Temperature (Kaggle)
  https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature
=======================================================

HƯỚNG DẪN SỬ DỤNG:
1. Cài đặt thư viện:
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib

2. Tải dataset từ Kaggle về, đặt file pmsm_temperature_data.csv
   vào cùng thư mục với script này.

3. Chạy script:
   python train_motor_temperature.py

4. Kết quả: model được lưu vào thư mục ./models/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost không có sẵn. Sẽ dùng GradientBoosting thay thế.")

# ─────────────────────────────────────────────
#  CẤU HÌNH
# ─────────────────────────────────────────────
DATA_PATH = "measures_v2.csv"
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Các cột feature đầu vào
FEATURE_COLS = [
    "ambient",       # Nhiệt độ môi trường
    "coolant",       # Nhiệt độ chất làm mát
    "u_d",           # Điện áp trục d
    "u_q",           # Điện áp trục q
    "motor_speed",   # Tốc độ motor
    "torque",        # Mô-men xoắn
    "i_d",           # Dòng điện trục d
    "i_q",           # Dòng điện trục q
]

# Mục tiêu dự đoán (chọn pm = nhiệt độ nam châm vĩnh cửu là quan trọng nhất)
TARGET_COLS = [
    "pm",            # Nhiệt độ nam châm vĩnh cửu (quan trọng nhất - dễ quá nhiệt)
    "stator_yoke",   # Nhiệt độ stator yoke
    "stator_tooth",  # Nhiệt độ stator tooth
    "stator_winding",# Nhiệt độ cuộn dây stator
]

PRIMARY_TARGET = "pm"  # Target chính để predict

# Ngưỡng cảnh báo quá nhiệt (°C)
OVERHEAT_THRESHOLD = {
    "pm": 100,
    "stator_yoke": 120,
    "stator_tooth": 120,
    "stator_winding": 120,
}


# ─────────────────────────────────────────────
#  1. TẢI VÀ KHÁM PHÁ DỮ LIỆU
# ─────────────────────────────────────────────
def load_and_explore(path):
    print("\n" + "="*60)
    print("  📂 BƯỚC 1: TẢI DỮ LIỆU")
    print("="*60)

    df = pd.read_csv(path)
    print(f"✅ Đã tải dữ liệu: {df.shape[0]:,} dòng x {df.shape[1]} cột")
    print(f"\n📊 Các cột: {list(df.columns)}")
    print(f"\n📈 Thống kê mô tả:\n{df[FEATURE_COLS + TARGET_COLS].describe().round(2)}")
    print(f"\n🔍 Giá trị null:\n{df.isnull().sum()}")

    # Phân tích phân phối target
    print(f"\n🌡️  Phân tích nhiệt độ target:")
    for col in TARGET_COLS:
        if col in df.columns:
            mean_t = df[col].mean()
            max_t = df[col].max()
            thresh = OVERHEAT_THRESHOLD.get(col, 100)
            overheat_pct = (df[col] > thresh).mean() * 100
            print(f"  {col:18s}: mean={mean_t:.1f}°C, max={max_t:.1f}°C, "
                  f"quá ngưỡng {thresh}°C: {overheat_pct:.1f}%")

    return df


# ─────────────────────────────────────────────
#  2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def feature_engineering(df):
    print("\n" + "="*60)
    print("  ⚙️  BƯỚC 2: FEATURE ENGINEERING")
    print("="*60)

    # Công suất điện (P = u_d*i_d + u_q*i_q)
    df["electrical_power"] = df["u_d"] * df["i_d"] + df["u_q"] * df["i_q"]

    # Tổng dòng điện
    df["i_magnitude"] = np.sqrt(df["i_d"]**2 + df["i_q"]**2)

    # Tổng điện áp
    df["u_magnitude"] = np.sqrt(df["u_d"]**2 + df["u_q"]**2)

    # Chênh lệch nhiệt độ môi trường với chất làm mát
    df["temp_diff_ambient_coolant"] = df["ambient"] - df["coolant"]

    # Tải nhiệt (heat load proxy)
    df["heat_load"] = df["i_magnitude"] * df["motor_speed"].abs()

    # Rolling features (nếu có profile_id - theo từng session)
    if "profile_id" in df.columns:
        df = df.sort_values(["profile_id"])
        for window in [10, 30]:
            df[f"i_mag_roll_mean_{window}"] = (
                df.groupby("profile_id")["i_magnitude"]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            df[f"power_roll_mean_{window}"] = (
                df.groupby("profile_id")["electrical_power"]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
    else:
        for window in [10, 30]:
            df[f"i_mag_roll_mean_{window}"] = (
                df["i_magnitude"].rolling(window, min_periods=1).mean()
            )
            df[f"power_roll_mean_{window}"] = (
                df["electrical_power"].rolling(window, min_periods=1).mean()
            )

    new_features = [
        "electrical_power", "i_magnitude", "u_magnitude",
        "temp_diff_ambient_coolant", "heat_load",
        "i_mag_roll_mean_10", "i_mag_roll_mean_30",
        "power_roll_mean_10", "power_roll_mean_30",
    ]
    print(f"✅ Đã tạo {len(new_features)} feature mới: {new_features}")

    return df, FEATURE_COLS + new_features


# ─────────────────────────────────────────────
#  3. CHUẨN BỊ DỮ LIỆU TRAIN/TEST
# ─────────────────────────────────────────────
def prepare_data(df, features, target):
    print("\n" + "="*60)
    print(f"  📦 BƯỚC 3: CHUẨN BỊ DỮ LIỆU (target: {target})")
    print("="*60)

    # Loại bỏ NaN
    cols_needed = features + [target]
    df_clean = df[cols_needed].dropna()
    print(f"✅ Sau khi xóa NaN: {len(df_clean):,} mẫu")

    X = df_clean[features].values
    y = df_clean[target].values

    # Split: 80% train, 20% test (giữ thứ tự thời gian)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"📊 Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Chuẩn hóa
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test, scaler


# ─────────────────────────────────────────────
#  4. TRAIN MODELS
# ─────────────────────────────────────────────
def train_models(X_train, y_train):
    print("\n" + "="*60)
    print("  🤖 BƯỚC 4: TRAIN MODELS")
    print("="*60)

    models = {}

    # Model 1: Ridge Regression (baseline nhanh)
    print("  ▶ Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    models["Ridge"] = ridge
    print("    ✅ Xong")

    # Model 2: Random Forest
    print("  ▶ Training Random Forest (n=200)...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf
    print("    ✅ Xong")

    # Model 3: XGBoost hoặc GradientBoosting
    if XGB_AVAILABLE:
        print("  ▶ Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        models["XGBoost"] = xgb_model
        print("    ✅ Xong")
    else:
        print("  ▶ Training GradientBoosting...")
        gb = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        gb.fit(X_train, y_train)
        models["GradientBoosting"] = gb
        print("    ✅ Xong")

    return models


# ─────────────────────────────────────────────
#  5. ĐÁNH GIÁ MODELS
# ─────────────────────────────────────────────
def evaluate_models(models, X_test, y_test):
    print("\n" + "="*60)
    print("  📊 BƯỚC 5: ĐÁNH GIÁ MODELS")
    print("="*60)

    results = {}
    best_model_name = None
    best_mae = float("inf")

    print(f"\n  {'Model':20s} | {'MAE':>8} | {'RMSE':>8} | {'R²':>8}")
    print("  " + "-"*52)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[name] = {"mae": mae, "rmse": rmse, "r2": r2}
        print(f"  {name:20s} | {mae:>7.3f}° | {rmse:>7.3f}° | {r2:>7.4f}")

        if mae < best_mae:
            best_mae = mae
            best_model_name = name

    print(f"\n  🏆 Model tốt nhất: {best_model_name} (MAE = {best_mae:.3f}°C)")
    return results, best_model_name


# ─────────────────────────────────────────────
#  6. LƯU MODEL
# ─────────────────────────────────────────────
def save_artifacts(models, scaler, features, results, best_model_name):
    print("\n" + "="*60)
    print("  💾 BƯỚC 6: LƯU MODEL")
    print("="*60)

    # Lưu tất cả models
    for name, model in models.items():
        path = os.path.join(MODEL_DIR, f"model_{name.lower()}.joblib")
        joblib.dump(model, path)
        print(f"  ✅ Lưu: {path}")

    # Lưu scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  ✅ Lưu: {scaler_path}")

    # Lưu metadata
    metadata = {
        "features": features,
        "primary_target": PRIMARY_TARGET,
        "target_cols": TARGET_COLS,
        "best_model": best_model_name,
        "overheat_thresholds": OVERHEAT_THRESHOLD,
        "evaluation": results,
        "feature_count": len(features),
    }
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Lưu metadata: {meta_path}")

    return metadata


# ─────────────────────────────────────────────
#  7. VẼ BIỂU ĐỒ KẾT QUẢ
# ─────────────────────────────────────────────
def plot_results(models, X_test, y_test, best_model_name, features):
    print("\n" + "="*60)
    print("  📈 BƯỚC 7: VẼ BIỂU ĐỒ")
    print("="*60)

    best_model = models[best_model_name]
    y_pred = best_model.predict(X_test)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Motor Temperature Prediction - {best_model_name}", fontsize=14)

    # Plot 1: Predicted vs Actual
    ax = axes[0, 0]
    n_show = min(2000, len(y_test))
    ax.scatter(y_test[:n_show], y_pred[:n_show], alpha=0.3, s=5, color="#2196F3")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Temperature (°C)")
    ax.set_ylabel("Predicted Temperature (°C)")
    ax.set_title("Predicted vs Actual")
    ax.legend()

    # Plot 2: Time series (1000 samples)
    ax = axes[0, 1]
    n = min(1000, len(y_test))
    ax.plot(y_test[:n], label="Actual", linewidth=1)
    ax.plot(y_pred[:n], label="Predicted", linewidth=1, alpha=0.8)
    thresh = OVERHEAT_THRESHOLD.get(PRIMARY_TARGET, 100)
    ax.axhline(y=thresh, color="red", linestyle="--", linewidth=1.5, label=f"Threshold {thresh}°C")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Time Series Comparison (first {n} samples)")
    ax.legend()

    # Plot 3: Residuals
    ax = axes[1, 0]
    residuals = y_test - y_pred
    ax.hist(residuals, bins=50, color="#4CAF50", edgecolor="white", linewidth=0.5)
    ax.axvline(x=0, color="red", linestyle="--")
    ax.set_xlabel("Residual (°C)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Residual Distribution (MAE={np.abs(residuals).mean():.2f}°C)")

    # Plot 4: Feature Importance
    ax = axes[1, 1]
    if hasattr(best_model, "feature_importances_"):
        importance = best_model.feature_importances_
        top_n = 12
        indices = np.argsort(importance)[-top_n:]
        top_features = [features[i] for i in indices]
        top_importance = importance[indices]
        ax.barh(top_features, top_importance, color="#FF9800")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importance")
    else:
        ax.text(0.5, 0.5, "Feature importance\nnot available\nfor this model",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Feature Importance")

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "training_results.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Lưu biểu đồ: {plot_path}")


# ─────────────────────────────────────────────
#  8. TẠO DEMO PREDICTION
# ─────────────────────────────────────────────
def demo_prediction(models, scaler, features, best_model_name):
    print("\n" + "="*60)
    print("  🔮 BƯỚC 8: DEMO DỰ BÁO")
    print("="*60)

    # Ví dụ: Motor đang chạy bình thường
    sample_normal = {
        "ambient": 22.0,
        "coolant": 25.0,
        "u_d": 0.5,
        "u_q": 2.0,
        "motor_speed": 1000.0,
        "torque": 10.0,
        "i_d": 0.2,
        "i_q": 3.0,
    }

    # Ví dụ: Motor đang tải nặng
    sample_heavy = {
        "ambient": 35.0,
        "coolant": 40.0,
        "u_d": 1.5,
        "u_q": 5.5,
        "motor_speed": 3000.0,
        "torque": 40.0,
        "i_d": 0.8,
        "i_q": 12.0,
    }

    def predict_sample(sample_dict, label):
        # Tính các feature engineered
        i_mag = np.sqrt(sample_dict["i_d"]**2 + sample_dict["i_q"]**2)
        u_mag = np.sqrt(sample_dict["u_d"]**2 + sample_dict["u_q"]**2)
        power = sample_dict["u_d"] * sample_dict["i_d"] + sample_dict["u_q"] * sample_dict["i_q"]
        temp_diff = sample_dict["ambient"] - sample_dict["coolant"]
        heat_load = i_mag * abs(sample_dict["motor_speed"])

        full_sample = [
            sample_dict["ambient"], sample_dict["coolant"],
            sample_dict["u_d"], sample_dict["u_q"],
            sample_dict["motor_speed"], sample_dict["torque"],
            sample_dict["i_d"], sample_dict["i_q"],
            power, i_mag, u_mag, temp_diff, heat_load,
            i_mag,   # roll_mean_10 (dùng giá trị hiện tại cho demo)
            i_mag,   # roll_mean_30
            power,   # power_roll_mean_10
            power,   # power_roll_mean_30
        ]

        x = np.array(full_sample[:len(features)]).reshape(1, -1)
        x_sc = scaler.transform(x)
        pred = models[best_model_name].predict(x_sc)[0]
        thresh = OVERHEAT_THRESHOLD.get(PRIMARY_TARGET, 100)
        status = "🔴 NGUY HIỂM - QUÁ NHIỆT!" if pred > thresh else \
                 "🟡 CẢNH BÁO" if pred > thresh * 0.85 else "🟢 AN TOÀN"

        print(f"\n  [{label}]")
        print(f"    Input: speed={sample_dict['motor_speed']}rpm, "
              f"torque={sample_dict['torque']}Nm, "
              f"ambient={sample_dict['ambient']}°C")
        print(f"    → Nhiệt độ dự báo (PM): {pred:.1f}°C  {status}")

    predict_sample(sample_normal, "TÌNH HUỐNG: Motor chạy bình thường")
    predict_sample(sample_heavy,  "TÌNH HUỐNG: Motor tải nặng")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "🔥"*30)
    print("   MOTOR TEMPERATURE PREDICTION - TRAINING PIPELINE")
    print("🔥"*30)

    # Kiểm tra file dữ liệu
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ Không tìm thấy file: {DATA_PATH}")
        print("   Vui lòng tải dataset từ Kaggle và đặt vào cùng thư mục.")
        print("   Link: https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature")
        return

    # Pipeline
    df = load_and_explore(DATA_PATH)
    df, features = feature_engineering(df)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, features, PRIMARY_TARGET)
    models = train_models(X_train, y_train)
    results, best_model_name = evaluate_models(models, X_test, y_test)
    metadata = save_artifacts(models, scaler, features, results, best_model_name)
    plot_results(models, X_test, y_test, best_model_name, features)
    demo_prediction(models, scaler, features, best_model_name)

    print("\n" + "="*60)
    print("  ✅ HOÀN THÀNH!")
    print(f"  📁 Tất cả file được lưu tại: {os.path.abspath(MODEL_DIR)}/")
    print("="*60)
    print("\n  Files được tạo:")
    for f in os.listdir(MODEL_DIR):
        fpath = os.path.join(MODEL_DIR, f)
        size = os.path.getsize(fpath)
        print(f"    • {f:35s} ({size/1024:.1f} KB)")
    print()


if __name__ == "__main__":
    main()