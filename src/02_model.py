"""CarbonCast Step 2: LightGBM Quantile Regression — 실제 데이터 기반"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import json, os

OUT_DIR = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(OUT_DIR, "data_merged.csv"), index_col=0, parse_dates=True)

TARGET = "eu_ets_eur"
HORIZON = 5  # 5영업일 ahead
LOOKBACK = 60

print(f"📊 데이터: {len(df)}일, 타겟: {TARGET}")
print(f"   현재 EU ETS: {df[TARGET].iloc[-1]:.2f} EUR/톤")

# ========== Feature Engineering ==========
print("\n🔧 피처 엔지니어링...")

feat = pd.DataFrame(index=df.index)

# 래그 변수
for lag in [1, 2, 3, 5, 10, 20]:
    feat[f"ets_lag{lag}"] = df[TARGET].shift(lag)

# 이동평균
for win in [5, 10, 20, 60]:
    feat[f"ets_ma{win}"] = df[TARGET].rolling(win).mean()
    feat[f"ets_std{win}"] = df[TARGET].rolling(win).std()

# 변화율
feat["ets_ret1"] = df[TARGET].pct_change(1)
feat["ets_ret5"] = df[TARGET].pct_change(5)
feat["ets_ret20"] = df[TARGET].pct_change(20)

# 공변량 래그 (당일은 미래 데이터이므로 1일 래그)
for col in ["gas", "brent", "eurkrw"]:
    feat[f"{col}_lag1"] = df[col].shift(1)
    feat[f"{col}_ret5"] = df[col].pct_change(5).shift(1)
    feat[f"{col}_ma20"] = df[col].rolling(20).mean().shift(1)

# 타겟: 5일 후 가격
feat["target"] = df[TARGET].shift(-HORIZON)

# 결측 제거
feat = feat.dropna()
print(f"   피처 수: {len([c for c in feat.columns if c != 'target'])}")
print(f"   학습 가능 샘플: {len(feat)}")

# ========== Train/Test Split ==========
split_date = feat.index[-60]  # 최근 60일 테스트
train = feat[feat.index < split_date]
test = feat[feat.index >= split_date]

X_train = train.drop("target", axis=1)
y_train = train["target"]
X_test = test.drop("target", axis=1)
y_test = test["target"]

print(f"\n📈 학습: {len(train)}일, 테스트: {len(test)}일")

# ========== LightGBM Quantile x 3 ==========
quantiles = [0.1, 0.5, 0.9]
models = {}
predictions = {}

params_base = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "verbose": -1,
}

for q in quantiles:
    print(f"\n🤖 LightGBM Quantile={q} 학습 중...")
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=q,
        **params_base,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    models[q] = model
    predictions[q] = pred

    mae = np.mean(np.abs(pred - y_test.values))
    print(f"   MAE: {mae:.2f} EUR, 평균 오차율: {mae / y_test.mean() * 100:.1f}%")

# ========== Quantile Crossing 보정 ==========
p10 = predictions[0.1]
p50 = predictions[0.5]
p90 = predictions[0.9]

# 10th <= 50th <= 90th 강제
for i in range(len(p10)):
    vals = sorted([p10[i], p50[i], p90[i]])
    p10[i], p50[i], p90[i] = vals

# ========== 평가 ==========
print("\n" + "="*60)
print("📊 테스트 결과 (최근 60일, 5영업일 ahead)")
print("="*60)

y_actual = y_test.values
mae_50 = np.mean(np.abs(p50 - y_actual))
mape_50 = np.mean(np.abs((p50 - y_actual) / y_actual)) * 100

# 방향성 정확도
actual_dir = np.sign(y_actual[1:] - y_actual[:-1])
pred_dir = np.sign(p50[1:] - p50[:-1])
dir_acc = np.mean(actual_dir == pred_dir) * 100

# 커버리지: 실제값이 10~90% 범위 안에 있는 비율
coverage = np.mean((y_actual >= p10) & (y_actual <= p90)) * 100

print(f"  중앙값(50%) MAE:     {mae_50:.2f} EUR")
print(f"  중앙값(50%) MAPE:    {mape_50:.1f}%")
print(f"  방향성 정확도:       {dir_acc:.1f}%")
print(f"  80% 구간 커버리지:   {coverage:.1f}% (목표: 80%)")

# 최근 5일 예시
print(f"\n📋 최근 예측 vs 실제:")
print(f"{'날짜':>12} {'실제':>8} {'10%':>8} {'50%':>8} {'90%':>8}")
for i in range(-5, 0):
    dt = test.index[i].strftime("%Y-%m-%d")
    print(f"{dt:>12} {y_actual[i]:>8.2f} {p10[i]:>8.2f} {p50[i]:>8.2f} {p90[i]:>8.2f}")

# ========== Feature Importance ==========
print(f"\n🔍 변수 중요도 (Top 10):")
imp = pd.Series(
    models[0.5].feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

for i, (name, val) in enumerate(imp.head(10).items()):
    print(f"  {i+1}. {name:20s} {val:>6.0f}")

# ========== 저장 ==========
results = {
    "test_days": len(test),
    "mae_eur": round(mae_50, 2),
    "mape_pct": round(mape_50, 1),
    "direction_acc": round(dir_acc, 1),
    "coverage_80": round(coverage, 1),
    "current_price": round(df[TARGET].iloc[-1], 2),
    "feature_importance": {k: int(v) for k, v in imp.head(10).items()},
}
with open(os.path.join(OUT_DIR, "model_results.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# 예측 결과 저장
pred_df = pd.DataFrame({
    "actual": y_actual,
    "p10": p10,
    "p50": p50,
    "p90": p90,
}, index=test.index)
pred_df.to_csv(os.path.join(OUT_DIR, "predictions.csv"))

# 모델 저장
for q, model in models.items():
    model.booster_.save_model(os.path.join(OUT_DIR, f"lgbm_q{int(q*100)}.txt"))

print(f"\n✅ 완료! 모델 3개 + 예측 결과 저장됨")
