"""CarbonCast 레이어 2: AI 배출량 추정 모델
K-ETS 명세서 데이터로 학습 → 비대상 중소기업에 적용
핵심: "EU 기본값 대비 얼마나 절감 가능한가"
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
import json, os

OUT = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(OUT, "synthetic_kets.csv"))

print(f"📊 데이터: {len(df)}개 업체")

# ========== Feature Engineering ==========
le_sector = LabelEncoder()
le_fuel = LabelEncoder()

features = pd.DataFrame({
    "sector_enc": le_sector.fit_transform(df["sector"]),
    "production_tons": df["production_tons"],
    "energy_tj": df["energy_tj"],
    "electricity_mwh": df["electricity_mwh"],
    "fuel_tj": df["fuel_tj"],
    "primary_fuel_enc": le_fuel.fit_transform(df["primary_fuel"]),
    "employees": df["employees"],
    "revenue_billion": df["revenue_billion_krw"],
    # 파생 변수
    "energy_per_ton": df["energy_tj"] / df["production_tons"].clip(lower=1),
    "elec_ratio": df["electricity_mwh"] / (df["energy_tj"].clip(lower=0.1) * 277.78),
    "revenue_per_employee": df["revenue_billion_krw"] / df["employees"].clip(lower=1),
    "log_production": np.log1p(df["production_tons"]),
    "log_energy": np.log1p(df["energy_tj"]),
})

target = df["emission_factor"]  # tCO2/톤 제품

print(f"   피처 수: {features.shape[1]}")

# ========== Stage 1: 최소 데이터 (업종 + 매출만) ==========
print("\n" + "="*60)
print("Stage 1 — 최소 데이터 추정 (업종 + 매출)")
print("="*60)

feat_stage1 = features[["sector_enc", "revenue_billion", "employees"]].copy()

model_s1 = lgb.LGBMRegressor(n_estimators=200, max_depth=4, verbose=-1)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores_s1 = cross_val_score(model_s1, feat_stage1, target, cv=cv, scoring="r2")
print(f"  R² = {scores_s1.mean():.3f} ± {scores_s1.std():.3f}")

model_s1.fit(feat_stage1, target)
pred_s1 = model_s1.predict(feat_stage1)
mape_s1 = np.mean(np.abs((pred_s1 - target) / target)) * 100
print(f"  MAPE = {mape_s1:.1f}%")

# ========== Stage 2: Activity-based (에너지 데이터 포함) ==========
print("\n" + "="*60)
print("Stage 2 — Activity-based ML (에너지 데이터 포함)")
print("="*60)

model_s2 = lgb.LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, verbose=-1)
scores_s2 = cross_val_score(model_s2, features, target, cv=cv, scoring="r2")
print(f"  R² = {scores_s2.mean():.3f} ± {scores_s2.std():.3f}")

model_s2.fit(features, target)
pred_s2 = model_s2.predict(features)
mape_s2 = np.mean(np.abs((pred_s2 - target) / target)) * 100
print(f"  MAPE = {mape_s2:.1f}%")

# Quantile 모델 (불확실성)
models_q = {}
for q in [0.1, 0.5, 0.9]:
    m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=500, max_depth=6, verbose=-1)
    m.fit(features, target)
    models_q[q] = m

pred_q10 = models_q[0.1].predict(features)
pred_q50 = models_q[0.5].predict(features)
pred_q90 = models_q[0.9].predict(features)

# ========== Feature Importance ==========
imp = pd.Series(model_s2.feature_importances_, index=features.columns).sort_values(ascending=False)
print(f"\n🔍 변수 중요도:")
for name, val in imp.items():
    print(f"  {name:25s} {val:>5.0f}")

# ========== EU 기본값 대비 절감 분석 ==========
print("\n" + "="*60)
print("💰 EU 기본값 대비 절감 분석")
print("="*60)

EU_DEFAULTS_2028 = {  # 2028년 기준, +30% 마크업
    "철강_고로": 3.90, "철강_전기로": 0.37,
    "시멘트": 1.13, "알루미늄": 1.90, "비료": 3.25,
}

cbam_companies = df[df["cbam_direct_target"]].copy()
cbam_companies["ai_estimate"] = pred_s2[df["cbam_direct_target"]]
cbam_companies["ai_q10"] = pred_q10[df["cbam_direct_target"]]
cbam_companies["ai_q90"] = pred_q90[df["cbam_direct_target"]]
cbam_companies["eu_default"] = cbam_companies["sector"].map(EU_DEFAULTS_2028)
cbam_companies["saving_per_ton_eur"] = (cbam_companies["eu_default"] - cbam_companies["ai_estimate"]) * 75
cbam_companies["annual_saving_billion"] = (
    cbam_companies["saving_per_ton_eur"]
    * cbam_companies["production_tons"]
    * cbam_companies["eu_export_pct"]
    * 1450 / 1e9
)

for sector in EU_DEFAULTS_2028:
    sub = cbam_companies[cbam_companies["sector"] == sector]
    if len(sub) == 0:
        continue
    eu_def = EU_DEFAULTS_2028[sector]
    avg_ai = sub["ai_estimate"].mean()
    avg_saving = sub["saving_per_ton_eur"].mean()
    total_saving = sub["annual_saving_billion"].sum()
    print(f"\n  [{sector}]")
    print(f"    EU 기본값(+30%): {eu_def:.2f} tCO₂/t")
    print(f"    AI 추정 평균:    {avg_ai:.2f} tCO₂/t")
    print(f"    톤당 절감:       €{avg_saving:.0f}")
    print(f"    EU수출 기업 연간 절감 합계: {total_saving:.1f}억원")
    print(f"    해당 기업 수:    {len(sub[sub['has_eu_export']]):,}개")

# ========== 저장 ==========
results = {
    "stage1_r2": round(scores_s1.mean(), 3),
    "stage1_mape": round(mape_s1, 1),
    "stage2_r2": round(scores_s2.mean(), 3),
    "stage2_mape": round(mape_s2, 1),
    "feature_importance": {k: int(v) for k, v in imp.items()},
    "n_companies": len(df),
    "n_cbam_direct": int(df["cbam_direct_target"].sum()),
    "n_cbam_downstream": int(df["cbam_downstream_2028"].sum()),
    "n_eu_exporters": int(df["has_eu_export"].sum()),
}
with open(os.path.join(OUT, "emission_model_results.json"), "w") as f:
    json.dump(results, f, indent=2)

cbam_companies.to_csv(os.path.join(OUT, "cbam_analysis.csv"), index=False)
print(f"\n💾 저장 완료")
