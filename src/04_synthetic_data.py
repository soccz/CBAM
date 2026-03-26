"""CarbonCast: K-ETS 기반 합성 데이터 생성
실제 K-ETS 명세서(data.go.kr/15053947)의 구조를 모방하되,
한국 산업 현실에 맞는 분포로 합성 생성.
"""
import pandas as pd
import numpy as np
import os, json

np.random.seed(42)
OUT = os.path.dirname(__file__)

# ========== 1. K-ETS 관리업체 합성 (700개) ==========
# 실제 K-ETS: 약 700개 할당대상업체, 업종/배출량/에너지사용량 보유

SECTORS = {
    "철강_고로":      {"n": 8,   "prod_range": (50_000, 5_000_000),  "ef_mean": 1.85, "ef_std": 0.15, "energy_per_ton": 5.5},
    "철강_전기로":    {"n": 25,  "prod_range": (10_000, 500_000),    "ef_mean": 0.40, "ef_std": 0.08, "energy_per_ton": 0.8},
    "시멘트":         {"n": 18,  "prod_range": (100_000, 3_000_000), "ef_mean": 0.80, "ef_std": 0.06, "energy_per_ton": 1.1},
    "알루미늄":       {"n": 12,  "prod_range": (5_000, 200_000),     "ef_mean": 1.50, "ef_std": 0.30, "energy_per_ton": 14.0},
    "비료":           {"n": 15,  "prod_range": (20_000, 500_000),    "ef_mean": 2.50, "ef_std": 0.40, "energy_per_ton": 2.0},
    "석유화학":       {"n": 45,  "prod_range": (50_000, 2_000_000),  "ef_mean": 1.20, "ef_std": 0.25, "energy_per_ton": 3.5},
    "발전_석탄":      {"n": 30,  "prod_range": (500_000, 10_000_000),"ef_mean": 0.90, "ef_std": 0.10, "energy_per_ton": 2.5},
    "발전_LNG":       {"n": 25,  "prod_range": (200_000, 5_000_000), "ef_mean": 0.35, "ef_std": 0.05, "energy_per_ton": 1.8},
    "자동차":         {"n": 20,  "prod_range": (10_000, 500_000),    "ef_mean": 0.60, "ef_std": 0.10, "energy_per_ton": 1.2},
    "자동차부품":     {"n": 80,  "prod_range": (1_000, 100_000),     "ef_mean": 0.45, "ef_std": 0.12, "energy_per_ton": 0.9},
    "조선":           {"n": 15,  "prod_range": (5_000, 200_000),     "ef_mean": 0.55, "ef_std": 0.10, "energy_per_ton": 1.0},
    "반도체":         {"n": 12,  "prod_range": (1_000, 50_000),      "ef_mean": 0.30, "ef_std": 0.08, "energy_per_ton": 25.0},
    "디스플레이":     {"n": 8,   "prod_range": (5_000, 100_000),     "ef_mean": 0.35, "ef_std": 0.07, "energy_per_ton": 8.0},
    "제지":           {"n": 20,  "prod_range": (10_000, 500_000),    "ef_mean": 0.65, "ef_std": 0.12, "energy_per_ton": 1.5},
    "유리_세라믹":    {"n": 15,  "prod_range": (5_000, 300_000),     "ef_mean": 0.70, "ef_std": 0.15, "energy_per_ton": 2.0},
    "섬유":           {"n": 30,  "prod_range": (1_000, 100_000),     "ef_mean": 0.50, "ef_std": 0.10, "energy_per_ton": 1.3},
    "식음료":         {"n": 40,  "prod_range": (5_000, 500_000),     "ef_mean": 0.25, "ef_std": 0.08, "energy_per_ton": 0.6},
    "기계장비":       {"n": 50,  "prod_range": (500, 50_000),        "ef_mean": 0.40, "ef_std": 0.10, "energy_per_ton": 0.7},
    "전자부품":       {"n": 60,  "prod_range": (100, 30_000),        "ef_mean": 0.35, "ef_std": 0.10, "energy_per_ton": 2.5},
    "금속가공":       {"n": 70,  "prod_range": (500, 80_000),        "ef_mean": 0.50, "ef_std": 0.12, "energy_per_ton": 1.0},
    "건설자재":       {"n": 30,  "prod_range": (5_000, 300_000),     "ef_mean": 0.55, "ef_std": 0.10, "energy_per_ton": 1.2},
}

# 기업명 접두사
PREFIXES = ["한국", "대한", "동양", "삼화", "세아", "동국", "한일", "대성", "현대", "삼성",
            "SK", "포스코", "롯데", "KCC", "OCI", "효성", "한화", "두산", "LS", "코오롱",
            "금호", "태광", "동부", "쌍용", "영풍", "고려", "대림", "GS", "CJ", "LG"]
SUFFIXES = ["", "산업", "제철", "화학", "에너지", "소재", "테크", "금속", "스틸", "시멘트"]

rows = []
company_id = 1000

for sector, params in SECTORS.items():
    for i in range(params["n"]):
        prod = np.random.uniform(*params["prod_range"])
        ef = max(0.05, np.random.normal(params["ef_mean"], params["ef_std"]))
        emissions = prod * ef
        energy = prod * params["energy_per_ton"] * np.random.uniform(0.8, 1.2)

        # 기업 규모에 따른 재무 데이터 생성
        revenue = prod * np.random.uniform(50, 500) * 1000  # 원
        employees = int(prod * np.random.uniform(0.005, 0.05))
        employees = max(5, min(employees, 50000))

        # 전력 vs 연료 비중
        elec_ratio = np.random.uniform(0.3, 0.8)
        elec_kwh = energy * elec_ratio * 277.78  # TJ → MWh 근사
        fuel_tj = energy * (1 - elec_ratio)

        # 연료 종류
        fuel_types = ["LNG", "유연탄", "경유", "중유", "LPG", "바이오매스"]
        fuel_probs = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
        primary_fuel = np.random.choice(fuel_types, p=fuel_probs)

        name = f"{np.random.choice(PREFIXES)}{np.random.choice(SUFFIXES)}"
        if np.random.random() < 0.3:
            name += f" {i+1}공장"

        # EU 수출 여부
        cbam_sectors = ["철강_고로", "철강_전기로", "시멘트", "알루미늄", "비료"]
        downstream_sectors = ["자동차부품", "기계장비", "금속가공", "건설자재"]
        has_eu_export = sector in cbam_sectors or (sector in downstream_sectors and np.random.random() < 0.3)
        eu_export_pct = np.random.uniform(0.02, 0.25) if has_eu_export else 0

        rows.append({
            "company_id": f"KR-{company_id:05d}",
            "company_name": name,
            "sector": sector,
            "ksic_code": f"C{np.random.randint(10,33):02d}{np.random.randint(10,99):02d}",
            "production_tons": round(prod),
            "emission_factor": round(ef, 3),
            "total_emissions_tco2": round(emissions),
            "energy_tj": round(energy, 1),
            "electricity_mwh": round(elec_kwh),
            "fuel_tj": round(fuel_tj, 1),
            "primary_fuel": primary_fuel,
            "employees": employees,
            "revenue_billion_krw": round(revenue / 1e9, 1),
            "ebitda_billion_krw": round(revenue / 1e9 * np.random.uniform(0.08, 0.25), 1),
            "debt_billion_krw": round(revenue / 1e9 * np.random.uniform(0.3, 1.5), 1),
            "interest_billion_krw": round(revenue / 1e9 * np.random.uniform(0.01, 0.06), 1),
            "has_eu_export": has_eu_export,
            "eu_export_pct": round(eu_export_pct, 3) if has_eu_export else 0,
            "cbam_direct_target": sector in cbam_sectors,
            "cbam_downstream_2028": sector in downstream_sectors,
        })
        company_id += 1

df = pd.DataFrame(rows)

print(f"✅ 합성 데이터 생성: {len(df)}개 업체")
print(f"\n📊 업종별 분포:")
print(df.groupby("sector").agg(
    업체수=("company_id", "count"),
    평균배출량=("total_emissions_tco2", "mean"),
    평균배출계수=("emission_factor", "mean"),
).round(1).to_string())

print(f"\n🎯 CBAM 대상:")
print(f"  직접 대상: {df['cbam_direct_target'].sum()}개")
print(f"  2028 다운스트림: {df['cbam_downstream_2028'].sum()}개")
print(f"  EU 수출 기업: {df['has_eu_export'].sum()}개")

df.to_csv(os.path.join(OUT, "synthetic_kets.csv"), index=False)
print(f"\n💾 저장: synthetic_kets.csv")
