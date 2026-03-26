"""CarbonCast Step 3: CBAM 비용 계산 엔진 — Phase-in Factor 반영"""
import pandas as pd
import numpy as np
import json, os

OUT_DIR = os.path.dirname(__file__)

# ========== CBAM Phase-in Schedule (EU Regulation 2023/956) ==========
CBAM_PHASE_IN = {
    2025: 0.000,
    2026: 0.025,
    2027: 0.050,
    2028: 0.100,
    2029: 0.225,
    2030: 0.485,
    2031: 0.610,
    2032: 0.735,
    2033: 0.860,
    2034: 1.000,
}

# ========== NGFS Phase V 시나리오별 EU ETS 경로 (€/톤) ==========
# 출처: NGFS Phase V (2024.11), IIASA Scenario Explorer
# Net Zero 2050, Below 2°C, Delayed Transition, Current Policies
NGFS_EU_ETS = {
    "Net Zero": {
        2026: 80, 2027: 85, 2028: 95, 2029: 100, 2030: 110,
        2031: 118, 2032: 125, 2033: 135, 2034: 145,
    },
    "Below 2°C": {
        2026: 75, 2027: 78, 2028: 82, 2029: 86, 2030: 90,
        2031: 95, 2032: 100, 2033: 105, 2034: 110,
    },
    "Delayed": {
        2026: 70, 2027: 68, 2028: 65, 2029: 70, 2030: 78,
        2031: 85, 2032: 92, 2033: 100, 2034: 110,
    },
    "Current": {
        2026: 65, 2027: 63, 2028: 60, 2029: 58, 2030: 55,
        2031: 55, 2032: 56, 2033: 58, 2034: 60,
    },
}

# K-ETS 경로 (원/톤) — 시나리오별 추정
KETA_KRW = {
    "Net Zero": {
        2026: 12000, 2027: 15000, 2028: 18000, 2029: 22000, 2030: 25000,
        2031: 28000, 2032: 32000, 2033: 36000, 2034: 40000,
    },
    "Below 2°C": {
        2026: 11000, 2027: 13000, 2028: 16000, 2029: 18000, 2030: 20000,
        2031: 22000, 2032: 24000, 2033: 26000, 2034: 28000,
    },
    "Delayed": {
        2026: 10000, 2027: 10500, 2028: 11000, 2029: 12000, 2030: 14000,
        2031: 16000, 2032: 18000, 2033: 20000, 2034: 22000,
    },
    "Current": {
        2026: 9500, 2027: 9800, 2028: 10000, 2029: 10200, 2030: 10500,
        2031: 10800, 2032: 11000, 2033: 11200, 2034: 11500,
    },
}

# ========== CBAM 대상 전 품목 (EU Regulation 2023/956, Annex I) ==========
# 배출계수 (tCO2/톤 제품) — 출처: EU 벤치마크 / 환경부 인벤토리 / IPCC
EMISSION_FACTORS = {
    # 철강 (HS 72류, 73류)
    "철강 — 고로 (BF-BOF)": 2.0,
    "철강 — 전기로 (EAF)": 0.4,
    "철강 — 한국 평균": 1.85,
    # 알루미늄 (HS 76류)
    "알루미늄 — 1차 제련": 8.0,
    "알루미늄 — 재활용": 0.5,
    "알루미늄 — 한국 평균": 1.5,
    # 시멘트 (HS 2523)
    "시멘트 — 일반": 0.8,
    "시멘트 — 클링커": 0.83,
    # 비료 (HS 28, 31류)
    "비료 — 요소": 2.5,
    "비료 — 질산암모늄": 3.0,
    "비료 — 혼합비료": 1.8,
    # 전력 (HS 2716)
    "전력 — 화력 평균": 0.46,  # tCO2/MWh → 간접배출
    "전력 — 석탄화력": 0.9,
    "전력 — LNG": 0.35,
    # 수소 (HS 2804)
    "수소 — 그레이 (SMR)": 9.0,
    "수소 — 블루 (CCS)": 1.5,
    "수소 — 그린 (전기분해)": 0.0,
}

# 업종별 HS코드 매핑 (심사용 참고)
HS_CODE_MAP = {
    "철강": "HS 72류, 73류 (7208, 7209, 7210, 7211, 7219, 7220, 7304~7306 등)",
    "알루미늄": "HS 76류 (7601, 7603~7614 등)",
    "시멘트": "HS 2523 (포틀랜드시멘트, 알루미나시멘트, 클링커)",
    "비료": "HS 2808, 2814, 3102~3105",
    "전력": "HS 2716",
    "수소": "HS 2804 10",
}

# 한국 주요 수출기업 예시 (PF 심사 시나리오용)
SAMPLE_COMPANIES = {
    "철강 — 한국 평균": {"예시": "포스코, 현대제철, 동국제강", "EU수출비중_추정": 0.08},
    "알루미늄 — 한국 평균": {"예시": "노벨리스코리아, 삼아알미늄", "EU수출비중_추정": 0.12},
    "시멘트 — 일반": {"예시": "쌍용C&E, 한일시멘트", "EU수출비중_추정": 0.03},
    "비료 — 요소": {"예시": "롯데정밀화학, 남해화학", "EU수출비중_추정": 0.05},
    "수소 — 그레이 (SMR)": {"예시": "SK E&S, 효성", "EU수출비중_추정": 0.01},
}


def calc_cbam_cost(
    scenario: str,
    sector: str,
    export_tons: float,
    eu_export_pct: float,
    exchange_rate: float = 1450,
    years: list = None,
) -> pd.DataFrame:
    """연도별 CBAM 비용 계산

    Returns:
        DataFrame with columns: year, phase_in, eu_ets, k_ets_eur, spread_eur,
                                cbam_per_ton_eur, total_cost_krw, total_cost_billion
    """
    if years is None:
        years = list(range(2026, 2035))

    ef = EMISSION_FACTORS.get(sector, 1.0)
    cbam_tons = export_tons * eu_export_pct * ef  # CBAM 대상 배출량 (tCO2)

    rows = []
    for y in years:
        phase_in = CBAM_PHASE_IN.get(y, 1.0)
        eu_price = NGFS_EU_ETS[scenario].get(y, 60)
        k_price_krw = KETA_KRW[scenario].get(y, 10000)
        k_price_eur = k_price_krw / exchange_rate

        spread = max(eu_price - k_price_eur, 0)
        cbam_per_ton = spread * phase_in
        total_eur = cbam_tons * cbam_per_ton
        total_krw = total_eur * exchange_rate
        total_billion = total_krw / 1e9

        rows.append({
            "year": y,
            "phase_in_pct": round(phase_in * 100, 1),
            "eu_ets_eur": eu_price,
            "k_ets_krw": k_price_krw,
            "k_ets_eur": round(k_price_eur, 1),
            "spread_eur": round(spread, 1),
            "cbam_per_ton_eur": round(cbam_per_ton, 2),
            "total_cost_billion_krw": round(total_billion, 2),
        })

    return pd.DataFrame(rows)


# ========== 데모 실행 ==========
if __name__ == "__main__":
    print("="*70)
    print("CarbonCast CBAM 비용 엔진 — 실제 계산 결과")
    print("="*70)

    # 사례 1: 철강 1만톤, EU 수출 100%
    print("\n📌 사례 1: 철강(고로) 1만톤, EU 수출 100%")
    for sc in ["Current", "Below 2°C", "Net Zero"]:
        result = calc_cbam_cost(sc, "철강 (고로)", 10000, 1.0)
        print(f"\n  [{sc}] 시나리오:")
        print(f"  {'연도':>6} {'적용률':>8} {'EU ETS':>8} {'스프레드':>8} {'톤당비용':>10} {'총비용(억)':>10}")
        for _, r in result.iterrows():
            print(f"  {r['year']:>6} {r['phase_in_pct']:>7.1f}% {r['eu_ets_eur']:>7.0f}€ {r['spread_eur']:>7.1f}€ {r['cbam_per_ton_eur']:>9.2f}€ {r['total_cost_billion_krw']*10:>9.1f}억")

    # 사례 2: 시멘트 공장 PF (200만톤 생산, EU 30%)
    print("\n" + "="*70)
    print("📌 사례 2: 시멘트 공장 PF — 200만톤 생산, EU 수출 30%")
    print("  (총사업비 3,000억, 기존 IRR 8.2%)")
    for sc in ["Current", "Below 2°C", "Net Zero"]:
        result = calc_cbam_cost(sc, "시멘트", 2000000, 0.3)
        total_20yr = result["total_cost_billion_krw"].sum()
        avg_annual = total_20yr / len(result)
        print(f"\n  [{sc}]")
        print(f"    2026년: {result.iloc[0]['total_cost_billion_krw']*10:.1f}억원")
        print(f"    2030년: {result.iloc[4]['total_cost_billion_krw']*10:.1f}억원")
        print(f"    2034년: {result.iloc[8]['total_cost_billion_krw']*10:.1f}억원")
        print(f"    9년 누적: {total_20yr*10:.0f}억원")

    # 결과 저장
    all_results = {}
    for sc in NGFS_EU_ETS.keys():
        df = calc_cbam_cost(sc, "철강 (고로)", 10000, 1.0)
        all_results[sc] = df.to_dict(orient="records")

    with open(os.path.join(OUT_DIR, "cbam_scenarios.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 시나리오 결과 저장 완료")
