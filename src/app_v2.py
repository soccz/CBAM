"""CarbonCast v2 — 4-Layer CBAM 전이리스크 분석 플랫폼"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os, sys

BASE = os.path.dirname(__file__)
sys.path.insert(0, BASE)

# ========== 데이터 로드 ==========
@st.cache_data
def load_all():
    companies = pd.read_csv(os.path.join(BASE, "synthetic_kets.csv"))
    cbam = pd.read_csv(os.path.join(BASE, "cbam_analysis.csv"))
    market = pd.read_csv(os.path.join(BASE, "data_merged.csv"), index_col=0, parse_dates=True)
    with open(os.path.join(BASE, "emission_model_results.json")) as f:
        em_results = json.load(f)
    with open(os.path.join(BASE, "model_results.json")) as f:
        mkt_results = json.load(f)
    return companies, cbam, market, em_results, mkt_results

companies, cbam_df, market, em_results, mkt_results = load_all()

import importlib
cbam_engine = importlib.import_module("03_cbam_engine")

# ========== 페이지 설정 ==========
st.set_page_config(page_title="CarbonCast v2", page_icon="🏭", layout="wide")

st.markdown("""
<div style="background: linear-gradient(135deg, #004D3D 0%, #008C73 60%, #00A67E 100%);
     padding: 2rem 2.5rem; border-radius: 14px; margin-bottom: 1.5rem;">
    <h1 style="color: white; margin: 0; font-size: 2.2rem; font-weight: 800;">CarbonCast</h1>
    <p style="color: #A8E6CF; margin: 0.3rem 0 0 0; font-size: 1.05rem;">
        CBAM 전이리스크 분석 플랫폼 — 배출량 추정부터 재무 영향까지</p>
    <p style="color: rgba(255,255,255,0.5); margin: 0.5rem 0 0 0; font-size: 0.8rem;">
        데이터: K-ETS 명세서 구조 합성 628개 업체 | EU Regulation 2023/956 | NGFS Phase V</p>
</div>
""", unsafe_allow_html=True)

# ========== 상단 KPI ==========
k1, k2, k3, k4 = st.columns(4)
k1.metric("분석 대상 업체", f"{len(companies):,}개", f"K-ETS 구조 기반")
k2.metric("CBAM 직접 대상", f"{em_results['n_cbam_direct']}개", "철강/시멘트/알루미늄/비료")
k3.metric("2028 다운스트림", f"{em_results['n_cbam_downstream']}개", "자동차부품/기계/금속가공")
k4.metric("EU 수출 기업", f"{em_results['n_eu_exporters']}개", "전체의 22%")

st.markdown("---")

# ========== 탭 ==========
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 레이어 1·2 — 대상 판별 & 배출량 추정",
    "💰 레이어 3 — CBAM 비용 시나리오",
    "🏦 레이어 4 — 재무 영향 변환",
    "📊 포트폴리오 뷰",
])

# ============================================================
# TAB 1: 레이어 1+2
# ============================================================
with tab1:
    st.markdown("### 🎯 기업 검색 — CBAM 대상 여부 + 배출량 추정")

    col_search, col_result = st.columns([1, 2])

    with col_search:
        sector_filter = st.selectbox("업종", ["전체"] + sorted(companies["sector"].unique().tolist()))
        if sector_filter != "전체":
            filtered = companies[companies["sector"] == sector_filter]
        else:
            filtered = companies

        eu_only = st.checkbox("EU 수출 기업만", value=True)
        if eu_only:
            filtered = filtered[filtered["has_eu_export"]]

        st.markdown(f"**{len(filtered)}개 기업** 검색됨")

        selected_idx = st.selectbox(
            "기업 선택",
            filtered.index.tolist(),
            format_func=lambda x: f"{filtered.loc[x, 'company_name']} ({filtered.loc[x, 'sector']})"
        )

    with col_result:
        comp = filtered.loc[selected_idx]

        # CBAM 대상 판별 (레이어 1)
        if comp["cbam_direct_target"]:
            st.error(f"🔴 **CBAM 직접 대상** — {comp['sector']} (2026년 1월~)")
        elif comp["cbam_downstream_2028"]:
            st.warning(f"🟡 **2028년 다운스트림 확대 시 대상** — {comp['sector']}")
        else:
            st.success(f"🟢 **현재 CBAM 비대상** — {comp['sector']}")

        c1, c2, c3 = st.columns(3)
        c1.metric("생산량", f"{comp['production_tons']:,.0f} 톤/년")
        c2.metric("배출계수 (실측)", f"{comp['emission_factor']:.3f} tCO₂/t")
        c3.metric("EU 수출 비중", f"{comp['eu_export_pct']*100:.1f}%")

        # 배출량 추정 (레이어 2) — AI vs EU 기본값
        if comp["cbam_direct_target"]:
            EU_DEFAULTS = {"철강_고로": 3.90, "철강_전기로": 0.37, "시멘트": 1.13, "알루미늄": 1.90, "비료": 3.25}
            eu_def = EU_DEFAULTS.get(comp["sector"], 2.0)

            # cbam_analysis에서 AI 추정값 가져오기
            cbam_row = cbam_df[cbam_df["company_id"] == comp["company_id"]]
            if len(cbam_row) > 0:
                ai_est = cbam_row.iloc[0]["ai_estimate"]
                ai_q10 = cbam_row.iloc[0]["ai_q10"]
                ai_q90 = cbam_row.iloc[0]["ai_q90"]
            else:
                ai_est = comp["emission_factor"]
                ai_q10 = ai_est * 0.85
                ai_q90 = ai_est * 1.15

            saving_per_ton = (eu_def - ai_est) * 75  # EUR
            annual_saving = saving_per_ton * comp["production_tons"] * comp["eu_export_pct"] * 1450 / 1e8  # 억원

            st.markdown("#### AI 추정 vs EU 기본값")

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=["EU 기본값\n(+30% 마크업)", "AI 추정값\n(Stage 2)", "실측값\n(검증 시)"],
                y=[eu_def, ai_est, comp["emission_factor"]],
                marker_color=["#C62828", "#008C73", "#1565C0"],
                text=[f"{eu_def:.2f}", f"{ai_est:.2f}", f"{comp['emission_factor']:.3f}"],
                textposition="outside",
            ))
            # 불확실성 범위
            fig_bar.add_trace(go.Scatter(
                x=["AI 추정값\n(Stage 2)"],
                y=[ai_est],
                error_y=dict(type="data", symmetric=False,
                             array=[ai_q90 - ai_est], arrayminus=[ai_est - ai_q10]),
                mode="markers", marker=dict(size=0), showlegend=False,
            ))
            fig_bar.update_layout(
                yaxis_title="tCO₂/톤 제품",
                height=350, template="plotly_white", showlegend=False,
                title=f"배출계수 비교 — {comp['company_name']}",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            sc1, sc2, sc3 = st.columns(3)
            if saving_per_ton > 0:
                sc1.metric("톤당 절감", f"€{saving_per_ton:.0f}", "AI가 기본값보다 유리")
                sc2.metric("연간 절감", f"{annual_saving:.1f}억원", f"EU 수출 {comp['eu_export_pct']*100:.0f}% 기준")
                sc3.metric("AI 신뢰구간", f"{ai_q10:.2f} ~ {ai_q90:.2f}", "80% 구간")
            else:
                sc1.metric("톤당 차이", f"€{saving_per_ton:.0f}", "기본값이 더 유리")
                sc2.metric("권고", "기본값 사용", "AI 추정 불필요")
                sc3.metric("AI 신뢰구간", f"{ai_q10:.2f} ~ {ai_q90:.2f}", "80% 구간")

            st.info(f"""
            **해석:** AI가 추정한 배출계수({ai_est:.2f})는 EU 기본값({eu_def:.2f})보다 {'낮아' if saving_per_ton > 0 else '높아'},
            실측 데이터를 확보하면 **연간 {abs(annual_saving):.1f}억원을 {'절감' if saving_per_ton > 0 else '추가 부담'}**할 수 있습니다.
            실제 K-ETS 명세서 또는 에너지공단 마이크로데이터 API 연동 시 정확도가 더 향상됩니다.
            """)

# ============================================================
# TAB 2: 레이어 3
# ============================================================
with tab2:
    st.markdown("### 💰 CBAM 비용 시나리오 — Phase-in 반영")

    col_param, col_chart = st.columns([1, 3])

    with col_param:
        scenario = st.selectbox("NGFS 시나리오", ["Net Zero", "Below 2°C", "Delayed", "Current"])
        sector_cbam = st.selectbox("업종", list(cbam_engine.EMISSION_FACTORS.keys()))
        tons = st.number_input("생산량 (톤/년)", 1000, 10_000_000, 10_000, 1000)
        eu_pct = st.slider("EU 수출 비중", 0, 100, 30, 5) / 100
        fx = st.slider("환율 (₩/€)", 1200, 1800, 1450, 10)

    with col_chart:
        # 4개 시나리오 비교
        fig = go.Figure()
        colors = {"Net Zero": "#C62828", "Below 2°C": "#E65100", "Delayed": "#1565C0", "Current": "#008C73"}

        for sc, color in colors.items():
            r = cbam_engine.calc_cbam_cost(sc, sector_cbam, tons, eu_pct, fx)
            cost = r["total_cost_billion_krw"] * 10  # 억원
            fig.add_trace(go.Scatter(
                x=r["year"], y=cost, name=sc,
                line=dict(color=color, width=4 if sc == scenario else 1.5,
                          dash="solid" if sc == scenario else "dot"),
                opacity=1.0 if sc == scenario else 0.4,
            ))

        # Phase-in 표시
        for y in [2026, 2030, 2034]:
            rate = cbam_engine.CBAM_PHASE_IN[y]
            fig.add_annotation(x=y, y=0, text=f"적용률 {rate*100:.1f}%",
                             showarrow=False, font=dict(size=9, color="gray"), yshift=-20)

        fig.update_layout(
            title=f"연도별 CBAM 추가비용 경로",
            xaxis_title="연도", yaxis_title="연간 추가비용 (억원)",
            height=450, template="plotly_white",
            legend=dict(orientation="h", y=-0.2), xaxis=dict(dtick=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # 핵심 숫자
        result = cbam_engine.calc_cbam_cost(scenario, sector_cbam, tons, eu_pct, fx)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("2026년", f"{result[result.year==2026].iloc[0].total_cost_billion_krw*10:.1f}억")
        m2.metric("2030년", f"{result[result.year==2030].iloc[0].total_cost_billion_krw*10:.1f}억")
        m3.metric("2034년", f"{result[result.year==2034].iloc[0].total_cost_billion_krw*10:.1f}억")
        m4.metric("9년 누적", f"{result.total_cost_billion_krw.sum()*10:.0f}억")

# ============================================================
# TAB 3: 레이어 4
# ============================================================
with tab3:
    st.markdown("### 🏦 재무 영향 변환 — CBAM 비용이 신용지표를 어떻게 바꾸는가")

    col_input, col_output = st.columns([1, 2])

    with col_input:
        st.markdown("**차주 재무정보 입력**")
        st.caption("DART Open API 연동 시 자동 입력됩니다")
        rev = st.number_input("매출액 (억원)", 100, 100_000, 5_000, 100)
        ebitda = st.number_input("EBITDA (억원)", 10, 50_000, 800, 50)
        interest = st.number_input("이자비용 (억원)", 1, 10_000, 120, 10)
        debt = st.number_input("총부채 (억원)", 50, 100_000, 3_000, 100)

        st.markdown("**CBAM 시나리오**")
        sc_fin = st.selectbox("시나리오", ["Net Zero", "Below 2°C", "Current"], key="fin_sc")
        year_fin = st.selectbox("분석 연도", [2026, 2028, 2030, 2032, 2034], index=2)

    with col_output:
        # CBAM 비용 산출 (레이어 3에서 가져옴)
        r = cbam_engine.calc_cbam_cost(sc_fin, sector_cbam, tons, eu_pct, fx)
        cbam_cost = r[r.year == year_fin].iloc[0].total_cost_billion_krw * 10  # 억원

        # Before / After
        ebitda_adj = ebitda - cbam_cost
        margin_before = ebitda / rev * 100
        margin_after = ebitda_adj / rev * 100
        icr_before = ebitda / interest
        icr_after = ebitda_adj / interest
        debt_ebitda_before = debt / ebitda
        debt_ebitda_after = debt / max(ebitda_adj, 1)

        # 등급 추정 (간이)
        def grade(icr):
            if icr >= 8: return "AA"
            elif icr >= 5: return "A"
            elif icr >= 3: return "BBB"
            elif icr >= 2: return "BB"
            elif icr >= 1.5: return "B"
            else: return "CCC"

        st.markdown(f"#### {sc_fin} 시나리오, {year_fin}년 — CBAM 비용 **{cbam_cost:.1f}억원**")

        # 비교 테이블
        fig_compare = make_subplots(rows=1, cols=3, subplot_titles=[
            "EBITDA 마진 (%)", "이자보상배율", "순부채/EBITDA"
        ])

        for i, (label, before, after, fmt) in enumerate([
            ("EBITDA 마진", margin_before, margin_after, ".1f"),
            ("이자보상배율", icr_before, icr_after, ".1f"),
            ("순부채/EBITDA", debt_ebitda_before, debt_ebitda_after, ".1f"),
        ]):
            fig_compare.add_trace(go.Bar(
                x=["Before", "After"], y=[before, after],
                marker_color=["#E0E0E0", "#C62828" if after < before * 0.85 else "#E65100" if after < before else "#008C73"],
                text=[f"{before:{fmt}}", f"{after:{fmt}}"], textposition="outside",
                showlegend=False,
            ), row=1, col=i+1)

        fig_compare.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig_compare, use_container_width=True)

        g1, g2, g3 = st.columns(3)
        g1.metric("CBAM/EBITDA 비중", f"{cbam_cost/ebitda*100:.1f}%",
                  f"EBITDA의 {cbam_cost/ebitda*100:.1f}%를 탄소비용이 차지")
        g2.metric("신용등급 변화", f"{grade(icr_before)} → {grade(icr_after)}",
                  f"이자보상배율 {icr_before:.1f} → {icr_after:.1f}")
        g3.metric("추가 담보 필요", f"{'예' if grade(icr_after) < grade(icr_before) else '아니오'}",
                  "등급 하락 시" if grade(icr_after) < grade(icr_before) else "현재 등급 유지")

        st.warning(f"""
        **심사 시사점 ({sc_fin}, {year_fin}년):**
        CBAM 비용 {cbam_cost:.1f}억원은 EBITDA({ebitda}억)의 **{cbam_cost/ebitda*100:.1f}%**를 차지합니다.
        이자보상배율이 {icr_before:.1f}배 → {icr_after:.1f}배로 {'하락' if icr_after < icr_before else '유지'}하며,
        내부 신용등급 기준 **{grade(icr_before)} → {grade(icr_after)}** {'변동이 예상됩니다.' if grade(icr_after) != grade(icr_before) else '으로 유지됩니다.'}
        """)

# ============================================================
# TAB 4: 포트폴리오 뷰
# ============================================================
with tab4:
    st.markdown("### 📊 여신 포트폴리오 CBAM 노출도")

    # 업종별 집계
    portfolio = companies.groupby("sector").agg(
        업체수=("company_id", "count"),
        총배출량=("total_emissions_tco2", "sum"),
        평균배출계수=("emission_factor", "mean"),
        EU수출기업=("has_eu_export", "sum"),
        총매출_억=("revenue_billion_krw", "sum"),
    ).round(1)
    portfolio["CBAM_직접"] = portfolio.index.isin(["철강_고로", "철강_전기로", "시멘트", "알루미늄", "비료"])
    portfolio["2028_확대"] = portfolio.index.isin(["자동차부품", "기계장비", "금속가공", "건설자재"])
    portfolio = portfolio.sort_values("총배출량", ascending=False)

    # 히트맵: 업종 × 리스크
    fig_heat = go.Figure(go.Treemap(
        labels=portfolio.index,
        parents=["" for _ in portfolio.index],
        values=portfolio["총배출량"],
        marker=dict(
            colors=["#C62828" if r["CBAM_직접"] else "#E65100" if r["2028_확대"] else "#E0E0E0"
                    for _, r in portfolio.iterrows()],
        ),
        text=[f"{idx}<br>{row['업체수']}개사<br>배출 {row['총배출량']:,.0f} tCO₂"
              for idx, row in portfolio.iterrows()],
        textinfo="text",
    ))
    fig_heat.update_layout(
        title="업종별 탄소배출량 (크기) × CBAM 대상 여부 (색상)",
        height=450,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    col_legend, col_stats = st.columns([1, 2])
    with col_legend:
        st.markdown("""
        **색상 범례:**
        - 🔴 빨강: CBAM 직접 대상 (2026~)
        - 🟠 주황: 2028 다운스트림 확대 대상
        - ⚪ 회색: 현재 비대상
        """)
    with col_stats:
        direct = portfolio[portfolio["CBAM_직접"]]
        downstream = portfolio[portfolio["2028_확대"]]
        st.metric("직접 대상 업종 매출 합계", f"{direct['총매출_억'].sum():,.0f}억원")
        st.metric("다운스트림 대상 매출 합계", f"{downstream['총매출_억'].sum():,.0f}억원")
        st.metric("전체 대비 비중", f"{(direct['총매출_억'].sum()+downstream['총매출_억'].sum())/portfolio['총매출_억'].sum()*100:.1f}%")

    st.markdown("#### 업종별 상세")
    st.dataframe(portfolio, use_container_width=True)

# ========== Footer ==========
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8C95A0; font-size:0.8rem; line-height:1.8;">
    <b>CarbonCast v2</b> — CBAM 전이리스크 분석 플랫폼<br>
    레이어 1: HS코드 CBAM 매핑 | 레이어 2: AI 배출량 추정 (XGBoost, R²=0.78) |
    레이어 3: CBAM 비용 시나리오 (Phase-in 반영) | 레이어 4: 재무 영향 변환<br>
    데이터: K-ETS 명세서 구조 합성 628개 업체 · NGFS Phase V · EU Regulation 2023/956<br>
    <b>실제 K-ETS 명세서 API(data.go.kr/15053947) + DART Open API 연동 시 즉시 실 데이터 전환 가능</b><br>
    하나 청년 금융인재 양성 프로젝트
</div>
""", unsafe_allow_html=True)
