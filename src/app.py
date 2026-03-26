"""CarbonCast — CBAM 탄소비용 시나리오 대시보드"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os

BASE = os.path.dirname(__file__)

# ========== 데이터 로드 ==========
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE, "data_merged.csv"), index_col=0, parse_dates=True)
    pred = pd.read_csv(os.path.join(BASE, "predictions.csv"), index_col=0, parse_dates=True)
    with open(os.path.join(BASE, "model_results.json")) as f:
        results = json.load(f)
    return df, pred, results

df, pred, model_results = load_data()

# CBAM 엔진 임포트
from importlib import import_module
import sys
sys.path.insert(0, BASE)
cbam = import_module("03_cbam_engine")

# ========== 페이지 설정 ==========
st.set_page_config(
    page_title="CarbonCast — CBAM 시나리오 엔진",
    page_icon="🏭",
    layout="wide",
)

# 헤더
st.markdown("""
<div style="background: linear-gradient(135deg, #004D3D, #008C73); padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;">
    <h1 style="color: white; margin: 0; font-size: 2rem;">CarbonCast</h1>
    <p style="color: #A8E6CF; margin: 0.5rem 0 0 0; font-size: 1.1rem;">CBAM 탄소비용 시나리오 분석 엔진 — 실제 데이터 기반</p>
</div>
""", unsafe_allow_html=True)

# ========== 사이드바: 파라미터 ==========
st.sidebar.markdown("## ⚙️ 파라미터 설정")

scenario = st.sidebar.selectbox(
    "NGFS 시나리오",
    ["Net Zero", "Below 2°C", "Delayed", "Current"],
    index=1,
    help="NGFS Phase V (2024.11) 기반 탄소가격 경로"
)

sector = st.sidebar.selectbox(
    "업종",
    list(cbam.EMISSION_FACTORS.keys()),
    index=0,
)

export_tons = st.sidebar.number_input(
    "연간 생산량 (톤)", min_value=100, max_value=10_000_000,
    value=10_000, step=1000, format="%d"
)

eu_pct = st.sidebar.slider(
    "EU 수출 비중 (%)", 0, 100, 100, 5
) / 100

exchange_rate = st.sidebar.slider(
    "환율 (원/유로)", 1200, 1800, 1450, 10
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**현재 EU ETS:** {model_results['current_price']:.1f} EUR/톤
**모델 MAPE:** {model_results['mape_pct']}%
**방향성 정확도:** {model_results['direction_acc']}%
""")

# ========== 탭 구성 ==========
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 CBAM 비용 경로", "📈 EU ETS 가격 분석", "🔍 변수 중요도", "📋 상세 테이블"
])

# ========== TAB 1: CBAM 비용 경로 ==========
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        # 모든 시나리오 비교
        fig = go.Figure()
        colors = {
            "Net Zero": "#C62828",
            "Below 2°C": "#E65100",
            "Delayed": "#1565C0",
            "Current": "#008C73",
        }

        for sc_name, color in colors.items():
            result = cbam.calc_cbam_cost(
                sc_name, sector, export_tons, eu_pct, exchange_rate
            )
            cost_billions = result["total_cost_billion_krw"] * 10  # 억원

            is_selected = (sc_name == scenario)
            fig.add_trace(go.Scatter(
                x=result["year"],
                y=cost_billions,
                name=sc_name,
                line=dict(
                    color=color,
                    width=4 if is_selected else 1.5,
                    dash="solid" if is_selected else "dot",
                ),
                opacity=1.0 if is_selected else 0.5,
                hovertemplate=f"<b>{sc_name}</b><br>" +
                    "연도: %{x}<br>비용: %{y:.1f}억원<extra></extra>",
            ))

        # Phase-in 배경
        for y, rate in cbam.CBAM_PHASE_IN.items():
            if 2026 <= y <= 2034:
                fig.add_annotation(
                    x=y, y=0, text=f"{rate*100:.1f}%",
                    showarrow=False, font=dict(size=9, color="gray"),
                    yshift=-20,
                )

        fig.update_layout(
            title=dict(
                text=f"CBAM 연간 추가비용 경로 — {sector} {export_tons:,}톤, EU {eu_pct*100:.0f}%",
                font=dict(size=16),
            ),
            xaxis_title="연도",
            yaxis_title="연간 추가비용 (억원)",
            height=480,
            template="plotly_white",
            legend=dict(orientation="h", y=-0.15),
            xaxis=dict(dtick=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 선택 시나리오 핵심 숫자
        result = cbam.calc_cbam_cost(
            scenario, sector, export_tons, eu_pct, exchange_rate
        )

        st.markdown(f"### 📌 {scenario}")

        c26 = result[result["year"] == 2026].iloc[0]["total_cost_billion_krw"] * 10
        c30 = result[result["year"] == 2030].iloc[0]["total_cost_billion_krw"] * 10
        c34 = result[result["year"] == 2034].iloc[0]["total_cost_billion_krw"] * 10
        total = result["total_cost_billion_krw"].sum() * 10

        st.metric("2026년", f"{c26:.1f}억원", "적용률 2.5%")
        st.metric("2030년", f"{c30:.1f}억원", f"적용률 48.5% (+{(c30/max(c26,0.01)-1)*100:.0f}%)")
        st.metric("2034년", f"{c34:.1f}억원", f"적용률 100% (+{(c34/max(c26,0.01)-1)*100:.0f}%)")
        st.metric("9년 누적", f"{total:.0f}억원", "2026~2034")

        st.markdown(f"""
        > **Phase-in이 핵심입니다.**
        > 2026년 {c26:.1f}억에서 2034년 {c34:.1f}억으로
        > **{c34/max(c26,0.01):.0f}배** 증가합니다.
        """)

# ========== TAB 2: EU ETS 가격 분석 ==========
with tab2:
    col1, col2 = st.columns([3, 1])

    with col1:
        fig2 = make_subplots(rows=1, cols=1)

        # 실제 가격
        fig2.add_trace(go.Scatter(
            x=df.index[-252:], y=df["eu_ets_eur"].iloc[-252:],
            name="EU ETS 실제 (CO2.L)",
            line=dict(color="#1B1B1B", width=1.5),
        ))

        # 모델 예측 범위
        fig2.add_trace(go.Scatter(
            x=pred.index, y=pred["p90"],
            name="90% 상한",
            line=dict(color="rgba(0,140,115,0.2)", width=0),
            showlegend=False,
        ))
        fig2.add_trace(go.Scatter(
            x=pred.index, y=pred["p10"],
            name="80% 구간",
            fill="tonexty",
            fillcolor="rgba(0,140,115,0.15)",
            line=dict(color="rgba(0,140,115,0.2)", width=0),
        ))
        fig2.add_trace(go.Scatter(
            x=pred.index, y=pred["p50"],
            name="50% 중앙값",
            line=dict(color="#008C73", width=2, dash="dash"),
        ))

        # 실제값 (테스트 기간)
        fig2.add_trace(go.Scatter(
            x=pred.index, y=pred["actual"],
            name="실제값",
            line=dict(color="#C62828", width=2),
        ))

        fig2.update_layout(
            title="EU ETS 가격 — 실제 vs LightGBM 예측 (5영업일 ahead)",
            xaxis_title="날짜",
            yaxis_title="EUR/톤",
            height=450,
            template="plotly_white",
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("### 모델 성능")
        st.metric("MAE", f"{model_results['mae_eur']} EUR")
        st.metric("MAPE", f"{model_results['mape_pct']}%")
        st.metric("방향성", f"{model_results['direction_acc']}%")
        st.metric("현재 가격", f"{model_results['current_price']} EUR")

        st.markdown("""
        **모델:** LightGBM Quantile Regression
        **입력:** 26개 피처 (래그, 이동평균, 변동성, 공변량)
        **출력:** 5영업일 후 10th/50th/90th 분위수
        """)

# ========== TAB 3: 변수 중요도 ==========
with tab3:
    imp = model_results["feature_importance"]
    imp_df = pd.DataFrame(list(imp.items()), columns=["Feature", "Importance"])
    imp_df = imp_df.sort_values("Importance", ascending=True)

    # 한글 레이블 매핑
    label_map = {
        "ets_std20": "ETS 20일 변동성",
        "gas_lag1": "천연가스 (1일전)",
        "eurkrw_ma20": "EUR/KRW 20일 평균",
        "ets_std5": "ETS 5일 변동성",
        "ets_std10": "ETS 10일 변동성",
        "gas_ma20": "천연가스 20일 평균",
        "ets_std60": "ETS 60일 변동성",
        "ets_ret1": "ETS 1일 수익률",
        "brent_ma20": "브렌트유 20일 평균",
        "brent_lag1": "브렌트유 (1일전)",
    }
    imp_df["Label"] = imp_df["Feature"].map(label_map).fillna(imp_df["Feature"])

    fig3 = go.Figure(go.Bar(
        x=imp_df["Importance"],
        y=imp_df["Label"],
        orientation="h",
        marker_color=["#008C73" if i >= len(imp_df)-3 else "#B0BEC5"
                       for i in range(len(imp_df))],
    ))
    fig3.update_layout(
        title="LightGBM 변수 중요도 (Split 기준) — Top 10",
        height=400,
        template="plotly_white",
        xaxis_title="중요도",
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    > **해석:** EU ETS 가격의 **변동성**(std)이 가장 중요한 변수입니다.
    > 그 다음이 **천연가스 가격** — 가스가 오르면 발전사들이 석탄으로 전환하고,
    > 탄소배출이 늘어 배출권 수요가 증가합니다.
    > 이 결과는 기존 학술 연구(Brent, NBP가 EU ETS의 주요 드라이버)와 일치합니다.
    """)

# ========== TAB 4: 상세 테이블 ==========
with tab4:
    st.markdown(f"### {scenario} — 연도별 CBAM 비용 상세")

    result = cbam.calc_cbam_cost(
        scenario, sector, export_tons, eu_pct, exchange_rate
    )
    result["total_cost_억원"] = (result["total_cost_billion_krw"] * 10).round(1)

    display_cols = {
        "year": "연도",
        "phase_in_pct": "적용률(%)",
        "eu_ets_eur": "EU ETS(€)",
        "k_ets_krw": "K-ETS(원)",
        "spread_eur": "스프레드(€)",
        "cbam_per_ton_eur": "톤당비용(€)",
        "total_cost_억원": "총비용(억원)",
    }
    show = result[list(display_cols.keys())].rename(columns=display_cols)
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown(f"""
    **계산 조건:**
    - 업종: {sector} (배출계수: {cbam.EMISSION_FACTORS[sector]} tCO₂/톤)
    - 생산량: {export_tons:,}톤, EU 수출: {eu_pct*100:.0f}%
    - 환율: {exchange_rate:,}원/유로
    - CBAM 대상 배출량: {export_tons * eu_pct * cbam.EMISSION_FACTORS[sector]:,.0f} tCO₂

    **공식:** `CBAM 부담 = (EU ETS − K-ETS/환율) × 적용률 × 배출량 × 환율`

    **출처:** EU Regulation 2023/956, NGFS Phase V (2024.11), KRX, 환경부 인벤토리
    """)

# ========== 푸터 ==========
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8C95A0; font-size: 0.85rem;">
    CarbonCast — CBAM 탄소비용 시나리오 엔진 | 하나 청년 금융인재 양성 프로젝트<br>
    데이터: Yahoo Finance (KRBN, CO2.L), NGFS Phase V, EU Regulation 2023/956<br>
    <b>이것은 "예측"이 아니라 "시나리오 분석"입니다. 모든 수치에는 불확실성이 있습니다.</b>
</div>
""", unsafe_allow_html=True)
