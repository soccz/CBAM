# CarbonCast — CBAM Transition Risk Analysis Platform

> **"AI estimates, Blockchain tracks, Finance connects."**

CarbonCast is a CBAM (Carbon Border Adjustment Mechanism) transition risk analysis platform developed for the **Hana Financial Group Youth Talent Development Project**. It helps Korean export companies and financial institutions quantify, verify, and simulate the cost impact of the EU's carbon border tax.

## What is CBAM?

The EU Carbon Border Adjustment Mechanism (EU Regulation 2023/956) imposes carbon costs on imports of steel, cement, aluminum, fertilizers, hydrogen, and electricity starting January 2026. The phase-in schedule gradually increases from 2.5% (2026) to 100% (2034).

**The core problem:** 87% of Korean SME exporters don't know how to calculate their emissions (KIEP survey). Without verified data, the EU applies default values with a +30% markup — meaning companies pay **2x or more** in carbon costs compared to their actual emissions.

## Architecture

CarbonCast consists of three layers:

```
┌─────────────────────────────────────────────────────┐
│                   CarbonCast                         │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │            AI Engine                           │  │
│  │  Emission estimation (Conformal Prediction)    │  │
│  │  Industry impact analysis (I-O + GNN)          │  │
│  │  Compliance Agent (LLM + RAG)                  │  │
│  └───────────────────────────────────────────────┘  │
│                        ↕                            │
│  ┌───────────────────────────────────────────────┐  │
│  │          Trust Layer                           │  │
│  │  Blockchain audit trail (SHA-256)              │  │
│  │  Digital Product Passport (DPP)                │  │
│  │  Federated Learning                            │  │
│  └───────────────────────────────────────────────┘  │
│                        ↕                            │
│  ┌───────────────────────────────────────────────┐  │
│  │        Financial Layer                         │  │
│  │  READIT OCR integration                        │  │
│  │  PCAF portfolio carbon analysis                │  │
│  │  Credit assessment + ESG finance               │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Module 1: AI Quality Scoring

Pre-screens emission data reliability before EU accredited verifier review. Uses Isolation Forest + rule-based checks on emission factors, energy intensity, and production volumes. Outputs a 0-100 quality score.

- Training data: K-ETS registry (~800 facilities, data.go.kr)
- Scoring: z-score based weighted sum across 4 criteria (emission factor range, energy range, energy-emission consistency, production scale)

### Module 2: Blockchain Verification

Records quality-verified emission data on blockchain for tamper-proof audit trails. Stores SHA-256 hashes on-chain with metadata; full data remains off-chain.

- Prototype: Ethereum Sepolia testnet
- Production target: Hyperledger Fabric (permissioned)

### Module 3: CBAM Cost Simulation

Calculates year-by-year CBAM cost paths (2026-2034) under 4 NGFS climate scenarios.

```
CBAM Cost = Export Volume × [Emission Factor − Benchmark × (1 − Phase-in Rate)]
            × EUA Price × Exchange Rate
```

- Phase-in rates: 2.5% (2026) → 100% (2034) per EU Regulation 2023/956, Annex IV
- Scenarios: Net Zero 2050, Below 2°C, Delayed Transition, Current Policies (NGFS Phase V)

## Project Structure

```
├── README.md
├── index.html              # Interactive demo (standalone HTML)
├── CarbonCast_app.html     # Full demo application
├── CarbonCast_v2.html      # Demo v2
├── src/
│   ├── 01_fetch_data.py    # EU ETS + market data collection
│   ├── 02_model.py         # LightGBM quantile price model
│   ├── 03_cbam_engine.py   # CBAM cost calculation engine
│   ├── 04_synthetic_data.py# Synthetic K-ETS data generation
│   ├── 05_emission_model.py# Emission factor prediction model
│   ├── app.py              # Streamlit dashboard v1
│   └── app_v2.py           # Streamlit dashboard v2
├── data/
│   ├── cbam_scenarios.json  # NGFS scenario parameters
│   ├── cbam_analysis.csv    # CBAM cost analysis results
│   ├── predictions.csv      # Model predictions
│   ├── model_results.json   # Market model metrics
│   └── emission_model_results.json
└── docs/
    ├── CarbonCast_Architecture.md
    └── [최종] CarbonCast 통합 제안서.md
```

## Key Results (Prototype)

| Metric | Value |
|:-------|:------|
| Emission model (with energy data) | R² = 0.778, MAPE 8.0% |
| EU ETS price model (LightGBM) | MAE 3.15 EUR, MAPE 4.4% |
| Steel: EU default vs actual savings | ~€154/ton (~22억 won per 10,000t) |
| Conformal prediction coverage | ≥90% (distribution-free guarantee) |

## CBAM Phase-in Schedule

| Year | 2026 | 2027 | 2028 | 2029 | 2030 | 2031 | 2032 | 2033 | 2034 |
|:-----|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| Rate | 2.5% | 5% | 10% | 22.5% | 48.5% | 61% | 73.5% | 86% | 100% |

*Source: EU Regulation 2023/956, Annex IV*

## Tech Stack

- **ML/AI:** LightGBM, XGBoost, MAPIE (Conformal Prediction), Isolation Forest
- **Visualization:** Streamlit, Plotly, HTML/JS Canvas
- **Blockchain:** Web Crypto API (SHA-256), Ethereum Sepolia (prototype)
- **Data Sources:** K-ETS registry, NGFS Phase V, EU ETS (ICE), DART Open API

## Data Sources

| Data | Source | Access |
|:-----|:-------|:-------|
| K-ETS facility data | data.go.kr/15053947 | Public API |
| NGFS scenarios | ngfs.net | Public |
| EU ETS prices | ICE Endex / Yahoo Finance | Free (delayed) |
| Industry I-O table | Bank of Korea (ecos.bok.or.kr) | Public |
| Company financials | DART Open API | Public |
| CBAM HS code mapping | Korea Customs Service | Public (2026.02) |

## Context

This project was developed for the **Hana Youth Financial Talent Development Project** (하나 청년 금융인재 양성 프로젝트), a competition organized by Hana Financial Group. The platform is part of the broader **ESG TradeGuard** system, which provides end-to-end trade finance compliance checking including OCR document parsing, regulation matching, carbon verification, and satellite environmental monitoring.

## References

- EU Regulation 2023/956 (CBAM)
- EU IR 2025/2621 (Default values and benchmarks)
- NGFS Phase V Scenarios (2024.11)
- ESPR 2024/1781 (Digital Product Passport)
- Angelopoulos & Bates (2021) "Conformal Prediction" arXiv:2107.07511
- Stanford AAAI 2025: "Learning Production Functions for Supply Chains with GNNs"

## License

This project is for educational and competition purposes.
