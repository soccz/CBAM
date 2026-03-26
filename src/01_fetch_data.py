"""CarbonCast Step 1: 실제 데이터 수집 — EU ETS 프록시 + 공변량"""
import yfinance as yf
import pandas as pd
import os

OUT_DIR = os.path.dirname(__file__)

def get_close(ticker, start="2020-01-01"):
    """yfinance MultiIndex 처리하여 Close 시리즈 반환"""
    raw = yf.download(ticker, start=start, progress=False)
    if raw.empty:
        return pd.Series(dtype=float)
    if hasattr(raw.columns, 'levels'):
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close
    return raw["Close"]

# 1. EU Carbon — KRBN ETF (EU ETS 추종) + CO2.L (WisdomTree Carbon ETC, 유로 기준)
print("📡 EU Carbon 데이터 수집...")
krbn = get_close("KRBN", "2020-01-01")
co2l = get_close("CO2.L", "2020-01-01")
print(f"  KRBN: {len(krbn)}일, CO2.L: {len(co2l)}일")

# 2. 천연가스 — TTF=F 또는 NG=F
print("📡 천연가스 수집...")
gas = get_close("TTF=F", "2020-01-01")
if len(gas) < 100:
    print("  → TTF=F 부족, NG=F (Henry Hub) 사용")
    gas = get_close("NG=F", "2020-01-01")
print(f"  Gas: {len(gas)}일")

# 3. 브렌트유
print("📡 브렌트유 수집...")
brent = get_close("BZ=F", "2020-01-01")
print(f"  Brent: {len(brent)}일")

# 4. EUR/KRW
print("📡 EUR/KRW 수집...")
eurkrw = get_close("EURKRW=X", "2020-01-01")
print(f"  EUR/KRW: {len(eurkrw)}일")

# 5. EUR/USD (CO2.L 유로→달러 변환용)
print("📡 EUR/USD 수집...")
eurusd = get_close("EURUSD=X", "2020-01-01")
print(f"  EUR/USD: {len(eurusd)}일")

# 6. 병합
print("\n🔧 데이터 병합...")
df = pd.DataFrame({
    "krbn": krbn,
    "co2_eur": co2l,
    "gas": gas,
    "brent": brent,
    "eurkrw": eurkrw,
    "eurusd": eurusd,
})
df.index = pd.to_datetime(df.index)
df = df.ffill().bfill().dropna()

# EU ETS 가격 추정: CO2.L(유로 기준 ETC)을 기본으로, 없으면 KRBN 사용
# CO2.L은 유로 표시 EU ETS 선물 가격에 가까움
df["eu_ets_eur"] = df["co2_eur"]

print(f"\n✅ 최종 데이터: {len(df)}일, {df.index.min().date()} ~ {df.index.max().date()}")
print(df.tail(10))
print(f"\n📊 기초 통계:")
print(df.describe().round(2))

out_path = os.path.join(OUT_DIR, "data_merged.csv")
df.to_csv(out_path)
print(f"\n💾 저장: {out_path}")
