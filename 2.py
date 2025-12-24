import os

tmp_dir = r"C:\prophet_tmp"
os.makedirs(tmp_dir, exist_ok=True)
os.environ["TMP"] = tmp_dir
os.environ["TEMP"] = tmp_dir

import pandas as pd
from prophet import Prophet
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt   

FILES = {
    "a10": {
        "path": r"C:\Users\이혜린\OneDrive\바탕 화면\dblab\fpp2\a10.csv",
        "value_col": "x",
        "start": "1991-07-01",
        "freq": "MS"
    },
    "arrivals": {
        "path": r"C:\Users\이혜린\OneDrive\바탕 화면\dblab\fpp2\arrivals.csv",
        "value_col": "Japan",
        "start": "1981-01-01",
        "freq": "QS"
    },
}

results = []

for name, cfg in FILES.items():
    print(f"\n===== {name} 처리 중 =====")

    df_raw = pd.read_csv(cfg["path"])

    n = len(df_raw)
    ds = pd.date_range(start=cfg["start"], periods=n, freq=cfg["freq"])

    df = pd.DataFrame({"ds": ds, "y": df_raw[cfg["value_col"]]}).dropna()

    # 파일마다 TEST_PERIODS 다르게 (월별 vs 분기)
    TEST_PERIODS = 12 if cfg["freq"] == "MS" else 8   # a10: 12개월, arrivals(QS): 8분기(2년)

    train = df.iloc[:-TEST_PERIODS]
    test  = df.iloc[-TEST_PERIODS:]

    model = Prophet()
    model.fit(train)

    future = model.make_future_dataframe(periods=TEST_PERIODS, freq=cfg["freq"])
    forecast = model.predict(future)

    # Test R²
    pred = test.merge(forecast[["ds", "yhat"]], on="ds", how="left")
    r2 = r2_score(pred["y"], pred["yhat"])

    results.append({"file": name, "R2_test": r2})
    print(f"Test R² = {r2:.4f}")

    # 그래프 출력 (for문 안에 있어야 파일마다 뜸)
    model.plot(forecast)
    plt.title(f"Forecast - {name}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.show()

result_df = pd.DataFrame(results).sort_values("R2_test", ascending=False)

print("\n===== 여러 파일 R² 비교 결과 =====")
print(result_df)
