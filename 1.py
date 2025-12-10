import os

tmp_dir = r"C:\prophet_tmp"
os.makedirs(tmp_dir, exist_ok=True)
os.environ["TMP"] = tmp_dir
os.environ["TEMP"] = tmp_dir

import pandas as pd
from prophet import Prophet
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

csv_path = r"C:\Users\이혜린\OneDrive\바탕 화면\dblab\fpp2\a10.csv"  # 네 경로 그대로 넣음

df_raw = pd.read_csv(csv_path)
print("원본 데이터 상위 5개:")
print(df_raw.head())

n = len(df_raw)

dates = pd.date_range(start="1991-07-01", periods=n, freq="MS")

df = pd.DataFrame({
    "ds": dates,      
    "y": df_raw["x"],  
})

print("\nProphet에 넣을 데이터 상위 5개:")
print(df.head())

model = Prophet()         
model.fit(df)           

future = model.make_future_dataframe(periods=12, freq="MS")
forecast = model.predict(future)

print("\n예측 결과 일부:")
print(forecast[["ds", "yhat"]].tail())

model.plot(forecast)
plt.title("a10 데이터 Prophet 예측")
plt.xlabel("Date")
plt.ylabel("Value")
plt.show()

merged = df.merge(forecast[["ds", "yhat"]], on="ds", how="left")

r2 = r2_score(merged["y"], merged["yhat"])
print(f"\nR² (결정계수): {r2:.4f}")
