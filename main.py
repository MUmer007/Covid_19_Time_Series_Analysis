import pandas as pd 
import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from statsmodels.tsa.holtwinters import ExponentialSmoothing 
from statsmodels.tsa.arima.model import ARIMA 
from sklearn.linear_model import LinearRegression 
import warnings
warnings.filterwarnings('ignore')

print("Loading COVID-19 data...")

url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
covid_data = pd.read_csv(url)

COUNTRY = "US"

country_data = covid_data[covid_data['Country/Region'] == COUNTRY]
date_columns = country_data.columns[4:]
cases = country_data[date_columns].sum()

data = pd.DataFrame({
    'Date': pd.to_datetime(date_columns),
    'Cases': cases.values
})

data['Daily_Cases'] = data['Cases'].diff().fillna(0)
data['Daily_Cases'] = data['Daily_Cases'].clip(lower=0)
data['Cases_7day_avg'] = data['Daily_Cases'].rolling(window=7, min_periods=1).mean()
data.set_index('Date', inplace=True)
data = data.tail(180)

print(f"Analyzing {COUNTRY}")
print(f"Date range: {data.index.min().date()} to {data.index.max().date()}")
print(f"Total cases: {data['Cases'].iloc[-1]:,.0f}")

plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(data.index, data['Cases'], linewidth=2, color='darkblue')
plt.title(f'Total COVID-19 Cases in {COUNTRY}', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative Cases', fontsize=11)
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(data.index, data['Daily_Cases'], alpha=0.3, color='gray', label='Daily Cases')
plt.plot(data.index, data['Cases_7day_avg'], linewidth=2.5, color='red', label='7-day Average')
plt.title(f'Daily New Cases in {COUNTRY}', fontsize=14, fontweight='bold')
plt.ylabel('Daily New Cases', fontsize=11)
plt.xlabel('Date', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('covid_original_data.png', dpi=150, bbox_inches='tight')
plt.close()

forecast_column = 'Cases_7day_avg'
ts_data = data[forecast_column].dropna()
forecast_periods = 30
last_date = data.index[-1]

print("\nRunning Exponential Smoothing...")

model_es = ExponentialSmoothing(ts_data, trend="add", seasonal=None, damped_trend=True)
es_fit = model_es.fit(optimized=True)
es_forecast = es_fit.forecast(forecast_periods)

future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
es_forecast.index = future_dates

plt.figure(figsize=(14, 7))
plt.plot(ts_data.index, ts_data, label="Actual Data", linewidth=2.5, color='blue', marker='o', markersize=3)
plt.plot(es_forecast.index, es_forecast, label="ES Forecast (30 days)", linewidth=2.5, color='red', linestyle='--', marker='s', markersize=5)
plt.axvline(x=last_date, color='green', linestyle=':', linewidth=2, label='Forecast Start')
plt.title(f'Exponential Smoothing Forecast - {COUNTRY}', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=13)
plt.ylabel('Daily Cases (7-day avg)', fontsize=13)
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('covid_exponential_smoothing.png', dpi=150, bbox_inches='tight')
plt.close()

print("Running ARIMA...")

arima_model = ARIMA(ts_data, order=(5, 1, 2))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=forecast_periods)
arima_forecast.index = future_dates

plt.figure(figsize=(14, 7))
plt.plot(ts_data.index, ts_data, label="Actual Data", linewidth=2.5, color='blue', marker='o', markersize=3)
plt.plot(arima_forecast.index, arima_forecast, label="ARIMA Forecast (30 days)", linewidth=2.5, color='green', linestyle='--', marker='s', markersize=5)
plt.axvline(x=last_date, color='orange', linestyle=':', linewidth=2, label='Forecast Start')
plt.title(f'ARIMA(5,1,2) Forecast - {COUNTRY}', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=13)
plt.ylabel('Daily Cases (7-day avg)', fontsize=13)
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('covid_arima_forecast.png', dpi=150, bbox_inches='tight')
plt.close()

print("Running Linear Regression...")

reg_df = data[[forecast_column]].dropna().reset_index()
reg_df['TimeIndex'] = np.arange(len(reg_df))

X = reg_df[['TimeIndex']]
y = reg_df[forecast_column]

model_lr = LinearRegression()
model_lr.fit(X, y)

future_time = np.arange(len(reg_df), len(reg_df) + forecast_periods).reshape(-1, 1)
lr_predictions = model_lr.predict(future_time)

plt.figure(figsize=(14, 7))
plt.plot(reg_df['Date'], y, label="Actual Data", linewidth=2.5, color='blue', marker='o', markersize=3)
plt.plot(future_dates, lr_predictions, label="Linear Regression Forecast (30 days)", linewidth=2.5, color='orange', linestyle='--', marker='s', markersize=5)
plt.axvline(x=last_date, color='purple', linestyle=':', linewidth=2, label='Forecast Start')
plt.title(f'Linear Regression Forecast - {COUNTRY}', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=13)
plt.ylabel('Daily Cases (7-day avg)', fontsize=13)
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('covid_linear_regression.png', dpi=150, bbox_inches='tight')
plt.close()

print("Creating comparison plot...")

plt.figure(figsize=(16, 8))
plt.plot(ts_data.index, ts_data, label="Actual Data", linewidth=3, color='blue', alpha=0.7)
plt.plot(es_forecast.index, es_forecast, label="Exponential Smoothing", linewidth=2.5, linestyle='--', marker='o', markersize=4, color='red')
plt.plot(arima_forecast.index, arima_forecast, label="ARIMA", linewidth=2.5, linestyle='--', marker='s', markersize=4, color='green')
plt.plot(future_dates, lr_predictions, label="Linear Regression", linewidth=2.5, linestyle='--', marker='^', markersize=4, color='orange')
plt.axvline(x=last_date, color='black', linestyle=':', linewidth=2, label='Forecast Start', alpha=0.7)
plt.title(f'COVID-19 Forecast Comparison - {COUNTRY}', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Daily Cases (7-day avg)', fontsize=14)
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('covid_all_models_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll graphs saved successfully!")
print("  - covid_original_data.png")
print("  - covid_exponential_smoothing.png")
print("  - covid_arima_forecast.png")
print("  - covid_linear_regression.png")
print("  - covid_all_models_comparison.png")
print("\nDone!")