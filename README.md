# Covid_19_Time_Series_Analysis

COVID-19 Time Series Forecasting (US)

This project analyzes and forecasts COVID-19 confirmed cases in the United States using multiple time series and regression models. It pulls real-world data from the Johns Hopkins CSSE COVID-19 repository, visualizes recent trends, and compares forecasts from different modeling approaches.

📊 Features

Automatic download of up-to-date global COVID-19 confirmed cases

Focused analysis on a selected country (default: US)

Data preprocessing:

Daily new cases calculation

7-day rolling average smoothing

Last 180 days of data used for modeling

Forecasting models:

Exponential Smoothing (Holt-Winters)

ARIMA (5,1,2)

Linear Regression

High-quality visualizations saved as PNG files

Side-by-side comparison of all models

🧠 Models Used
1. Exponential Smoothing

Additive trend

Damped trend enabled

Suitable for short-term forecasting with smooth trends

2. ARIMA (5,1,2)

Captures autoregressive and moving average components

Uses differencing to handle non-stationarity

3. Linear Regression

Simple baseline model

Uses time index as the independent variable

📁 Output Files

The script generates the following plots:

covid_original_data.png
Cumulative cases and daily new cases (7-day average)

covid_exponential_smoothing.png
30-day forecast using Exponential Smoothing

covid_arima_forecast.png
30-day forecast using ARIMA

covid_linear_regression.png
30-day forecast using Linear Regression

covid_all_models_comparison.png
Comparison of all forecasting models

🛠️ Requirements

Install the required Python packages:

pip install pandas numpy matplotlib statsmodels scikit-learn


Note:
The script uses the TkAgg backend for Matplotlib. Make sure your environment supports GUI backends.

▶️ How to Run

Clone the repository or download the script

Ensure you have Python 3.8+ installed

Run the script:

python covid_forecasting.py


Check the generated .png files in the project directory

🌍 Data Source

Johns Hopkins University CSSE COVID-19 Dataset
Public GitHub Repository (time series confirmed cases)

⚠️ Disclaimer

This project is for educational and analytical purposes only.
Forecasts are based on historical data and simplified models and should not be used for medical, public health, or policy decisions.

📌 Customization

Change the country by modifying:

COUNTRY = "US"


Adjust forecast horizon:

forecast_periods = 30


Modify ARIMA order or model parameters as needed

✨ Author

Developed for time series analysis and forecasting practice using real-world data.
**Muhammad Umer**
