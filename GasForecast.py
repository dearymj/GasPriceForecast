"""
Natural Gas Price Forecasting with Prophet

This script demonstrates:
1. Loading monthly natural gas prices from a CSV file.
2. Training a Prophet model to forecast natural gas prices.
3. Estimating the gas price on any date (past or future).
4. Visualizing both the historical data and future forecasts with explanatory annotations.

Requirements:
    pip install prophet
    pip install matplotlib
    pip install pandas

Usage:
    python NaturalGasPrices.py

Author: Your Name
Date: YYYY-MM-DD
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# 1. Suppress cmdstanpy logs so they don't clutter the console
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def load_data(csv_file: str) -> pd.DataFrame:
    """
    Load monthly natural gas price data from a CSV file and return a DataFrame
    compatible with Prophet. Assumes date strings are in 'MM/DD/YY' format.

    :param csv_file: Path to the CSV file containing 'Dates' and 'Prices'.
    :return: A pandas DataFrame with columns 'ds' and 'y'.
    """
    df = pd.read_csv(csv_file)

    # Specify the exact date format to avoid warnings (e.g., '10/31/20').
    df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')

    # Rename columns for Prophet
    df.rename(columns={'Dates': 'ds', 'Prices': 'y'}, inplace=True)

    # Sort and reset index
    df.sort_values('ds', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def train_prophet_model(df: pd.DataFrame) -> Prophet:
    """
    Train a Prophet model on the monthly natural gas price data.

    :param df: DataFrame with columns 'ds' (dates) and 'y' (prices).
    :return: A fitted Prophet model.
    """
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    model.fit(df)
    return model


def create_forecast(model: Prophet, periods: int = 12) -> pd.DataFrame:
    """
    Create a forecast for a specified number of future months.

    :param model: A trained Prophet model.
    :param periods: Number of future months to forecast. (Default=12 for one year)
    :return: A DataFrame containing forecast results including:
             ds (date), yhat (forecasted value), yhat_lower, yhat_upper (confidence intervals).
    """
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)
    return forecast


def get_price_estimate(model: Prophet, input_date: str) -> float:
    """
    Estimate the natural gas price for a given date (past or future).

    :param model: A trained Prophet model.
    :param input_date: Date string in 'YYYY-MM-DD' format.
    :return: Estimated natural gas price as a float.
    """
    df_input = pd.DataFrame({'ds': [pd.to_datetime(input_date)]})
    forecast_input = model.predict(df_input)
    return float(forecast_input['yhat'].iloc[0])

def visualize_data_and_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame, model: Prophet):
    """
    Plot historical data and forecasted values using Matplotlib, with annotations
    positioned so they do not overlap the main forecast curves.

    Figure 1 (main forecast):
      - Black dots: Historical data used for training (actual values).
      - Blue line: Model's forecasted values (yhat).
      - Light blue shaded areas: Uncertainty intervals (yhat_lower and yhat_upper).

    Figure 2 (forecast components):
      - Trend: Overall upward or downward movement in natural gas prices over time.
      - Yearly: Seasonal patterns repeating each year (e.g., monthly peaks/troughs).
    """
    # 1. Plot the forecast
    fig1 = model.plot(forecast_df, xlabel='Date', ylabel='Price')

    # Access the main plot axis
    ax1 = fig1.gca()

    # Change colors of components
    for line in ax1.get_lines():
        line.set_color('darkorange')  # Set forecast line to dark orange

    # Change historical data points color
    ax1.get_lines()[0].set_color('black')  # Historical points remain black
    ax1.get_lines()[0].set_marker('o')  # Ensure historical points are visible

    # Modify uncertainty interval color
    for child in ax1.get_children():
        if isinstance(child, plt.Polygon):  # Uncertainty interval is a Polygon
            child.set_facecolor('lightyellow')  # Change to light yellow
            child.set_alpha(0.3)  # Keep transparency

    plt.title('Natural Gas Price Forecast (Prophet) | By MJ Yuan', fontsize=14)

    # Place annotation in the top-left corner of the main forecast plot
    ax1 = fig1.axes[0]
    ax1.text(
        0.02, 0.95,  # x=0.02, y=0.95 in axes fraction
        "Black dots = Historical data\nOrange line = Forecast\nLight blue area = Uncertainty interval",
        transform=ax1.transAxes,
        fontsize=10,
        color='darkgreen',
        va='top',
        ha='left',
        bbox=dict(boxstyle="round,pad=0.5", ec="green", fc="lightgreen", alpha=0.3)
    )

    plt.tight_layout()  # Ensure the annotation is not cut off
    plt.show()

    # 2. Plot the forecast components (trend, yearly seasonality, etc.)
    fig2 = model.plot_components(forecast_df)
    fig2.suptitle('Forecast Components | By MJ Yuan', fontsize=14)

    # Grab axes for the subplots
    ax_components = fig2.axes

    # Annotation for the 'Trend' subplot (first component)
    ax_components[0].text(
        0.02, 0.95,
        "Trend:\nOverall movement of natural gas prices",
        transform=ax_components[0].transAxes,
        fontsize=9, color='darkgreen',
        va='top',
        ha='left',
        bbox=dict(boxstyle="round,pad=0.3", ec="green", fc="lightgreen", alpha=0.3)
    )

    # Annotation for the 'Yearly Seasonality' subplot (second component)
    ax_components[1].text(
        0.02, 0.95,
        "Yearly Seasonality:\nRepeating seasonal pattern each year",
        transform=ax_components[1].transAxes,
        fontsize=9, color='darkgreen',
        va='top',
        ha='left',
        bbox=dict(boxstyle="round,pad=0.3", ec="green", fc="lightgreen", alpha=0.3)
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Path to CSV file (change if needed)
    csv_file_path = "Nat_Gas.csv"

    # 1. Load data
    df_prices = load_data(csv_file_path)

    # 2. Train Prophet model
    prophet_model = train_prophet_model(df_prices)

    # 3. Forecast for the next 12 months
    forecast_df = create_forecast(prophet_model, periods=12)

    # 4. Estimate price on a past date
    sample_past_date = "2022-05-15"
    estimated_past_price = get_price_estimate(prophet_model, sample_past_date)
    print(f"Estimated price on {sample_past_date}: ${estimated_past_price:.2f}")

    # 5. Estimate price on a future date
    sample_future_date = "2025-05-15"
    estimated_future_price = get_price_estimate(prophet_model, sample_future_date)
    print(f"Estimated price on {sample_future_date}: ${estimated_future_price:.2f}")

    # 6. Visualize data and forecast
    visualize_data_and_forecast(df_prices, forecast_df, prophet_model)
