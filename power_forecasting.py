"""
Written by J.M. (Shaun) Lowis, for ENME618: Future Project Assignment.

electricity.xlsx was retrieved from:
https://www.mbie.govt.nz/building-and-energy/energy-and-natural-resources/energy-statistics-and-modelling/energy-statistics/electricity-statistics

Please see the attached report for further information regarding model parameters.

In VS Code, you can go: Ctrl+SHIFT+P --> Select Interpreter --> + Create Virtual Environment for setup.

You can then install the needed packages with: 
python -m pip install -r requirements.txt
"""

# File I/O
import os

# Data analysis
import pandas as pd
import numpy as np

# Curve fitting
import prophet

from model_visualisation import plot_monte_carlo, plot_prophet, plot_min_max_df
import matplotlib.pyplot as plt


def prophet_forecast(cleaned_df):
    """Worked really well."""

    # Make sure this column is recognised as dates.
    cleaned_df["ds"] = pd.to_datetime(cleaned_df["ds"], errors="coerce", format="%Y")

    cleaned_df["cap"] = (
        72000  # see: https://facebook.github.io/prophet/docs/saturating_forecasts.html#forecasting-growth
    )

    m = prophet.Prophet(growth="logistic")
    # m = prophet.Prophet()
    m.fit(cleaned_df)

    # docs: https://facebook.github.io/prophet/docs/non-daily_data.html#monthly-data
    # you can input different frequencies for forecasting, from this set of strings:
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

    future = m.make_future_dataframe(periods=27, freq="YS")
    future["cap"] = 72000
    fcst = m.predict(future)

    return m, fcst


def monte_carlo_growth(cleaned_df):
    # We want our output to be reproducible for data analysis
    np.random.seed(42)
    # use our historical data
    base_value = cleaned_df["y"].values[-1]
    years = 26
    num_simulations = 1000
    # These should be taken from the dataset maybe
    target_growth_rate = 0.02  # 5% target annual growth rate
    # std_growth_rate = 0.02  # get std from data.
    std_growth_rate = cleaned_df["y"].std() / cleaned_df["y"].mean()
    predictions = []

    # This is a very basic approach, but good enough for our case.
    for _ in range(num_simulations):
        future_values = [base_value]

        # We iterate through the number of years for each simulatio
        # so we can generate a large enough sample size to conformt to the central limit theorem.
        for __ in range(years):
            sampled_growth_rate = np.random.normal(
                loc=target_growth_rate, scale=std_growth_rate
            )
            # growth projection comes in here.
            next_value = future_values[-1] * (1 + sampled_growth_rate)
            future_values.append(next_value)

        predictions.append(future_values)

    return pd.DataFrame(predictions)


def apply_energy_mix(emissions_array: np.ndarray, add_nuclear=False):
    if add_nuclear:
        energy_df = pd.DataFrame(
            {
                "energy_type": [
                    "Hydro",
                    "Geothermal",
                    "Wind",
                    "Solar",
                    "Coal",
                    "Gas",
                    "Nuclear",
                ],
                "energy_proportion": [0.6, 0.18, 0.07, 0.01, 0, 0, 0.11],
                "min_co2e": [6, 21, 7.8, 27, 751, 430, 5.1],
                "max_co2e": [147, 304, 16, 122, 1095, 513, 6.4],
            }
        )
    else:
        energy_df = pd.DataFrame(
            {
                "energy_type": [
                    "Hydro",
                    "Geothermal",
                    "Wind",
                    "Solar",
                    "Coal",
                    "Gas",
                    "Nuclear",
                ],
                "energy_proportion": [0.6, 0.18, 0.07, 0.01, 0.02, 0.09, 0],
                "min_co2e": [6, 21, 7.8, 27, 751, 430, 5.1],
                "max_co2e": [147, 304, 16, 122, 1095, 513, 6.4],
            }
        )

    power_types = energy_df["energy_type"].to_list()

    output_df_min = pd.DataFrame(columns=power_types)
    output_df_max = pd.DataFrame(columns=power_types)

    # Initially, power units from electricity.xlsx is net generation in GWh.
    # The value for CO2-e is g/kWh.
    # 1 GWh = 1e6 kWh, so multiply power value by 1e6
    # Then multiply electricity value by CO2-e equivalent, for g.
    # 1g = 1e-9 kt. Then final units are CO2-e [kt]

    for i, power_type in enumerate(power_types):
        output_df_min[power_type] = (
            emissions_array
            * energy_df["energy_proportion"].iloc[i]
            * energy_df["min_co2e"].iloc[i]
            * 1e6
            * 1e-9
        )
        output_df_max[power_type] = (
            emissions_array
            * energy_df["energy_proportion"].iloc[i]
            * energy_df["max_co2e"].iloc[i]
            * 1e6
            * 1e-9
        )

    return output_df_min, output_df_max


def main():
    fp = os.path.join(os.getcwd(), "data/electricity.xlsx")
    # Retrieve annual power generation:
    df = pd.read_excel(fp, sheet_name=2, skiprows=8)

    # Data wrangling, retrieve the calendar year,Net Generation in GWh and Renewable share:
    reduced_df = pd.DataFrame(
        {
            # We have to name the columns as per: https://facebook.github.io/prophet/docs/quick_start.html
            "ds": np.array(df.iloc[1, 0:-1].index[1:]),  # Calendar year
            "y": df.iloc[1, 0:-1].values[1:],  # Net Generation (Gwh)
            # "Renewable share (%)": df.iloc[12, 0:-1].values[1:],
        }
    )

    prophet_obj, df_fcst = prophet_forecast(reduced_df)
    plot_prophet(prophet_obj, df_fcst, "report_plots/projection_historical.pdf")
    df_carlo = monte_carlo_growth(reduced_df)
    plot_monte_carlo(df_carlo, "report_plots/first_monte_carlo.pdf")

    df_carlo_min, df_carlo_max = apply_energy_mix(df_carlo.mean(axis=0).values)
    df_prophet_min, df_prophet_max = apply_energy_mix(df_fcst.yhat.iloc[-27:].values)

    plot_min_max_df(
        prophet_dfs=[df_prophet_min, df_prophet_max],
        monte_carlo_dfs=[df_carlo_min, df_carlo_max],
        filename="report_plots/sim_co2e.pdf",
        title="Simulated CO2-e values \n using 2023 NZ energy mix.",
    )

    # Now with nuclear replacing fossil fuels
    df_carlo_min, df_carlo_max = apply_energy_mix(
        df_carlo.mean(axis=0).values, add_nuclear=True
    )
    df_prophet_min, df_prophet_max = apply_energy_mix(
        df_fcst.yhat.iloc[-27:].values, add_nuclear=True
    )

    plot_min_max_df(
        prophet_dfs=[df_prophet_min, df_prophet_max],
        monte_carlo_dfs=[df_carlo_min, df_carlo_max],
        filename="report_plots/sim_co2e_nuclear.pdf",
        title="Simulated CO2-e values \n using 2023 NZ energy mix, replacing fossil fuels with nuclear.",
    )


if __name__ == main():
    main()
