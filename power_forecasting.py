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

# Plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def prophet_forecast(cleaned_df):
    """This method and framework did not seem to work very well."""

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

    future = m.make_future_dataframe(periods=26, freq="YS")
    future["cap"] = 72000
    fcst = m.predict(future)

    # This is a matplotlib figure, so we can update it to look nice with this import.
    import matplotlib_rc

    fig = m.plot(fcst)
    ax = fig.gca()

    ax.set_ylabel("Net Power Generation [Gwh]")
    ax.set_xlabel("Time [years]")
    fig.suptitle("Power generation in New Zealand, only fitting historical data.")

    # Make x axis more legible.
    ax.set_xlim([pd.Timestamp("1974-01-01"), pd.Timestamp("2050-01-01")])
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)

    fig.tight_layout()
    ax.legend()

    fig.savefig("projection_historical.pdf")


def monte_carlo_growth(cleaned_df):
    # use our historical data
    base_value = cleaned_df["y"].values[-1]
    years = 26
    num_simulations = 1000
    # These should be taken from the dataset maybe
    target_growth_rate = 0.02  # 5% target annual growth rate
    std_growth_rate = 0.02  # get std from data.
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


def plot_monte_carlo(df):

    # Analyze results
    mean_prediction = df.mean(axis=0)
    lower_bound = df.quantile(0.025, axis=0)
    upper_bound = df.quantile(0.975, axis=0)

    # Styling so output looks nice
    import matplotlib_rc

    fig, ax = plt.subplots()

    ax.plot(mean_prediction, label="Mean Prediction", color="blue")
    ax.fill_between(
        range(26 + 1),
        lower_bound,
        upper_bound,
        color="blue",
        alpha=0.2,
        label="95% Confidence Interval",
    )

    ax.hlines(
        y=72000,
        xmin=0,
        xmax=26,
        linewidth=0.5,
        linestyle="--",
        label="Upper limit for power generation",
        color="r",
    )

    ax.set_title("Projected Future Values with Positive Growth Scenario")
    ax.set_xlabel("Years")
    ax.set_ylabel("Projected Value")
    ax.legend()
    ax.grid()

    # plt.show()
    plt.savefig("first_monte_carlo.pdf")


def main():
    fp = os.path.join(os.getcwd(), "electricity.xlsx")
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

    # prophet_forecast(reduced_df)
    df_carlo = monte_carlo_growth(reduced_df)
    plot_monte_carlo(df_carlo)


if __name__ == main():
    main()
