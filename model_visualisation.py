import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Styling so output looks nice
import matplotlib_rc


def plot_prophet(prophet_obj, df_fcst, filename):
    fig = prophet_obj.plot(df_fcst)
    ax = fig.gca()

    print("Terminal predicted values:\n")
    print(f"{df_fcst.ds.iloc[-1]}: {df_fcst.yhat.iloc[-1]}")

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

    fig.savefig(filename)


def plot_monte_carlo(df, filename):
    mean_prediction = df.mean(axis=0)
    # lower_bound = df.quantile(0.025, axis=0)
    # upper_bound = df.quantile(0.975, axis=0)

    sigma1_lower = df.quantile(0.32, axis=0)
    sigma1_upper = df.quantile(0.68, axis=0)

    sigma2_lower = df.quantile(0.05, axis=0)
    sigma2_upper = df.quantile(0.95, axis=0)

    fig, ax = plt.subplots()

    ax.plot(mean_prediction, label="Mean Prediction", color="blue")

    ax.fill_between(
        range(26 + 1),
        sigma2_lower,
        sigma2_upper,
        color="red",
        alpha=0.2,
        label=r"$2 \sigma$" + " confidence interval",
    )

    ax.fill_between(
        range(26 + 1),
        sigma1_lower,
        sigma1_upper,
        color="blue",
        alpha=0.2,
        label=r"$1 \sigma$" + " confidence interval",
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

    ax.set_title("Simulating the Innovation scenario using Monte Carlo.")
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("Power generation [GWh]")
    ax.legend()
    ax.grid()

    # plt.show()
    plt.savefig(filename)


def plot_min_max_df(prophet_dfs: list, monte_carlo_dfs: list):
    # fig, ax = plt.subplots(2, 1, constrained_layout=True, sharex=True)

    # def _helper_plotter(df, ax, added_text):
    #     for column in df.columns:
    #         ax.plot(
    #             df[column].index + 2023,
    #             df[column].values,
    #             label=f"{column} {added_text}",
    #         )

    # _helper_plotter(prophet_dfs[0], ax[0], "Prophet minimal CO2-e")
    # _helper_plotter(prophet_dfs[1], ax[0], "Prophet maximum CO2-e")
    # _helper_plotter(monte_carlo_dfs[0], ax[1], "Monte Carlo minimal CO2-e")
    # _helper_plotter(monte_carlo_dfs[1], ax[1], "Monte Carlo maximum CO2-e")

    # ax[0].set_title("Prophet CO2-e emissions.")
    # ax[1].set_title("Monte Carlo CO2-e emissions.")

    # ax[0].grid()
    # ax[1].grid()

    # ax[0].legend()
    # ax[1].legend()

    fig, ax = plt.subplots(constrained_layout=True)

    x = prophet_dfs[0][prophet_dfs[0].columns[0]].index + 2023

    for energy_type in prophet_dfs[0].columns:
        prophet_min = prophet_dfs[0][energy_type]
        prophet_max = prophet_dfs[1][energy_type]
        # This is a little gross, but works
        prophet_y = pd.concat([prophet_min, prophet_max], axis=1).mean(axis=1).values

        monte_carlo_min = monte_carlo_dfs[0][energy_type]
        monte_carlo_max = monte_carlo_dfs[1][energy_type]
        monte_carlo_y = (
            pd.concat([monte_carlo_min, monte_carlo_max], axis=1).mean(axis=1).values
        )

        ax.errorbar(
            x,
            prophet_y,
            yerr=[prophet_min, prophet_max],
            label=f"Prophet {energy_type} CO2-e",
        )

        ax.errorbar(
            x,
            monte_carlo_y,
            yerr=[monte_carlo_min, monte_carlo_max],
            label=f"Monte Carlo {energy_type} CO2-e",
        )

    ax.legend()

    plt.show()
