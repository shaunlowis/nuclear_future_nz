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

    plt.savefig(filename)


def plot_min_max_df(
    prophet_dfs: list, monte_carlo_dfs: list, filename: str, title: str
):
    fig, ax = plt.subplots(constrained_layout=True)

    x = prophet_dfs[0][prophet_dfs[0].columns[0]].index + 2024

    co2e_vals = []

    for energy_type in prophet_dfs[0].columns:
        prophet_min = prophet_dfs[0][energy_type]
        prophet_max = prophet_dfs[1][energy_type]
        prophet_y = pd.concat([prophet_min, prophet_max], axis=1).mean(axis=1).values

        monte_carlo_min = monte_carlo_dfs[0][energy_type]
        monte_carlo_max = monte_carlo_dfs[1][energy_type]
        monte_carlo_y = (
            pd.concat([monte_carlo_min, monte_carlo_max], axis=1).mean(axis=1).values
        )

        combined_y = np.mean([prophet_y, monte_carlo_y], axis=0)
        co2e_vals.append(combined_y)

        ax.errorbar(
            x,
            combined_y,
            yerr=[prophet_y, monte_carlo_y],
            label=f"{energy_type}",
            linewidth=1.5,
            elinewidth=1,
            capsize=2,
        )

    co2e_vals = np.array(co2e_vals).sum(axis=0)
    ax.plot(x, co2e_vals, label="Total")

    print(co2e_vals)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        fancybox=True,
        shadow=True,
        ncol=4,
    )
    ax.set_title(title)
    ax.set_ylabel("CO2-e [kt]")
    ax.set_xlabel("Time [years]")
    ax.grid(linewidth=0.5)

    # Make x axis more legible.
    ax.set_xlim([2023, 2051])
    plt.xticks(rotation=45)

    plt.savefig(filename)
