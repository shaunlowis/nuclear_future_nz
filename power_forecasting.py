"""
Written by J.M. (Shaun) Lowis, for ENME618: Future Project Assignment.

electricity.xlsx was retrieved from:
https://www.mbie.govt.nz/building-and-energy/energy-and-natural-resources/energy-statistics-and-modelling/energy-statistics/electricity-statistics

Please see the attached report for further information regarding model parameters.

In VS Code, you can go: Ctrl+SHIFT+P --> Select Interpreter --> + Create Virtual Environment for setup.
"""

import os, prophet
import pandas as pd
import numpy as np


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

    # Make sure this column is recognised as dates.
    reduced_df["ds"] = pd.to_datetime(reduced_df["ds"], errors="coerce", format="%Y")

    reduced_df["cap"] = (
        60000  # see: https://facebook.github.io/prophet/docs/saturating_forecasts.html#forecasting-growth
    )

    # m = prophet.Prophet(growth="logistic")
    m = prophet.Prophet()
    m.fit(reduced_df)

    future = m.make_future_dataframe(periods=365)
    fcst = m.predict(future)
    fig = m.plot(fcst)

    # future = m.make_future_dataframe(periods=1826)  # days
    # fcst = m.predict(future)
    # fig = m.plot(fcst)

    fig.savefig("projection.png")


if __name__ == main():
    main()
