import sys
import pandas as pd
import matplotlib.pyplot as plt
from day_ahead_v2.data import DataHandler

if __name__ == "__main__":
    # Set up data handler for wind power forecast data
    wind_FC = DataHandler("Enfor_DA_wind_power_forecast.csv", "data/raw/")

    wind_FC = wind_FC.transform_data(
        normalize_columns=[
            "PowerPred", "SettlementPowerMeas", "SCADAPowerMeas"
            ]
        )
    wind_FC.data["Time_begin"] = pd.to_datetime(
        wind_FC.data["Time_begin"], format="%Y-%m-%d %H:%M:%S")

    # Check for duplicates in Time_begin
    datetime_duplicates = wind_FC.data["Time_begin"].duplicated().sum()
    print(f"Number of duplicate Time_begin entries: {datetime_duplicates}")
    # Print all rows that have duplicate Time_begin values
    dup_rows = wind_FC.data[wind_FC.data["Time_begin"].duplicated(keep=False)].sort_values("Time_begin")
    if dup_rows.empty:
        print("No duplicate Time_begin rows found.")
    else:
        print(f"Found {len(dup_rows)} rows with duplicate Time_begin values:")
        print(dup_rows)

    sys.exit()

    # Plot predictions vs measurements
    plt.figure(figsize=(12, 6))
    plt.plot(wind_FC.data["Time_begin"], wind_FC.data["PowerPred"],
             label="Power Prediction", alpha=0.7, linewidth=0.3)
    plt.plot(wind_FC.data["Time_begin"], wind_FC.data["SCADAPowerMeas"],
             label="SCADA Power Measurement", alpha=0.7, linewidth=0.3)
    plt.plot(wind_FC.data["Time_begin"], wind_FC.data["SettlementPowerMeas"],
             label="Settlement Power Measurement", alpha=0.7, linewidth=0.3)
    plt.xlabel("Time")
    plt.ylabel("Normalized Power")
    plt.legend()
    plt.show()

    # Compute residual and plot it
    wind_FC.data["residual"] = wind_FC.data["PowerPred"] - wind_FC.data["SCADAPowerMeas"]
    print(wind_FC.data["residual"].iloc[:10])

    plt.figure(figsize=(12, 6))
    plt.plot(wind_FC.data["Time_begin"], wind_FC.data["residual"],
             label="Residual (PowerPred - SCADA)", alpha=0.8, linewidth=0.6)
    plt.axhline(0, color="k", linestyle="--", linewidth=0.8)
    plt.ylim(-1, 1)
    plt.xlabel("Time")
    plt.ylabel("Normalized Residual Power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Compute residual and plot it
    wind_FC.data["residual_1"] = wind_FC.data["SettlementPowerMeas"] - wind_FC.data["SCADAPowerMeas"]
    print(wind_FC.data["residual_1"].iloc[:10])

    plt.figure(figsize=(12, 6))
    plt.plot(wind_FC.data["Time_begin"], wind_FC.data["residual_1"],
             label="Residual (SettlementPowerMeas - SCADA)", alpha=0.8, linewidth=0.6)
    plt.axhline(0, color="k", linestyle="--", linewidth=0.8)
    plt.ylim(-1, 1)
    plt.xlabel("Time")
    plt.ylabel("Normalized Residual Power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
