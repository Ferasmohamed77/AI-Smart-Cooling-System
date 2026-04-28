import time
import pandas as pd
import matplotlib.pyplot as plt

plt.ion()
fig, ax = plt.subplots()

def label_to_num(x):
    if x == "LOW":
        return 0
    elif x == "MEDIUM":
        return 5
    else:
        return 10

while True:
    try:
        df = pd.read_csv("live_data.csv")

        if len(df) < 2:
            time.sleep(1)
            continue

        df["pred_num"] = df["prediction"].apply(label_to_num)

        ax.clear()

        # � Temperature
        ax.plot(df["temp"], label="Temperature", linewidth=2)

        # � Temp Change (trend)
        ax.plot(df["temp_change"], label="Temp Change", linestyle=":")

        # � Avg Temp (smooth)
        ax.plot(df["avg_temp"], label="Average Temp", linestyle="--")

        # � AI Prediction
        ax.plot(df["pred_num"], label="AI Prediction", linewidth=2)

        ax.set_title("AI Predictive Temperature System")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        ax.legend()
        ax.grid(True)

        plt.pause(1)

    except Exception as e:
        print("Waiting for data...", e)

    time.sleep(1)