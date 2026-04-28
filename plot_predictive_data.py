import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("Raspi_data - Sheet1 (2).csv")

# Clean column names
df.columns = df.columns.str.strip()

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Plot
plt.figure()

plt.plot(df["timestamp"], df["temperature"], label="Temperature")
plt.plot(df["timestamp"], df["avg_temp"], label="Moving Avg")

plt.title("Temperature Trend with AI Features")
plt.xlabel("Time")
plt.ylabel("Temperature (C)")
plt.legend()

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()