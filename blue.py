import RPi.GPIO as GPIO
import time
import dht11
import joblib
import pandas as pd
import csv
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque

# ================= GPIO =================
GPIO.setmode(GPIO.BCM)

GREEN   = 17
YELLOW  = 27
RED     = 22
FAN_LED = 24

GPIO.setup(GREEN,   GPIO.OUT)
GPIO.setup(YELLOW,  GPIO.OUT)
GPIO.setup(RED,     GPIO.OUT)
GPIO.setup(FAN_LED, GPIO.OUT)

# ================= SENSOR =================
sensor = dht11.DHT11(pin=4)

# ================= MODEL =================
model = joblib.load("model_v2.pkl")

# ================= SHARED DATA (thread-safe) =================
MAXLEN = 60

data_lock     = threading.Lock()
temps         = deque(maxlen=MAXLEN)
temp_changes  = deque(maxlen=MAXLEN)
avg_temps     = deque(maxlen=MAXLEN)
predictions   = deque(maxlen=MAXLEN)
fan_events    = deque(maxlen=MAXLEN)
timestamps    = deque(maxlen=MAXLEN)
reading_index = [0]

# ================= STOP EVENT =================
stop_event = threading.Event()

# ================= TRACKING =================
prev_temp    = None
temp_history = []

# ================= HELPERS =================
def all_off():
    GPIO.output(GREEN,   GPIO.LOW)
    GPIO.output(YELLOW,  GPIO.LOW)
    GPIO.output(RED,     GPIO.LOW)
    GPIO.output(FAN_LED, GPIO.LOW)

# ================= INIT CSV =================
with open("live_data.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "temp", "temp_change", "avg_temp", "prediction", "fan"])

# ================= SENSOR / AI THREAD =================
def sensor_loop():
    global prev_temp, temp_history

    while not stop_event.is_set():
        try:
            result = sensor.read()
        except Exception:
            break

        if result.is_valid():
            temperature = result.temperature
            humidity    = result.humidity

            if prev_temp is None:
                temp_change = 0
            else:
                temp_change = temperature - prev_temp
            prev_temp = temperature

            temp_history.append(temperature)
            if len(temp_history) > 3:
                temp_history.pop(0)
            avg_temp = sum(temp_history) / len(temp_history)

            input_data = pd.DataFrame(
                [[temperature, humidity, temp_change, avg_temp]],
                columns=["temperature", "humidity", "temp_change", "avg_temp"]
            )
            prediction = model.predict(input_data)[0]

            if not stop_event.is_set():
                all_off()
                fan_on = False

                if prediction == "LOW":
                    GPIO.output(GREEN, GPIO.HIGH)
                elif prediction == "MEDIUM":
                    GPIO.output(YELLOW, GPIO.HIGH)
                elif prediction == "HIGH":
                    GPIO.output(RED,     GPIO.HIGH)
                    GPIO.output(FAN_LED, GPIO.HIGH)
                    fan_on = True

                # Fan also turns on if AI predicted HIGH due to fast rise
                # (already handled above since prediction == "HIGH" covers all HIGH cases)

                with data_lock:
                    reading_index[0] += 1
                    idx = reading_index[0]
                    temps.append(temperature)
                    temp_changes.append(temp_change)
                    avg_temps.append(avg_temp)
                    predictions.append(prediction)
                    fan_events.append(fan_on)
                    timestamps.append(idx)

                with open("live_data.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([idx, temperature, temp_change, avg_temp, prediction, int(fan_on)])

                print(f"T={temperature}  Δ={temp_change:.2f}  Avg={avg_temp:.2f}  Fan={'ON' if fan_on else 'OFF'}  → {prediction}")

        for _ in range(20):
            if stop_event.is_set():
                break
            time.sleep(0.1)

# ================= GRAPH =================
def graph_loop():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle("AI Smart Cooling System — Live Monitor", fontsize=14, fontweight='bold')

    # AI prediction background colours (what the model decided)
    PRED_BG   = {"LOW": "#d4edda", "MEDIUM": "#fff3cd", "HIGH": "#f8d7da"}

    # LED line colours
    LINE_COLORS = {"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#dc3545"}

    # Fan ON colour — cyan/teal, completely separate from prediction colours
    FAN_COLOR = "#0dcaf0"

    while not stop_event.is_set():
        time.sleep(1)

        with data_lock:
            if len(temps) < 2:
                continue
            t_list  = list(timestamps)
            te_list = list(temps)
            tc_list = list(temp_changes)
            av_list = list(avg_temps)
            pr_list = list(predictions)
            fa_list = list(fan_events)

        # ── TOP CHART ──────────────────────────────────────────
        ax1.clear()

        # Layer 1: AI prediction background (green / yellow / red)
        for i in range(len(t_list) - 1):
            ax1.axvspan(t_list[i], t_list[i+1],
                        color=PRED_BG.get(pr_list[i], "#ffffff"),
                        alpha=0.35, zorder=1)

        # Layer 2: Fan ON background — cyan band drawn ON TOP of prediction colour
        # This clearly shows fan running even during LOW/MEDIUM predictions
        fan_drawn = False
        for i in range(len(t_list) - 1):
            if fa_list[i]:
                label = "Fan ON (Cyan)" if not fan_drawn else ""
                ax1.axvspan(t_list[i], t_list[i+1],
                            color=FAN_COLOR,
                            alpha=0.30, zorder=2, label=label)
                fan_drawn = True

        # Temperature lines (drawn on top of backgrounds)
        ax1.plot(t_list, te_list, color="#1a73e8", linewidth=2,
                 label="Temperature (°C)", zorder=3)
        ax1.plot(t_list, av_list, color="#6c757d", linewidth=1.5,
                 linestyle="--", label="Avg Temp (°C)", zorder=3)
        ax1.plot(t_list, tc_list, color="#fd7e14", linewidth=1.5,
                 linestyle=":", label="Temp Change (°C)", zorder=3)

        # Threshold reference lines
        ax1.axhline(y=30, color="#ffc107", linewidth=1,
                    linestyle=":", alpha=0.7, label="30°C threshold", zorder=3)
        ax1.axhline(y=35, color="#dc3545", linewidth=1,
                    linestyle=":", alpha=0.7, label="35°C threshold", zorder=3)

        ax1.set_ylabel("Temperature / Change (°C)", fontsize=10)
        ax1.set_title("Temperature & AI Decision  |  Background: AI Prediction   Cyan: Fan ON", fontsize=10)
        ax1.legend(loc="upper left", fontsize=8, ncol=3)
        ax1.grid(True, alpha=0.3, zorder=0)

        # Live status annotation
        last_pred  = pr_list[-1]
        last_temp  = te_list[-1]
        last_fan   = fa_list[-1]
        color      = LINE_COLORS.get(last_pred, "black")
        fan_status = "FAN: ON" if last_fan else "FAN: OFF"
        ax1.annotate(
            f"  Latest: {last_temp}°C  |  AI: {last_pred}  |  {fan_status}",
            xy=(0.01, 0.02), xycoords='axes fraction',
            fontsize=9, color=color, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.8)
        )

        # ── BOTTOM CHART: AI prediction + fan indicator ────────
        ax2.clear()

        pred_num   = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        p_list     = [pred_num[p] for p in pr_list]
        bar_colors = [LINE_COLORS[p] for p in pr_list]

        # AI prediction bars
        ax2.bar(t_list, p_list, color=bar_colors, width=1.0, alpha=0.75, label="AI Prediction", zorder=2)

        # Fan ON overlay bars — cyan, drawn on top of prediction bars
        fan_bar_drawn = False
        for i, fan in enumerate(fa_list):
            if fan:
                label = "Fan ON" if not fan_bar_drawn else ""
                ax2.bar(t_list[i], p_list[i], color=FAN_COLOR,
                        width=1.0, alpha=0.65, label=label, zorder=3)
                fan_bar_drawn = True

        ax2.set_yticks([1, 2, 3])
        ax2.set_yticklabels(["LOW", "MEDIUM", "HIGH"], fontsize=9)
        ax2.set_ylabel("AI Decision", fontsize=10)
        ax2.set_xlabel("Reading Number", fontsize=10)
        ax2.set_title("AI Prediction Over Time  |  Cyan bars = Fan ON", fontsize=10)
        ax2.grid(True, axis='y', alpha=0.3, zorder=0)

        # Legend
        patches = [
            mpatches.Patch(color="#28a745", label="LOW — Green LED"),
            mpatches.Patch(color="#ffc107", label="MEDIUM — Yellow LED"),
            mpatches.Patch(color="#dc3545", label="HIGH — Red LED"),
            mpatches.Patch(color=FAN_COLOR, label="Fan ON (any temperature)"),
        ]
        ax2.legend(handles=patches, loc="upper left", fontsize=8, ncol=2)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.pause(0.1)

# ================= MAIN =================
if __name__ == "__main__":
    print("Starting AI Smart Cooling System...")
    print("Graph will appear after first 2 readings.")
    print("Press Ctrl+C to stop.\n")

    sensor_thread = threading.Thread(target=sensor_loop, daemon=True)
    sensor_thread.start()

    try:
        graph_loop()
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_event.set()
        sensor_thread.join(timeout=3)
        all_off()
        GPIO.cleanup()
        plt.close()
        print("Done.")