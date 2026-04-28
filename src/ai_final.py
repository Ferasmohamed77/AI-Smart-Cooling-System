# ================================================================
# AI CONTROL V2 — Live AI Control + Real-Time Graph
# ================================================================
# PURPOSE:
#   This script is the THIRD and final step in the AI pipeline.
#   It loads the trained model (model_v2.pkl), reads live sensor
#   data every 2 seconds, feeds it into the AI model, controls
#   the LEDs and fan based on the prediction, and displays a
#   live graph showing everything in real time.
#
# TWO THINGS RUN IN PARALLEL:
#   1. Sensor thread  — reads sensor, runs AI, controls hardware
#   2. Main thread    — updates the live matplotlib graph
#
#   Threading is used so the graph never slows down the sensor
#   readings and the sensor never freezes the graph.
# ================================================================

import RPi.GPIO as GPIO                  # Controls GPIO pins on the Raspberry Pi
import time                              # Used for sleep delays
import dht11                             # Reads the DHT11 temperature/humidity sensor
import joblib                            # Loads the saved AI model from the .pkl file
import pandas as pd                      # Formats sensor data into the shape the model expects
import csv                               # Saves each reading to a local CSV file
import threading                         # Runs sensor loop and graph loop at the same time
import matplotlib.pyplot as plt          # Draws the live graph
import matplotlib.patches as mpatches   # Creates colored legend boxes for the graph
from collections import deque           # A list with a maximum size — old data is auto-removed

# ================================================================
# GPIO SETUP
# ================================================================
# Set up the Raspberry Pi GPIO pins for all 4 LEDs.
# BCM mode means we use the GPIO number (e.g. GPIO17),
# not the physical pin number on the board.

GPIO.setmode(GPIO.BCM)

GREEN   = 17   # Green LED  — indicates LOW temperature (safe)
YELLOW  = 27   # Yellow LED — indicates MEDIUM temperature (warm)
RED     = 22   # Red LED    — indicates HIGH temperature (hot)
FAN_LED = 24   # Fan LED    — turns on whenever cooling is needed

GPIO.setup(GREEN,   GPIO.OUT)   # Set each pin as an output (we send signals, not receive)
GPIO.setup(YELLOW,  GPIO.OUT)
GPIO.setup(RED,     GPIO.OUT)
GPIO.setup(FAN_LED, GPIO.OUT)

# ================================================================
# SENSOR SETUP
# ================================================================

sensor = dht11.DHT11(pin=4)   # DHT11 sensor connected to GPIO pin 4

# ================================================================
# MODEL SETUP
# ================================================================
# Load the trained AI model from the file saved by train_model_v2.py.
# This loads all 200 decision trees into memory.
# The model stays loaded for the entire runtime — no reloading needed.

model = joblib.load("model_v2.pkl")

# ================================================================
# SHARED DATA STORAGE
# ================================================================
# These deques (double-ended queues) store the last 60 readings
# for the graph. A deque with maxlen=60 automatically removes
# the oldest value when a new one is added, so the graph always
# shows a rolling window of the last 60 readings.
#
# data_lock is a threading lock — it prevents both threads from
# reading/writing the data at the exact same moment, which could
# cause corrupted or half-written values. Before accessing the
# shared data, a thread must "acquire" the lock, and releases it
# when done. The other thread waits until the lock is free.

MAXLEN = 60   # Number of readings to show on the graph at once

data_lock     = threading.Lock()          # Prevents data corruption between threads
temps         = deque(maxlen=MAXLEN)      # Temperature readings
temp_changes  = deque(maxlen=MAXLEN)      # Temperature change per reading
avg_temps     = deque(maxlen=MAXLEN)      # Rolling average temperature
predictions   = deque(maxlen=MAXLEN)      # AI prediction (LOW / MEDIUM / HIGH)
fan_events    = deque(maxlen=MAXLEN)      # Whether the fan was ON for each reading (True/False)
early_trigger = deque(maxlen=MAXLEN)      # Whether the fan turned on due to fast rise (True/False)
timestamps    = deque(maxlen=MAXLEN)      # Reading index number (1, 2, 3...) used as x-axis
reading_index = [0]                       # Counter for reading number (in a list so threads can modify it)

# ================================================================
# STOP EVENT
# ================================================================
# stop_event is a flag shared between both threads.
# When you press Ctrl+C, the main thread sets stop_event,
# which signals the sensor thread to stop its loop cleanly.
# This ensures GPIO.cleanup() is only called AFTER the sensor
# thread has fully stopped — preventing the "unknown handle" crash.

stop_event = threading.Event()

# ================================================================
# TRACKING VARIABLES
# ================================================================

prev_temp    = None   # Previous temperature reading (for calculating temp_change)
temp_history = []     # Last 3 readings (for calculating avg_temp)

# ================================================================
# HELPER FUNCTION — TURN OFF ALL LEDs
# ================================================================
# Called before turning on the correct LED each reading.
# This ensures only one LED is on at a time.

def all_off():
    GPIO.output(GREEN,   GPIO.LOW)
    GPIO.output(YELLOW,  GPIO.LOW)
    GPIO.output(RED,     GPIO.LOW)
    GPIO.output(FAN_LED, GPIO.LOW)

# ================================================================
# INIT CSV FILE
# ================================================================
# Create a fresh CSV file at startup with column headers.
# Every reading is appended to this file by the sensor loop.
# This file is also used as a backup record of all readings.

with open("live_data.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "temp", "temp_change", "avg_temp", "prediction", "fan", "early_trigger"])

# ================================================================
# BAR COLOR LOGIC
# ================================================================
# Determines the color of each bar in the bottom graph chart.
# We use 4 colors to clearly show what caused each decision:
#
#   BLUE   — Fan ON because of fast rise (AI early trigger, temp < 35°C)
#            This is the smart proactive behaviour — most important to show
#   RED    — Fan ON because temperature reached 35°C (threshold trigger)
#   YELLOW — MEDIUM temperature, fan off
#   GREEN  — LOW temperature, fan off

def get_bar_color(prediction, fan_on, is_early):
    if fan_on and is_early:
        return "#1a73e8"   # Blue  — AI detected fast rise before overheating
    elif prediction == "HIGH":
        return "#dc3545"   # Red   — Temperature already reached 35°C
    elif prediction == "MEDIUM":
        return "#ffc107"   # Yellow — Warm but not critical
    else:
        return "#28a745"   # Green  — Safe temperature

# ================================================================
# SENSOR / AI THREAD
# ================================================================
# This function runs in a separate background thread.
# It loops every 2 seconds: read sensor → calculate features →
# run AI model → control LEDs → store data → save to CSV.
#
# It checks stop_event.is_set() at every step so it can exit
# quickly and cleanly when Ctrl+C is pressed.

def sensor_loop():
    global prev_temp, temp_history   # Access the tracking variables defined above

    while not stop_event.is_set():   # Keep looping until stop is requested

        # ── READ SENSOR ───────────────────────────────────────
        try:
            result = sensor.read()   # Read temperature and humidity from DHT11
        except Exception:
            break   # If GPIO was already cleaned up, exit the loop gracefully

        if result.is_valid():   # Only process if the sensor returned valid data
                                # DHT11 occasionally returns errors — skip those readings

            temperature = result.temperature   # Current temperature in Celsius
            humidity    = result.humidity      # Current humidity in percentage

            # ── CALCULATE TEMP_CHANGE ─────────────────────────
            # How much did the temperature change since the last reading?
            # This is the key feature for detecting fast temperature rises.
            # First reading has no previous value, so we default to 0.

            if prev_temp is None:
                temp_change = 0
            else:
                temp_change = temperature - prev_temp
            prev_temp = temperature   # Save for next reading

            # ── CALCULATE AVG_TEMP ────────────────────────────
            # Rolling average of the last 3 readings.
            # Smooths out noise and confirms the temperature trend.
            # Window of 3 matches exactly what was used during training —
            # this is critical, a different window would give the model
            # different avg_temp values than it was trained on.

            temp_history.append(temperature)
            if len(temp_history) > 3:
                temp_history.pop(0)   # Remove oldest reading, keep only last 3
            avg_temp = sum(temp_history) / len(temp_history)

            # ── RUN AI MODEL (INFERENCE) ──────────────────────
            # Package the 4 features into a DataFrame matching the
            # exact format the model was trained on (same column names,
            # same order). Then call model.predict() which runs the
            # input through all 200 decision trees and returns the
            # majority vote as a single label: LOW, MEDIUM, or HIGH.

            input_data = pd.DataFrame(
                [[temperature, humidity, temp_change, avg_temp]],
                columns=["temperature", "humidity", "temp_change", "avg_temp"]
            )
            prediction = model.predict(input_data)[0]   # [0] gets the string label from the result array

            # ── CONTROL LEDs AND FAN ──────────────────────────
            if not stop_event.is_set():   # Double-check stop wasn't requested while model was predicting

                all_off()      # Turn off all LEDs before turning the correct one on
                fan_on   = False
                is_early = False

                if prediction == "LOW":
                    GPIO.output(GREEN, GPIO.HIGH)    # Green LED on — safe temperature

                elif prediction == "MEDIUM":
                    GPIO.output(YELLOW, GPIO.HIGH)   # Yellow LED on — warm temperature

                elif prediction == "HIGH":
                    GPIO.output(RED,     GPIO.HIGH)  # Red LED on — high temperature or fast rise
                    GPIO.output(FAN_LED, GPIO.HIGH)  # Fan LED on — cooling needed
                    fan_on = True

                    # DETERMINE WHY THE FAN TURNED ON:
                    # If temp < 35, the threshold rule didn't trigger it.
                    # The only reason for HIGH at temp < 35 is the fast-rise rule.
                    # This is the "AI Early Trigger" — the proactive behaviour.
                    # If temp >= 35, the temperature threshold triggered it.
                    is_early = temperature < 35

                # ── STORE DATA FOR GRAPH ──────────────────────
                # Use the lock before writing to shared deques.
                # This prevents the graph thread from reading
                # half-written data at the same time.

                with data_lock:
                    reading_index[0] += 1   # Increment reading counter
                    idx = reading_index[0]
                    temps.append(temperature)
                    temp_changes.append(temp_change)
                    avg_temps.append(avg_temp)
                    predictions.append(prediction)
                    fan_events.append(fan_on)
                    early_trigger.append(is_early)
                    timestamps.append(idx)

                # ── SAVE TO CSV ───────────────────────────────
                with open("live_data.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([idx, temperature, temp_change, avg_temp, prediction, int(fan_on), int(is_early)])

                # ── PRINT TO TERMINAL ─────────────────────────
                # Show a readable summary of each reading.
                # Fan label changes based on WHY the fan turned on.

                if fan_on and is_early:
                    fan_label = "ON  (AI Early Trigger)"   # Fast rise detected
                elif fan_on:
                    fan_label = "ON  (High Temp)"           # Temperature reached 35°C
                else:
                    fan_label = "OFF"

                print(f"T={temperature}  Δ={temp_change:.2f}  Avg={avg_temp:.2f}  Fan={fan_label}  → {prediction}")

        # ── WAIT 2 SECONDS ────────────────────────────────────
        # Sleep in 20 x 0.1s increments instead of one time.sleep(2).
        # This way the loop checks stop_event every 0.1 seconds
        # and can exit quickly when Ctrl+C is pressed, rather than
        # being stuck sleeping for up to 2 full seconds.

        for _ in range(20):
            if stop_event.is_set():
                break
            time.sleep(0.1)

# ================================================================
# GRAPH FUNCTION (runs on main thread)
# ================================================================
# Matplotlib must run on the main thread — this is a technical
# requirement of most operating systems' GUI systems.
# So the graph loop runs here and the sensor runs in a thread.
#
# The graph has two charts stacked vertically:
#   TOP    — Temperature lines + fan trigger markers
#   BOTTOM — Color-coded AI decision bars per reading

def graph_loop():
    plt.ion()   # Enable interactive mode — allows graph to update without blocking

    # Create figure with 2 subplots. Top chart is 3x taller than bottom.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle("AI Smart Cooling System — Live Monitor", fontsize=14, fontweight='bold')

    # Background tint colors for the top chart (light versions of traffic light colors)
    PRED_BG_COLORS = {"LOW": "#d4edda", "MEDIUM": "#fff3cd", "HIGH": "#f8d7da"}

    while not stop_event.is_set():
        time.sleep(1)   # Refresh graph every 1 second

        # Read shared data safely using the lock
        with data_lock:
            if len(temps) < 2:
                continue   # Need at least 2 points to draw a line — wait for more data

            # Copy all deques to regular lists for plotting
            # (deques can't be directly indexed by matplotlib)
            t_list  = list(timestamps)
            te_list = list(temps)
            tc_list = list(temp_changes)
            av_list = list(avg_temps)
            pr_list = list(predictions)
            fa_list = list(fan_events)
            ei_list = list(early_trigger)

        # ── TOP CHART ──────────────────────────────────────────
        ax1.clear()   # Clear previous frame before redrawing

        # Draw background color bands based on AI prediction at each reading.
        # axvspan fills a vertical band between two x values with a color.
        # This makes it visually obvious when the AI state changes.
        for i in range(len(t_list) - 1):
            ax1.axvspan(t_list[i], t_list[i+1],
                        color=PRED_BG_COLORS.get(pr_list[i], "#ffffff"), alpha=0.3)

        # Draw the three temperature lines
        ax1.plot(t_list, te_list, color="#1a73e8", linewidth=2,   label="Temperature (°C)")
        ax1.plot(t_list, av_list, color="#6c757d", linewidth=1.5, linestyle="--", label="Avg Temp (°C)")
        ax1.plot(t_list, tc_list, color="#fd7e14", linewidth=1.5, linestyle=":",  label="Temp Change (°C)")

        # Draw vertical dashed lines wherever the fan turned on.
        # Blue lines = AI early trigger (fast rise, temp < 35°C)
        # Red lines  = Temperature threshold trigger (temp >= 35°C)
        # Only add each label once to avoid duplicate legend entries.
        early_drawn = False
        high_drawn  = False
        for i, (fan, early) in enumerate(zip(fa_list, ei_list)):
            if fan and early:
                label = "Fan ON — AI Early Trigger" if not early_drawn else ""
                ax1.axvline(x=t_list[i], color="#1a73e8", linewidth=1.5,
                            linestyle="--", alpha=0.9, label=label)
                early_drawn = True
            elif fan and not early:
                label = "Fan ON — High Temp" if not high_drawn else ""
                ax1.axvline(x=t_list[i], color="#dc3545", linewidth=1.5,
                            linestyle="--", alpha=0.9, label=label)
                high_drawn = True

        # Horizontal threshold lines showing the 30°C and 35°C boundaries
        ax1.axhline(y=30, color="#ffc107", linewidth=1, linestyle=":", alpha=0.7, label="30°C threshold")
        ax1.axhline(y=35, color="#dc3545", linewidth=1, linestyle=":", alpha=0.7, label="35°C threshold")

        ax1.set_ylabel("Temperature / Change (°C)", fontsize=10)
        ax1.set_title("Temperature & AI Decision", fontsize=11)
        ax1.legend(loc="upper left", fontsize=8, ncol=3)
        ax1.grid(True, alpha=0.3)

        # Live annotation box at bottom-left showing the latest reading summary.
        # Color changes based on current state to draw attention.
        last_pred  = pr_list[-1]
        last_temp  = te_list[-1]
        last_fan   = fa_list[-1]
        last_early = ei_list[-1]

        if last_fan and last_early:
            fan_status = "FAN: ON — AI Early Trigger"
            box_color  = "#1a73e8"   # Blue box for early trigger
        elif last_fan:
            fan_status = "FAN: ON — High Temp"
            box_color  = "#dc3545"   # Red box for threshold trigger
        else:
            fan_status = "FAN: OFF"
            box_color  = "#28a745" if last_pred == "LOW" else "#6c757d"

        ax1.annotate(
            f"  Latest: {last_temp}°C  |  AI: {last_pred}  |  {fan_status}",
            xy=(0.01, 0.02), xycoords='axes fraction',   # Position: bottom-left of chart
            fontsize=9, color=box_color, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=box_color, alpha=0.8)
        )

        # ── BOTTOM CHART ───────────────────────────────────────
        ax2.clear()

        # Convert prediction labels to numbers for the bar chart height
        # LOW=1, MEDIUM=2, HIGH=3 — so bars sit at different heights
        pred_num   = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        p_list     = [pred_num[p] for p in pr_list]

        # Get the correct color for each bar using get_bar_color()
        # This is where the 4-color system (green/yellow/red/blue) is applied
        bar_colors = [get_bar_color(p, f, e) for p, f, e in zip(pr_list, fa_list, ei_list)]

        ax2.bar(t_list, p_list, color=bar_colors, width=1.0, alpha=0.9)
        ax2.set_yticks([1, 2, 3])
        ax2.set_yticklabels(["LOW", "MEDIUM", "HIGH"], fontsize=9)
        ax2.set_ylabel("AI Decision", fontsize=10)
        ax2.set_xlabel("Reading Number", fontsize=10)
        ax2.set_title("AI Prediction Over Time", fontsize=11)
        ax2.grid(True, axis='y', alpha=0.3)

        # Legend explaining what each color means
        patches = [
            mpatches.Patch(color="#28a745", label="LOW — Fan OFF"),
            mpatches.Patch(color="#ffc107", label="MEDIUM — Fan OFF"),
            mpatches.Patch(color="#1a73e8", label="AI Early Trigger — Fan ON (fast rise, temp < 35°C)"),
            mpatches.Patch(color="#dc3545", label="HIGH — Fan ON (temp >= 35°C)"),
        ]
        ax2.legend(handles=patches, loc="upper left", fontsize=8, ncol=2)

        plt.tight_layout(rect=[0, 0, 1, 0.95])   # Adjust layout to fit title and charts neatly
        plt.pause(0.1)   # Pause briefly to render the updated graph (required for interactive mode)

# ================================================================
# MAIN ENTRY POINT
# ================================================================
# Python runs this block when you execute the script directly.
# It starts the sensor thread and then runs the graph on the
# main thread. The two run in parallel until Ctrl+C is pressed.

if __name__ == "__main__":
    print("Starting AI Smart Cooling System...")
    print("Graph will appear after first 2 readings.")
    print("Press Ctrl+C to stop.\n")

    # Start the sensor loop in a background thread.
    # daemon=True means this thread will automatically stop
    # if the main program exits, even without stop_event.
    sensor_thread = threading.Thread(target=sensor_loop, daemon=True)
    sensor_thread.start()

    try:
        graph_loop()   # Run the graph on the main thread (matplotlib requirement)

    except KeyboardInterrupt:
        # Ctrl+C was pressed — shut everything down cleanly in the correct order:
        print("\nShutting down...")
        stop_event.set()              # 1. Signal sensor thread to stop its loop
        sensor_thread.join(timeout=3) # 2. Wait up to 3 seconds for sensor thread to fully finish
        all_off()                     # 3. Turn off all LEDs and fan
        GPIO.cleanup()                # 4. Release all GPIO pins (safe because sensor thread is done)
        plt.close()                   # 5. Close the graph window
        print("Done.")