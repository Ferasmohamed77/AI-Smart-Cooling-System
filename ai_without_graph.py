# ================================================================
# AI CONTROL V2 — Live AI Control (No Graph)
# ================================================================
# PURPOSE:
#   Loads the trained AI model, reads live sensor data every
#   2 seconds, controls the LEDs and fan based on the AI
#   prediction, and saves all readings to a CSV file.
# ================================================================

import RPi.GPIO as GPIO
import time
import dht11
import joblib
import pandas as pd
import csv

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
    writer.writerow(["index", "temp", "temp_change", "avg_temp", "prediction", "fan", "early_trigger"])

# ================= MAIN =================
print("Starting AI Smart Cooling System...")
print("Press Ctrl+C to stop.\n")

reading_index = 0

try:
    while True:
        result = sensor.read()

        if result.is_valid():
            temperature = result.temperature
            humidity    = result.humidity

            # temp_change
            if prev_temp is None:
                temp_change = 0
            else:
                temp_change = temperature - prev_temp
            prev_temp = temperature

            # avg_temp (window of 3 — matches training)
            temp_history.append(temperature)
            if len(temp_history) > 3:
                temp_history.pop(0)
            avg_temp = sum(temp_history) / len(temp_history)

            # AI prediction
            input_data = pd.DataFrame(
                [[temperature, humidity, temp_change, avg_temp]],
                columns=["temperature", "humidity", "temp_change", "avg_temp"]
            )
            prediction = model.predict(input_data)[0]

            # GPIO control
            all_off()
            fan_on   = False
            is_early = False

            if prediction == "LOW":
                GPIO.output(GREEN, GPIO.HIGH)
            elif prediction == "MEDIUM":
                GPIO.output(YELLOW, GPIO.HIGH)
            elif prediction == "HIGH":
                GPIO.output(RED,     GPIO.HIGH)
                GPIO.output(FAN_LED, GPIO.HIGH)
                fan_on   = True
                is_early = temperature < 35 and temp_change > 0   # True = AI early trigger (fast rise), False = temp threshold

            # Save to CSV
            reading_index += 1
            with open("live_data.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([reading_index, temperature, temp_change, avg_temp, prediction, int(fan_on), int(is_early)])

            # Print to terminal
            if fan_on and is_early:
                fan_label = "ON  (AI Early Trigger)"
            elif fan_on:
                fan_label = "ON  (High Temp)"
            else:
                fan_label = "OFF"

            print(f"T={temperature}  Δ={temp_change:.2f}  Avg={avg_temp:.2f}  Fan={fan_label}  → {prediction}")

        time.sleep(2)

except KeyboardInterrupt:
    print("\nShutting down...")
    all_off()
    GPIO.cleanup()
    print("Done.")