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
    writer.writerow(["temp", "temp_change", "avg_temp", "prediction"])

# ================= MAIN =================
try:
    while True:
        result = sensor.read()

        if result.is_valid():
            temperature = result.temperature
            humidity    = result.humidity

            # ---------- temp_change ----------
            if prev_temp is None:
                temp_change = 0
            else:
                temp_change = temperature - prev_temp
            prev_temp = temperature

            # ---------- avg_temp ----------
            # ✅ FIX: Window size changed from 10 → 3 to match training data
            temp_history.append(temperature)
            if len(temp_history) > 3:
                temp_history.pop(0)
            avg_temp = sum(temp_history) / len(temp_history)

            # ---------- AI INPUT ----------
            input_data = pd.DataFrame(
                [[temperature, humidity, temp_change, avg_temp]],
                columns=["temperature", "humidity", "temp_change", "avg_temp"]
            )
            prediction = model.predict(input_data)[0]

            # ---------- LED CONTROL ----------
            all_off()

            if prediction == "LOW":
                GPIO.output(GREEN, GPIO.HIGH)

            elif prediction == "MEDIUM":
                GPIO.output(YELLOW, GPIO.HIGH)

            elif prediction == "HIGH":
                GPIO.output(RED,     GPIO.HIGH)
                GPIO.output(FAN_LED, GPIO.HIGH)  # Fan turns on for HIGH

            # ---------- SAVE FOR GRAPH ----------
            with open("live_data.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([temperature, temp_change, avg_temp, prediction])

            print(f"T={temperature}  Δ={temp_change:.2f}  Avg={avg_temp:.2f}  → {prediction}")

        time.sleep(2)

except KeyboardInterrupt:
    print("Stopped.")
    all_off()
    GPIO.cleanup()