import time
import signal
import sys
import os
import joblib
import pandas as pd
import RPi.GPIO as GPIO
import dht11

DHT_PIN = 4
GREEN = 17
YELLOW = 27
RED = 22

GPIO.setwarnings(True)
GPIO.setmode(GPIO.BCM)

sensor = dht11.DHT11(pin=DHT_PIN)
GPIO.setup(GREEN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(YELLOW, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(RED, GPIO.OUT, initial=GPIO.LOW)

model = joblib.load("model_3class.pkl")

def all_off():
    GPIO.output(GREEN, GPIO.LOW)
    GPIO.output(YELLOW, GPIO.LOW)
    GPIO.output(RED, GPIO.LOW)

def apply_prediction(pred):
    all_off()
    if pred == 0:
        GPIO.output(GREEN, GPIO.HIGH)
        return "LOW", "GREEN", "[█░░]", "SAFE"
    elif pred == 1:
        GPIO.output(YELLOW, GPIO.HIGH)
        return "MEDIUM", "YELLOW", "[██░]", "CAUTION"
    else:
        GPIO.output(RED, GPIO.HIGH)
        return "HIGH", "RED", "[███]", "ALERT"

def cleanup_and_exit(*args):
    try:
        all_off()
        GPIO.cleanup()
    except:
        pass
    print("\nSystem stopped. LEDs OFF.")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)

def clear_screen():
    os.system("clear")

while True:
    result = sensor.read()

    if result.is_valid():
        temp = result.temperature
        hum = result.humidity

        X_live = pd.DataFrame([{
            "temperature": temp,
            "humidity": hum
        }])

        pred = model.predict(X_live)[0]
        action, color, bar, level = apply_prediction(pred)

        clear_screen()
        print("================================================")
        print("         AI SMART COOLING DECISION SYSTEM       ")
        print("================================================")
        print(f" Live Temperature  : {temp:.1f} °C")
        print(f" Live Humidity     : {hum:.1f} %")
        print(f" AI Classification : {action}")
        print(f" Alert Level       : {level}")
        print(f" Output Indicator  : {color}")
        print(f" Cooling Demand    : {bar}")
        print("================================================")
        print(" Green  = LOW      Yellow = MEDIUM    Red = HIGH")
        print("================================================")
        print(" Press Ctrl+C to stop")
        print("================================================")
        time.sleep(2)

    else:
        clear_screen()
        print("Sensor reading invalid...")
        time.sleep(0.5)