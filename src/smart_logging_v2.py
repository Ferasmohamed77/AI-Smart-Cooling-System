#!/usr/bin/env python3

# ================================================================
# SMART LOGGING V2 — Temperature Data Collection Script
# ================================================================
# PURPOSE:
#   This script is the FIRST step in the AI pipeline.
#   It reads temperature and humidity from the DHT11 sensor
#   every 2 seconds, calculates useful AI features (temp_change
#   and avg_temp), and logs everything to Google Sheets.
#   The data collected here is later used to train the AI model.
# ================================================================

import time                              # Used for timestamps and sleep delays
import RPi.GPIO as GPIO                  # Controls the GPIO pins on the Raspberry Pi
import dht11                             # Library to read the DHT11 temperature/humidity sensor
from google.oauth2 import service_account   # Handles Google API authentication using a service account key file
from googleapiclient.discovery import build # Builds the Google Sheets API connection

# ================================================================
# GPIO SETUP
# ================================================================

GPIO.setwarnings(False)       # Suppresses GPIO warning messages (e.g. pin already in use)
GPIO.setmode(GPIO.BCM)        # Use BCM pin numbering (the GPIO number, not physical pin number)

sensor = dht11.DHT11(pin=4)   # Connect the DHT11 sensor to GPIO pin 4

# ================================================================
# GOOGLE SHEETS SETUP
# ================================================================
# This function connects to the Google Sheets API using a
# service account credentials file (mydata.json).
# A service account is like a special Google account for programs,
# it lets the script write to Google Sheets automatically
# without needing a human to log in.

def get_service():
    creds = service_account.Credentials.from_service_account_file(
        "mydata.json",               # Your downloaded Google service account key file
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",  # Permission to read/write Sheets
            "https://www.googleapis.com/auth/drive"          # Permission to access Drive (needed for Sheets)
        ]
    )
    return build('sheets', 'v4', credentials=creds)  # Returns a Sheets API service object

# ================================================================
# LOG TO GOOGLE SHEETS FUNCTION
# ================================================================
# This function takes one row of data (a list of values) and
# appends it as a new row at the bottom of the Google Sheet.
# "append" means it always adds to the end, never overwrites.

def log_to_gsheet(service, spreadsheet_id, values):
    body = {"values": [values]}              # Wrap values in the format Google Sheets API expects
    service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,        # The unique ID of your Google Sheet (from the URL)
        range="A:F",                         # Columns A to F (timestamp, temp, humidity, temp_change, avg_temp, action)
        valueInputOption="USER_ENTERED",     # Treat values as if a user typed them (handles numbers correctly)
        body=body
    ).execute()                              # Actually send the request to Google Sheets

# ================================================================
# CONFIGURATION
# ================================================================

spreadsheet_id = "11fqrKUUVjjGClogmOlw4MpB23IjDMQLTEM1_-mJohUw"  # Your Google Sheet ID (from the sheet URL)
service = get_service()   # Connect to Google Sheets once at the start (reused for every log)

# ================================================================
# TRACKING VARIABLES
# ================================================================

prev_temp    = None   # Stores the previous temperature reading to calculate how much it changed
temp_history = []     # Stores the last 3 temperature readings to calculate the rolling average

# ================================================================
# MAIN LOOP
# ================================================================
# This loop runs forever (until you press Ctrl+C).
# Every 2 seconds it reads the sensor, calculates features,
# labels the reading, prints it, and logs it to Google Sheets.

try:
    while True:
        result = sensor.read()   # Read temperature and humidity from the DHT11 sensor

        if result.is_valid():    # Only process the reading if the sensor returned valid data
                                 # (DHT11 sometimes returns errors, this skips those)

            # ── GET SENSOR VALUES ─────────────────────────────
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # Current date and time as a string
            temp      = result.temperature                   # Current temperature in Celsius
            hum       = result.humidity                      # Current humidity in percentage

            # ── CALCULATE TEMP CHANGE (Feature 1) ────────────
            # temp_change tells the AI how fast the temperature is rising or falling.
            # It is calculated as: current temp minus previous temp.
            # Positive value = temperature is rising.
            # Negative value = temperature is falling.
            # Zero = temperature is stable.
            # On the very first reading, there is no previous temp, so we set it to 0.

            if prev_temp is None:
                temp_change = 0         # First reading — no previous value to compare
            else:
                temp_change = temp - prev_temp   # How much did temperature change since last reading?

            # ── CALCULATE AVERAGE TEMPERATURE (Feature 2) ────
            # avg_temp is the rolling average of the last 3 readings.
            # It smooths out sudden spikes and gives the AI a sense of
            # the recent temperature trend, not just the current snapshot.
            # Window of 3 is kept small so the average reacts quickly to changes.

            temp_history.append(temp)        # Add current temp to history list
            if len(temp_history) > 3:        # Keep only the last 3 readings
                temp_history.pop(0)          # Remove the oldest reading when list exceeds 3
            avg_temp = sum(temp_history) / len(temp_history)   # Calculate the average

            prev_temp = temp   # Save current temp so next reading can calculate temp_change

            # ── LABEL THE READING (Action) ────────────────────
            # This assigns a label to each reading based on simple temperature thresholds.
            # These labels are stored in Google Sheets alongside the raw data.
            # Note: train_model_v2.py will RE-LABEL the data using smarter logic
            # that also considers temp_change. The labels here are just for reference.

            if temp < 30:
                action = "LOW"       # Safe temperature — green LED
            elif temp < 35:
                action = "MEDIUM"    # Warm temperature — yellow LED
            else:
                action = "HIGH"      # Hot temperature — red LED + fan

            # ── PRINT TO TERMINAL ─────────────────────────────
            # Shows a live summary of each reading in the terminal so you can
            # monitor what is being logged in real time.

            print(f"{timestamp} | Temp={temp}C Hum={hum}% Δ={temp_change:.2f} Avg={avg_temp:.2f} → {action}")

            # ── LOG TO GOOGLE SHEETS ──────────────────────────
            # Sends one row of data to Google Sheets:
            # Column A: timestamp
            # Column B: temperature
            # Column C: humidity
            # Column D: temp_change
            # Column E: avg_temp
            # Column F: action (label)

            log_to_gsheet(service, spreadsheet_id,
                          [timestamp, temp, hum, temp_change, avg_temp, action])

        time.sleep(2)   # Wait 2 seconds before the next reading
                        # DHT11 needs at least 1 second between readings to be reliable

# ================================================================
# GRACEFUL SHUTDOWN
# ================================================================
# When you press Ctrl+C, the script stops cleanly.
# GPIO.cleanup() releases all GPIO pins so they are not left
# in an active state, which could cause issues next time.

except KeyboardInterrupt:
    GPIO.cleanup()   # Release all GPIO pins