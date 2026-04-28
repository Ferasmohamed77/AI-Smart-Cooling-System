# -*- coding: utf-8 -*-
# Author: Noorul Ghousiah Binti Noordeen Sahib (https://github.com/noorulghousiah)
# Sensor reading app that logs data to Google Sheets

# Import necessary libraries
import json
import time
import RPi.GPIO as GPIO
import dht11
from google.oauth2 import service_account
from googleapiclient.discovery import build

# GPIO setup
GPIO.setwarnings(True)
GPIO.setmode(GPIO.BCM)
sensor = dht11.DHT11(pin=4)

# Function to authenticate and connect to Google Sheets API
def get_service():
    creds = service_account.Credentials.from_service_account_file(
        "mydata.json",
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    return build('sheets', 'v4', credentials=creds)

# Function to log sensor data to Google Sheets
def log_to_gsheet(service, spreadsheet_id, values):
    try:
        body = {"values": [values]}

        service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range="A:A",
            valueInputOption="USER_ENTERED",
            body=body
        ).execute()

        print("Data logged.")

    except Exception as e:
        if "RATE_LIMIT_EXCEEDED" in str(e):
            print("Rate limit exceeded. Retrying...")
            time.sleep(10)
            log_to_gsheet(service, spreadsheet_id, values)
        else:
            print(f"Error: {e}")

# Main program
spreadsheet_id = "11fqrKUUVjjGClogmOlw4MpB23IjDMQLTEM1_-mJohUw"
service = get_service()

try:
    while True:
        result = sensor.read()

        if result.is_valid():
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            temperature = result.temperature
            humidity = result.humidity

            # Log to Google Sheets
            log_to_gsheet(service, spreadsheet_id, [timestamp, temperature, humidity])

            # Print to console
            print(f"Time: {timestamp}, Temp: {temperature} C, Humidity: {humidity} %")

            time.sleep(2)
        else:
            time.sleep(0.1)

except KeyboardInterrupt:
    print("Cleanup")
    GPIO.cleanup()