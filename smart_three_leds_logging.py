import time
import RPi.GPIO as GPIO
import dht11
from google.oauth2 import service_account
from googleapiclient.discovery import build

DHT_PIN = 4
GREEN_PIN = 17
YELLOW_PIN = 27
RED_PIN = 22

GPIO.setwarnings(True)
GPIO.setmode(GPIO.BCM)

sensor = dht11.DHT11(pin=DHT_PIN)

GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(YELLOW_PIN, GPIO.OUT)
GPIO.setup(RED_PIN, GPIO.OUT)

def all_off():
    GPIO.output(GREEN_PIN, GPIO.LOW)
    GPIO.output(YELLOW_PIN, GPIO.LOW)
    GPIO.output(RED_PIN, GPIO.LOW)

def get_service():
    creds = service_account.Credentials.from_service_account_file(
        "mydata.json",
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    return build("sheets", "v4", credentials=creds)

def log_to_gsheet(service, spreadsheet_id, values):
    try:
        body = {"values": [values]}
        service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range="A:D",
            valueInputOption="USER_ENTERED",
            body=body
        ).execute()
        print("Data logged.")
    except Exception as e:
        print(f"Error: {e}")

def decide_action(temp):
    if temp > 35:
        return "HIGH"
    elif temp >= 30:
        return "MEDIUM"
    else:
        return "LOW"

def apply_action(action):
    all_off()

    if action == "LOW":
        GPIO.output(GREEN_PIN, GPIO.HIGH)
    elif action == "MEDIUM":
        GPIO.output(YELLOW_PIN, GPIO.HIGH)
    elif action == "HIGH":
        GPIO.output(RED_PIN, GPIO.HIGH)

spreadsheet_id = "11fqrKUUVjjGClogmOlw4MpB23IjDMQLTEM1_-mJohUw"
service = get_service()

try:
    while True:
        result = sensor.read()

        if result.is_valid():
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            temp = result.temperature
            hum = result.humidity

            action = decide_action(temp)
            apply_action(action)

            log_to_gsheet(service, spreadsheet_id, [timestamp, temp, hum, action])

            print(f"Time={timestamp}, Temp={temp}, Humidity={hum}, Action={action}")
            time.sleep(2)
        else:
            print("Invalid reading")
            time.sleep(0.5)

except KeyboardInterrupt:
    print("Cleanup")
    all_off()
    GPIO.cleanup()