import time
import RPi.GPIO as GPIO
import dht11
from google.oauth2 import service_account
from googleapiclient.discovery import build

DHT_PIN = 4
LED_PIN = 17
TEMP_THRESHOLD = 28

GPIO.setwarnings(True)
GPIO.setmode(GPIO.BCM)

sensor = dht11.DHT11(pin=DHT_PIN)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)

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
    if temp >= TEMP_THRESHOLD:
        return "LED_ON"
    return "LED_OFF"

def apply_action(action):
    if action == "LED_ON":
        GPIO.output(LED_PIN, GPIO.HIGH)
    else:
        GPIO.output(LED_PIN, GPIO.LOW)

spreadsheet_id = "11fqrKUUVjjGClogmOlw4MpB23IjDMQLTEM1_-mJohUw"
service = get_service()

try:
    while True:
        result = sensor.read()

        if result.is_valid():
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            temperature = result.temperature
            humidity = result.humidity

            action = decide_action(temperature)
            apply_action(action)

            log_to_gsheet(service, spreadsheet_id, [
                timestamp,
                temperature,
                humidity,
                action
            ])

            print(
                f"Time: {timestamp}, Temp: {temperature} C, "
                f"Humidity: {humidity} %, Action: {action}"
            )

            time.sleep(2)
        else:
            print("Invalid reading")
            time.sleep(0.5)

except KeyboardInterrupt:
    print("Cleanup")
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()