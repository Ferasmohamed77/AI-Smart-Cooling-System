import time
import RPi.GPIO as GPIO
import dht11

GPIO.setwarnings(True)
GPIO.setmode(GPIO.BCM)

sensor = dht11.DHT11(pin=4)

try:
    while True:
        result = sensor.read()
        if result.is_valid():
            print("Temp:", result.temperature, "C   Humidity:", result.humidity, "%")
            time.sleep(2)
        else:
            print("Invalid reading")
            time.sleep(0.5)

except KeyboardInterrupt:
    GPIO.cleanup()