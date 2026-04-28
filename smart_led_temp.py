import time
import RPi.GPIO as GPIO
import dht11

DHT_PIN = 4
LED_PIN = 17
TEMP_THRESHOLD = 25   # change this if needed

GPIO.setwarnings(True)
GPIO.setmode(GPIO.BCM)

sensor = dht11.DHT11(pin=DHT_PIN)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)

try:
    while True:
        result = sensor.read()

        if result.is_valid():
            temp = result.temperature
            hum = result.humidity

            print(f"Temp: {temp} C   Humidity: {hum} %")

            if temp >= TEMP_THRESHOLD:
                GPIO.output(LED_PIN, GPIO.HIGH)
                print("LED ON - temperature is high")
            else:
                GPIO.output(LED_PIN, GPIO.LOW)
                print("LED OFF - temperature is normal")

            time.sleep(2)
        else:
            print("Invalid reading")
            time.sleep(0.5)

except KeyboardInterrupt:
    print("Cleanup")
    GPIO.cleanup()