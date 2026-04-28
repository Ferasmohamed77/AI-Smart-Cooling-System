import time
import RPi.GPIO as GPIO

LED_PIN = 17

GPIO.setwarnings(True)
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

try:
    while True:
        GPIO.output(LED_PIN, GPIO.HIGH)
        print("LED ON")
        time.sleep(1)

        GPIO.output(LED_PIN, GPIO.LOW)
        print("LED OFF")
        time.sleep(1)

except KeyboardInterrupt:
    print("Cleanup")
    GPIO.cleanup()