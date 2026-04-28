import time
import RPi.GPIO as GPIO

GREEN_PIN = 17
YELLOW_PIN = 27
RED_PIN = 22

GPIO.setwarnings(True)
GPIO.setmode(GPIO.BCM)

GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(YELLOW_PIN, GPIO.OUT)
GPIO.setup(RED_PIN, GPIO.OUT)

def all_off():
    GPIO.output(GREEN_PIN, GPIO.LOW)
    GPIO.output(YELLOW_PIN, GPIO.LOW)
    GPIO.output(RED_PIN, GPIO.LOW)

try:
    while True:
        all_off()
        GPIO.output(GREEN_PIN, GPIO.HIGH)
        print("GREEN ON")
        time.sleep(1)

        all_off()
        GPIO.output(YELLOW_PIN, GPIO.HIGH)
        print("YELLOW ON")
        time.sleep(1)

        all_off()
        GPIO.output(RED_PIN, GPIO.HIGH)
        print("RED ON")
        time.sleep(1)

except KeyboardInterrupt:
    print("Cleanup")
    all_off()
    GPIO.cleanup()