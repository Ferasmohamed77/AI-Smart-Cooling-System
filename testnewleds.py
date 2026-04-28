import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

GPIO.setup(17, GPIO.OUT)
GPIO.setup(27, GPIO.OUT)
GPIO.setup(22, GPIO.OUT)

while True:
    GPIO.output(17, 1)
    time.sleep(1)
    GPIO.output(17, 0)

    GPIO.output(27, 1)
    time.sleep(1)
    GPIO.output(27, 0)

    GPIO.output(22, 1)
    time.sleep(1)
    GPIO.output(22, 0)