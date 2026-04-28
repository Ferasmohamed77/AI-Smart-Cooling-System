import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)

pins = [17, 27, 22, 24]

for p in pins:
    GPIO.setup(p, GPIO.OUT)
    GPIO.output(p, GPIO.LOW)

GPIO.cleanup()

print("All LEDs OFF")