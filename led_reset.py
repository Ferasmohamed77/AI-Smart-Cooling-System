import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
for pin in [17, 27, 22]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)
GPIO.cleanup()

print("All LEDs OFF")