import time
import joblib
import RPi.GPIO as GPIO
import dht11

DHT_PIN = 4
LED_PIN = 17

GPIO.setwarnings(True)
GPIO.setmode(GPIO.BCM)

sensor = dht11.DHT11(pin=DHT_PIN)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)

model = joblib.load("model.pkl")

try:
    while True:
        result = sensor.read()

        if result.is_valid():
            temperature = result.temperature
            humidity = result.humidity

            prediction = model.predict([[temperature, humidity]])[0]

            if prediction == 1:
                GPIO.output(LED_PIN, GPIO.HIGH)
                action = "LED_ON"
            else:
                GPIO.output(LED_PIN, GPIO.LOW)
                action = "LED_OFF"

            print(
                f"Temp={temperature} C, Humidity={humidity} %, "
                f"Predicted Action={action}"
            )

            time.sleep(2)
        else:
            print("Invalid reading")
            time.sleep(0.5)

except KeyboardInterrupt:
    print("Cleanup")
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()