# AI Smart Cooling System with Raspberry Pi 5

An AI-powered smart cooling system that uses a trained Random Forest 
machine learning model to detect rapid temperature rises and activate 
a cooling fan before overheating occurs — proactive cooling, not reactive.

## Hardware
- Raspberry Pi 5
- DHT11 Temperature & Humidity Sensor (GPIO 4)
- Green LED — LOW temperature indicator (GPIO 17)
- Yellow LED — MEDIUM temperature indicator (GPIO 27)
- Red LED — HIGH temperature indicator (GPIO 22)
- Fan LED — Cooling fan output (GPIO 24)

## Pipeline
| Step | Script | Purpose |
|------|--------|---------|
| 1 | `smart_logging_v2.py` | Reads sensor every 2 seconds and logs to Google Sheets |
| 2 | `train_model_v2.py` | Trains Random Forest model on collected data |
| 3 | `ai_control_v2.py` | Runs live AI predictions and controls LEDs and fan |

## Key Feature — Proactive Fan Trigger
The fan activates when the AI detects a **fast temperature rise**
(temp_change ≥ 0.5°C per reading) even before the temperature
reaches 35°C. A conventional thermostat would only react after the
threshold is crossed.

## Temperature Thresholds
| State | Condition | LED | Fan |
|-------|-----------|-----|-----|
| LOW | temp < 30°C, stable | Green | OFF |
| MEDIUM | 30°C ≤ temp < 35°C, stable | Yellow | OFF |
| HIGH (AI Early Trigger) | Any temp, fast rise detected | Red | ON |
| HIGH (Threshold) | temp ≥ 35°C | Red | ON |

## Installation
```bash
pip install RPi.GPIO dht11 scikit-learn joblib pandas matplotlib
pip install google-auth google-auth-httplib2 google-api-python-client
```

## Training Results
- Training rows: 1160
- Model accuracy: 100%
- temp_change importance: 0.16 (fast-rise detection confirmed)
- All 5 sanity checks passed ✅

## Built With
- Python 3.13
- scikit-learn Random Forest Classifier
- Raspberry Pi OS 64-bit
- Google Sheets API
- Matplotlib (live graph)
