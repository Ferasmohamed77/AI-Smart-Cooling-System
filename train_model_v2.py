# ================================================================
# TRAIN MODEL V2 — AI Model Training Script
# ================================================================
# PURPOSE:
#   This script is the SECOND step in the AI pipeline.
#   It takes the data collected by smart_logging_v2.py,
#   applies smart labeling logic, and trains a Random Forest
#   AI model to classify temperature readings as LOW, MEDIUM,
#   or HIGH — including detecting fast temperature rises early.
#   The trained model is saved as model_v2.pkl and used by
#   ai_control_v2.py to make real-time decisions.
# ================================================================

import pandas as pd                              # Used to load and manipulate the CSV data as a table
from sklearn.ensemble import RandomForestClassifier  # The AI model we are training
import joblib                                    # Used to save the trained model to a file

# ================================================================
# STEP 1 — LOAD THE DATA
# ================================================================
# Load the CSV file downloaded from Google Sheets.
# This file contains all the sensor readings collected by
# smart_logging_v2.py: timestamp, temperature, humidity,
# temp_change, avg_temp, and the original action label.

df = pd.read_csv("Raspi_data - Sheet1 (4).csv")

# Remove any accidental spaces from column names
# (Google Sheets sometimes adds spaces when exporting)
df.columns = df.columns.str.strip()

print(f"Total rows loaded: {len(df)}")   # Show how many readings were loaded

# ================================================================
# STEP 2 — SMART LABELING FUNCTION
# ================================================================
# This function decides the correct label (LOW, MEDIUM, HIGH)
# for each row of data. This is the most important part of the
# training script because the model learns EXACTLY what we teach
# it here. If the labels are wrong, the model learns the wrong thing.
#
# KEY DESIGN DECISION:
#   The fast-rise check comes FIRST, before the temperature check.
#   This is intentional — it means a fast rise at ANY temperature
#   gets labeled HIGH, even if the temperature is still low.
#   This teaches the model to be proactive, not reactive.
#
# PARAMETERS:
#   temp        — current temperature reading
#   temp_change — how much temperature changed since last reading
#   avg_temp    — rolling average of last 3 readings

def get_action(temp, temp_change, avg_temp):

    # FAST RISE CHECK (checked FIRST — highest priority)
    # If temperature is rising by 0.5°C or more per reading
    # AND the baseline average is already above 25°C,
    # label it HIGH immediately — the fan should turn on early.
    # This is the "proactive" AI behaviour that makes this system
    # smarter than a simple thermostat.
    if temp_change >= 0.5 and avg_temp > 25:
        return "HIGH"

    # TEMPERATURE THRESHOLD CHECKS (checked after fast-rise)
    # These are the normal rules based on current temperature.
    if temp >= 35:
        return "HIGH"       # Already very hot — fan must be on
    elif temp >= 30:
        return "MEDIUM"     # Getting warm — yellow LED warning
    else:
        return "LOW"        # Safe temperature — green LED

# Apply the labeling function to every row in the dataset.
# lambda means: for each row, call get_action() with its values.
# axis=1 means: apply the function row by row (not column by column).
df["action"] = df.apply(
    lambda row: get_action(row["temperature"], row["temp_change"], row["avg_temp"]),
    axis=1
)

# Show how many rows got each label so we can verify the distribution
print("\nLabel distribution in training data:")
print(df["action"].value_counts())
# We want a reasonable number of all three labels.
# If HIGH is 0, the model will never predict HIGH.
# If LOW is 0, the green LED will never turn on.

# ================================================================
# STEP 3 — DEFINE FEATURES AND TARGET
# ================================================================
# Features (X) = the input values the model uses to make decisions.
# These are the 4 columns the model will look at for each reading.
#
# temperature  — the current temperature (strongest signal)
# humidity     — current humidity (minor influence)
# temp_change  — how fast temperature is changing (key for early detection)
# avg_temp     — recent average temperature (confirms trend, filters noise)
#
# Target (y) = the answer we want the model to predict (LOW/MEDIUM/HIGH).
# This is what the model is being trained to output.

X = df[["temperature", "humidity", "temp_change", "avg_temp"]]   # Input features
y = df["action"]                                                   # Target labels

# ================================================================
# STEP 4 — TRAIN THE RANDOM FOREST MODEL
# ================================================================
# A Random Forest builds 200 decision trees, each trained on a
# slightly different random sample of the data.
# When predicting, all 200 trees vote and the majority wins.
# This makes it more accurate and reliable than a single tree.
#
# n_estimators=200:
#   Build 200 trees. More trees = more reliable predictions.
#   We increased from 100 to 200 because our fast-rise examples
#   are rare and we want the model to learn them consistently.
#
# class_weight="balanced":
#   CRITICAL for our dataset. We only have ~17 fast-rise examples
#   at LOW/MEDIUM temperatures out of 1160 total rows. Without this,
#   the model would ignore those rare examples because being wrong
#   about 17 rows barely affects the overall accuracy score.
#   "balanced" automatically increases the importance of rare classes
#   so the model takes fast-rise examples as seriously as common ones.
#
# random_state=42:
#   Makes the training reproducible — running the script twice
#   will produce the exact same model. Without this, results
#   vary slightly each run due to randomness in tree building.

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X, y)   # Train the model — this is where the learning happens.
                  # The model analyzes all 1160 rows and builds 200 trees.

# ================================================================
# STEP 5 — CHECK ACCURACY
# ================================================================
# model.score() tests the model on the same training data and
# returns the percentage of correct predictions (0.0 to 1.0).
# 1.0 means 100% accuracy on training data, which is expected
# for a Random Forest — it memorizes the training examples well.
# The real test is how it performs on live sensor data.

acc = model.score(X, y)
print(f"\nTraining Accuracy: {acc:.4f}")

# ================================================================
# STEP 6 — FEATURE IMPORTANCE
# ================================================================
# After training, we can ask the model: which features did you
# actually rely on when making decisions?
# Each feature gets a score from 0.0 to 1.0, and all scores add up to 1.0.
#
# This is critical to verify that temp_change has a meaningful score.
# If temp_change is near 0, the model is ignoring it and the fast-rise
# detection will not work in real life, even if accuracy is 100%.
# We need temp_change to be at least 0.05 (5%) to be effective.

features = ["temperature", "humidity", "temp_change", "avg_temp"]
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

print("\nFeature Importance (higher = model relies on it more):")
for feat, imp in importances.items():
    bar = "█" * int(imp * 40)   # Visual bar proportional to importance score
    print(f"  {feat:<15} {imp:.4f}  {bar}")

# Warn if temp_change importance is too low to be useful
if importances["temp_change"] < 0.05:
    print("\n⚠ WARNING: temp_change importance is very low.")
    print("  The model may not detect fast rises reliably.")
    print("  Consider collecting more fast-rise data at LOW/MEDIUM temperatures.")
else:
    print("\n✅ temp_change has meaningful importance — fast-rise detection should work.")

# ================================================================
# STEP 7 — SANITY CHECK
# ================================================================
# Before saving the model, we test it on 5 known scenarios
# to verify it produces the correct predictions.
# These are not from the training data — they are manually
# created test cases to confirm the model learned correctly.
#
# If any of these fail (❌), the model has not learned properly
# and should not be used on the hardware until fixed.

print("\nSanity check — fast-rise at LOW temperature:")
test_cases = [
    [27.5, 60.0, 1.1, 26.8],   # Fast rise at low temp  → must predict HIGH
    [28.9, 65.0, 1.0, 27.9],   # Fast rise at low temp  → must predict HIGH
    [27.0, 55.0, 0.1, 27.0],   # Stable low temp        → must predict LOW
    [31.0, 70.0, 0.1, 30.5],   # Stable medium temp     → must predict MEDIUM
    [36.0, 75.0, 0.2, 35.5],   # High temp              → must predict HIGH
]

test_df = pd.DataFrame(test_cases, columns=["temperature", "humidity", "temp_change", "avg_temp"])
preds = model.predict(test_df)

labels = ["fast rise LOW  → expect HIGH ",
          "fast rise LOW  → expect HIGH ",
          "stable LOW     → expect LOW  ",
          "stable MEDIUM  → expect MEDIUM",
          "high temp      → expect HIGH "]

for label, pred in zip(labels, preds):
    status = "✅" if (
        ("HIGH"   in label and pred == "HIGH")   or
        ("LOW"    in label and pred == "LOW")    or
        ("MEDIUM" in label and pred == "MEDIUM")
    ) else "❌"
    print(f"  {status}  {label}  got: {pred}")

# ================================================================
# STEP 8 — SAVE THE MODEL
# ================================================================
# joblib.dump() saves the entire trained model to a file.
# This file (model_v2.pkl) contains all 200 decision trees
# and everything the model learned during training.
# ai_control_v2.py loads this file at startup to make
# real-time predictions without needing to retrain.

joblib.dump(model, "model_v2.pkl")
print("\nModel saved as model_v2.pkl")