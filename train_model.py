import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv("Raspi_data - Sheet1.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Fix wrong spelling from sheet
df = df.rename(columns={"humididty": "humidity"})

print("Columns:", df.columns)

# Keep needed columns
df = df[["temperature", "humidity", "action"]].dropna()

# Convert labels
df["action"] = df["action"].map({
    "LED_OFF": 0,
    "LED_ON": 1
})

# Remove rows where mapping failed
df = df.dropna()

# Inputs and outputs
X = df[["temperature", "humidity"]]
y = df["action"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")