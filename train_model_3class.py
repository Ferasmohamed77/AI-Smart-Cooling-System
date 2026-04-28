import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
df = pd.read_csv("sensor_3class.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

print("Columns:", df.columns)

# Fix possible typo
df = df.rename(columns={"humididty": "humidity"})

# Keep needed columns
df = df[["temperature", "humidity", "action"]].dropna()

# Clean action labels
df["action"] = df["action"].astype(str).str.strip().str.upper()

# Convert labels to numbers
df["action"] = df["action"].map({
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2
})

# Drop invalid rows
df = df.dropna()

print("Total rows:", len(df))
print(df.head())

# Split
X = df[["temperature", "humidity"]]
y = df["action"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model_3class.pkl")
print("\nModel saved as model_3class.pkl")