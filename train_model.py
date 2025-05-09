import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load your data
np.random.seed(42)  # For reproducibility

n_samples = 500  # Adjust this value to generate more samples

# Generating data
age = np.random.randint(30, 80, size=n_samples)
height = np.random.randint(150, 190, size=n_samples)
weight = np.random.randint(50, 100, size=n_samples)
systolic_bp = np.random.randint(110, 180, size=n_samples)
diastolic_bp = np.random.randint(60, 120, size=n_samples)
cholesterol = np.random.randint(1, 4, size=n_samples)
glucose = np.random.randint(1, 3, size=n_samples)
smoking = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # 30% smokers
alcohol = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # 30% alcohol consumers
active = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])  # 60% active
target = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])  # Randomly assigned target (binary classification)

# Creating the DataFrame with the generated data
data = {
    "age": age,
    "height": height,
    "weight": weight,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "cholesterol": cholesterol,
    "glucose": glucose,
    "smoking": smoking,
    "alcohol": alcohol,
    "active": active,
    "target": target
}

df = pd.DataFrame(data)
print(df.head())

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
