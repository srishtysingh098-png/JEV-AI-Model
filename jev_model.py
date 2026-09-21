import numpy as np
from sklearn.linear_model import LogisticRegression

# Training data
# [Python, SQL, Excel, Power BI, Experience]

X = np.array([
    [1, 1, 1, 1, 12],
    [1, 1, 1, 0, 6],
    [0, 1, 1, 1, 8],
    [1, 0, 1, 0, 3],
    [0, 0, 1, 0, 2],
    [1, 1, 1, 1, 18],
    [0, 1, 0, 1, 4],
    [1, 1, 0, 1, 10],
    [0, 0, 0, 0, 1],
    [1, 1, 1, 0, 15]
])

# 1 = Eligible, 0 = Not Eligible
y = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1])

# Create AI model
model = LogisticRegression()

# Train model
model.fit(X, y)

print("==============================")
print("       JEV AI MODEL")
print("==============================")

# Get candidate information
python = int(input("Python knowledge (1/0): "))
sql = int(input("SQL knowledge (1/0): "))
excel = int(input("Excel knowledge (1/0): "))
powerbi = int(input("Power BI knowledge (1/0): "))
experience = int(input("Experience in months: "))

candidate = np.array([[
    python,
    sql,
    excel,
    powerbi,
    experience
]])

# Make prediction
prediction = model.predict(candidate)[0]
probability = model.predict_proba(candidate)[0][1]

print("\n========== RESULT ==========")

if prediction == 1:
    print("Status: Eligible")
else:
    print("Status: Not Eligible")

print(f"Probability: {probability * 100:.2f}%")
