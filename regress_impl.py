import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess data
ejection_df = pd.read_csv('Fighter_Pilot_Ejection_Success.csv')
ejection_df["Ejection_Type"] = ejection_df["Ejection_Type"].map({"Zero-Zero": 0, "Conventional": 1})
ejection_df["Pilot_Posture"] = ejection_df["Pilot_Posture"].map({"Optimal": 1, "Slouched": 0})

X = ejection_df[["Altitude_ft", "Airspeed_knots", "G_Force", "Ejection_Type", "Pilot_Posture"]].values
Y = ejection_df["Ejection_Success"].values.reshape(-1, 1)

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
np.save("Mean.npy", mean)
np.save("Standard_dev.npy", std)

X = (X - mean) / std
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Logistic regression helpers
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, Y, theta):
    h = sigmoid(X @ theta)
    cost = (-Y * np.log(h) - (1 - Y) * np.log(1 - h)).mean()
    return cost

def gradient_descent(X, Y, theta, alpha, iterations):
    m = len(Y)
    cost_history = []
    
    for _ in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (X.T @ (h - Y)) / m
        theta = theta - alpha * gradient
        cost_history.append(compute_cost(X, Y, theta))
    
    return theta, cost_history

# Train model
theta = np.zeros((X.shape[1], 1))
alpha = 0.1
iterations = 1000
theta, cost_history = gradient_descent(X, Y, theta, alpha, iterations)
np.save("Trained_theta.npy", theta)

# Prediction input
def get_user_inputs():
    print("\nEnter the Ejection Parameters to test \n")
    altitude = float(input("Altitude(ft): "))
    airspeed = float(input("Airspeed(Knots): "))
    G_Force = float(input("G Force (G): "))
    ejection_type = int(input("Ejection Type (0 for Zero-Zero, 1 for Conventional): "))
    posture = int(input("Pilot Posture (1 for Optimal, 0 for Slouched): "))
    return np.array([1, altitude, airspeed, G_Force, ejection_type, posture])

# Load model and stats
theta = np.load("Trained_theta.npy")
mean = np.load("Mean.npy")
std = np.load("Standard_dev.npy")

# Predict loop
while True:
    try:
        user_input = get_user_inputs()
        normalized_input = (user_input[1:] - mean) / std
        normalized_input = np.hstack([1, normalized_input])

        probability = sigmoid(normalized_input @ theta)
        prediction = "SUCCESS" if probability >= 0.5 else "FAILURE"

        print(f"\nPrediction: {prediction} (Probability: {probability.item():.2f})")
        print("--------------------------------------------------------------------")

    except ValueError:
        print("Error: Invalid Input! Please enter proper numerical values.")
        
    except KeyboardInterrupt:
        print("\nExiting...")
        break
      
      
# Plot for How well the model is learning
plt.plot(range(len(cost_history)), cost_history, color='blue')
plt.title("Cost Function V/s. Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost function")
plt.grid(True)
plt.tight_layout()
plt.show()
