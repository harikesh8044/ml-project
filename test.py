import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess data
ejection_df = pd.read_csv('Fighter_Pilot_Ejection_Success.csv')
ejection_df["Ejection_Type"] = ejection_df["Ejection_Type"].map({"Zero-Zero": 0, "Conventional": 1})
ejection_df["Pilot_Posture"] = ejection_df["Pilot_Posture"].map({"Optimal": 1, "Slouched": 0})
ejection_df["Weather"] = ejection_df["Weather"].map({"Clear": 0, "Rainy": 1, "Windy": 2})
ejection_df["Time_of_Day"] = ejection_df["Time_of_Day"].map({"Day": 0, "Night": 1})

X = ejection_df[["Altitude_ft", "Airspeed_knots", "G_Force", "Ejection_Type", "Pilot_Posture", "Weather", "Time_of_Day"]].values
Y = ejection_df["Ejection_Success"].values.reshape(-1, 1)

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

# Avoid division by zero in normalization
std[std == 0] = 1

np.save("Mean.npy", mean)
np.save("Standard_dev.npy", std)

X = (X - mean) / std
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Logistic regression helpers
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-z))

def compute_cost(X, Y, theta):
    h = sigmoid(X @ theta)
    h = np.clip(h, 1e-10, 1 - 1e-10)  # Avoid log(0)
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

# Model training 
theta = np.zeros((X.shape[1], 1))
alpha = 0.1
iterations = 1000
theta, cost_history = gradient_descent(X, Y, theta, alpha, iterations)
np.save("Trained_theta.npy", theta)

# Plot cost function
plt.figure(figsize=(8, 5))
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Vs. Iterations')
plt.grid(True)
plt.tight_layout()
plt.savefig("cost_plot.png")
plt.show(block=True)

# Enhanced user input
def get_user_inputs():
    print("\n======= EJECTION PREDICTION INPUT =======")
    print("Please enter the following parameters:")
    try:
        altitude = float(input("1. Altitude (ft): "))
        airspeed = float(input("2. Airspeed (knots): "))
        G_Force = float(input("3. G Force (G): "))
        ejection_type = int(input("4. Ejection Type (0 = Zero-Zero, 1 = Conventional): "))
        posture = int(input("5. Pilot Posture (1 = Optimal, 0 = Slouched): "))
        weather = int(input("6. Weather (0 = Clear, 1 = Rainy, 2 = Windy): "))
        time_of_day = int(input("7. Time of Day (0 = Day, 1 = Night): "))
    except ValueError:
        raise ValueError("Invalid input: Please enter numerical values only.")

    return np.array([1, altitude, airspeed, G_Force, ejection_type, posture, weather, time_of_day])

# Load model and stats
theta = np.load("Trained_theta.npy")
mean = np.load("Mean.npy")
std = np.load("Standard_dev.npy")
std[std == 0] = 1  # Again, to ensure division safety

# Predict loop
while True:
    try:
        user_input = get_user_inputs()
        normalized_input = (user_input[1:] - mean) / std
        normalized_input = np.hstack([1, normalized_input])

        z = normalized_input @ theta
        probability = sigmoid(z)
        prediction = "SUCCESS" if probability >= 0.5 else "FAILURE"

        print("\n=======================================")
        print(f"Prediction: {prediction} (Probability: {probability.item():.2f})")
        print("=======================================\n")

    except ValueError as e:
        print(f"\nError: {e}\n")
    except KeyboardInterrupt:
        print("\nExiting Prediction Console. Goodbye!")
        break
