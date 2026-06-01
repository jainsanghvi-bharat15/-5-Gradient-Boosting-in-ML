# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load stock market dataset
df = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(5) Gradient Boosting/Stock_Market.csv")

# Convert Date column into datetime format
# This allows extraction of day, month, and year separately
df["Date"] = pd.to_datetime(df["Date"])     # Check missing values in dataset
print("Number of Null Values:\n")
print(df.isnull().sum())
print("\nDataset Information:")
print(df.info())

# Feature Engineering
# Extracting Day, Month, and Year from Date column
# These values help the model identify time-based patterns
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Display updated dataset
print("\nUpdated Dataset:")
print(df.head())

x = df[["Open", "High", "Low", "Volume", "Day", "Month", "Year"]]
y = df["Close"]

# Split dataset into training and testing data
# 80% data → Training    20% data → Testing
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

print("\nLength of Training Data:", len(xtrain))
print("Length of Testing Data:", len(xtest))

# Create Gradient Boosting Regressor model
# n_estimators = Number of decision trees
# learning_rate = Controls contribution of each tree
# max_depth = Maximum depth of each tree
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42)
model.fit(xtrain, ytrain)   # Train the model
print("\nModel Trained Successfully")
print(model)

# Predict stock closing prices using testing data
ypred = model.predict(xtest)
print("\nPredicted Closing Prices:")
print(ypred)


# Model Evaluation
mse = mean_squared_error(ytest, ypred)
rmse = mse ** 0.5
r2 = r2_score(ytest, ypred)
print("\nMean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)


# User Input Section
# Predict stock closing price for today's market data

print("\nEnter Today's Market Data")
open_val = float(input("Enter Open Price: "))
high_val = float(input("Enter High Price: "))
low_val = float(input("Enter Low Price: "))
volume_val = int(input("Enter Volume: "))
day = int(input("Enter Day: "))
month = int(input("Enter Month: "))
year = int(input("Enter Year: "))


# Prepare input data in DataFrame format
# Model expects input in tabular form
input_data = pd.DataFrame([{
    "Open": open_val,
    "High": high_val,
    "Low": low_val,
    "Volume": volume_val,
    "Day": day,
    "Month": month,
    "Year": year}])

# Predict closing price
predicted_close = model.predict(input_data)
print("\nPredicted Closing Price:", predicted_close[0])

# Visual Representation
# Compare Actual vs Predicted Closing Prices
plt.figure(figsize=(8, 5))
plt.scatter(ytest, ypred, color='blue') # Scatter plot
# Reference line
plt.plot(
    [min(ytest), max(ytest)],
    [min(ytest), max(ytest)],
    color='red')

# Labels and title
plt.xlabel("Actual Closing Price")
plt.ylabel("Predicted Closing Price")
plt.title("Actual vs Predicted Stock Closing Price")

# Grid for better readability
plt.grid(True)

# Display graph
plt.show()