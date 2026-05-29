import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ERROR CORRECTION:
# Original code used GradientBoostingClassifier()
# That is used for classification problems.
# California Housing predicts house price/value, which is a continuous numeric output.
# Therefore we must use: GradientBoostingRegressor()
# 1. Load dataset
# ---------------------------------------------------------
df = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(5) Gradient Boosting/California_Housing.csv")
print(df.head())

# 2. Define input and output data
df_input = df.drop('MedHouseValue', axis='columns')
df_output = df['MedHouseValue']
# 3. Split dataset into training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(
    df_input,
    df_output,
    test_size=0.2,
    random_state=42)
print("Training Records:", len(xtrain))
print("Testing Records:", len(xtest))

# 4. Create Gradient Boosting Regressor model
# n_estimators = total number of decision trees
# learning_rate = contribution of each tree
# max_depth = depth of each weak decision tree
# random_state = same output every run

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42)

# 5. Train model
model.fit(xtrain, ytrain)

# 6. Training score
print("Training Score:", model.score(xtrain, ytrain))

# 7. Predict testing data
y_pred = model.predict(xtest)
print("Predicted Values:\n", y_pred)

# 8. Evaluate model
    # MSE = Mean Squared Error
    # Lower value = better model
mse = mean_squared_error(ytest, y_pred)
rmse = mse ** 0.5   # RMSE = Square root of MSE

# R2 Score = closer to 1 means better prediction
r2 = r2_score(ytest, y_pred)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)


