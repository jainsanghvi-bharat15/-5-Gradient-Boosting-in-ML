import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(5) Gradient Boosting/Salary_Prediction.csv")
print("Original Dataset:\n",df.head())

# Convert categorical columns into numeric values
# EducationLevel and JobRole contain text values, so encoding is required for machine learning models.

Edu_Lev = LabelEncoder()
JR = LabelEncoder()
df['EduLev_New'] = Edu_Lev.fit_transform(df['EducationLevel'])
df['JR_New'] = JR.fit_transform(df['JobRole'])

# Remove original text columns
df.drop('EducationLevel', axis='columns', inplace=True)
df.drop('JobRole', axis='columns', inplace=True)

print("\nCleaned Dataset:\n",df.head())

# Define independent and dependent variables
x = df.drop('Salary(LPA)', axis='columns')
y = df['Salary(LPA)']

# Split data into training and testing sets, 80% training and 20% testing
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
print("\nTraining Records:", len(xtrain))
print("Testing Records :", len(xtest))

# Create Gradient Boosting Regressor model
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.5,
    max_depth=2,
    random_state=42)
model.fit(xtrain, ytrain)
print("\nTraining Score:", model.score(xtrain, ytrain))
y_pred = model.predict(xtest)
print("\nPredicted Salaries:\n", y_pred)

# Model evaluation
mse = mean_squared_error(ytest, y_pred)
rmse = mse ** 0.5
r2 = r2_score(ytest, y_pred)
print("\nMean Squared Error :", mse)
print("Root Mean Squared Error :", rmse)
print("R2 Score :", r2)

# User Input for Salary Prediction
print("\nEnter Employee Details for Salary Prediction")

experience = float(input("Enter Years of Experience: "))
age = int(input("Enter Age: "))

print("\nEducation Level Options:")
print("0 = Bachelor\t1 = Master\t2 = PhD")
edu = int(input("Enter Education Level Code: "))
print("\nJob Role Options:")
for i, role in enumerate(JR.classes_):
    print(i, "=", role)
job = int(input("Enter Job Role Code: "))
predicted_salary = model.predict([[experience, age, edu, job]])
print("\nPredicted Salary:", predicted_salary, "LPA")