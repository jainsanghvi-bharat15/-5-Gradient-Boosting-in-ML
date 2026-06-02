# Project: Student Dropout Risk Prediction
# Objective: Predict whether a student is at risk of dropping out using Gradient Boosting Classification Algorithm.


# Step 1: Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Step 2: Load Dataset
df = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(5) Gradient Boosting/Student_Dropout.csv")
print("First 5 Records:\n",df.head())
print("\nMissing Values:\n",df.isnull().sum())
print("\nDataset Shape:", df.shape)
print("\nDataset Information:\n",df.info())

# Step 3: Encode Categorical Variables
# Machine Learning models work only with numeric values
le_gender = LabelEncoder()
le_support = LabelEncoder()
le_job = LabelEncoder()

# Convert text data into numeric format
df["Gender"] = le_gender.fit_transform(df["Gender"])
df["ParentalSupport"] = le_support.fit_transform(df["ParentalSupport"])
df["PartTimeJob"] = le_job.fit_transform(df["PartTimeJob"])
print("\nDataset After Encoding:\n",df.head())

# Step 4: Define Features and Target Variable
X = df.drop("DropoutRisk", axis=1)
y = df["DropoutRisk"]
print("\nInput Features:\n",X.head())
print("\nTarget Variable:\n",y.head())

# Step 5: Split Dataset into Training and Testing Data
# 80% → Training Data    20% → Testing Data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("\nLength of Training Data:", len(X_train))
print("Length of Testing Data:", len(X_test))

# Step 6: Create and Train Gradient Boosting Model
# n_estimators = Number of decision trees
# learning_rate = Contribution of each tree
# max_depth = Maximum depth of each tree
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42)
model.fit(X_train, y_train)
print("\nModel Trained Successfully")
print(model)

# Step 7: Model Evaluation
y_pred = model.predict(X_test)  # Predict output using testing data
acc = accuracy_score(y_test, y_pred)    # Calculate accuracy
cm = confusion_matrix(y_test, y_pred)   # Create confusion matrix
print("\nModel Accuracy:", round(acc * 100, 2), "%")
print("\nConfusion Matrix:\n",cm)

# Step 8: User Input for Prediction
print("\nEnter Student Details to Predict Dropout Risk")
gender = input("Gender (Male/Female): ")
support = input("Parental Support (Low/Medium/High): ")
job = input("Part-Time Job? (Yes/No): ")
age = int(input("Age: "))
gpa = float(input("Current GPA (0.0–4.0): "))
attendance = float(input("Attendance Rate (0–100): "))
study_hours = int(input("Study Hours Per Week: "))

# Gender Encoding
if gender == "Male":
    gender_encoded = 1
else:
    gender_encoded = 0

# Parental Support Encoding
if support == "Low":
    support_encoded = 0
elif support == "Medium":
    support_encoded = 1
else:
    support_encoded = 2

# Part-Time Job Encoding
if job == "Yes":
    job_encoded = 1
else:
    job_encoded = 0

# Create input dataframe for prediction
input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender_encoded,
    "StudyHoursPerWeek": study_hours,
    "AttendanceRate": attendance,
    "ParentalSupport": support_encoded,
    "PartTimeJob": job_encoded,
    "CurrentGPA": gpa}])
prediction = model.predict(input_data)
print("\nPrediction Result:")

if prediction[0] == 1:
    print("Student is At Risk of Dropping Out")
else:
    print("Student is Not at Risk")

# Step 9: Feature Importance Visualization
# Shows which features affect prediction the most
# Get importance scores
importances = model.feature_importances_
print("\nFeature Importance Scores:\n")
print(importances)

feature_names = X.columns
print("\nFeature Names:\n")
print(feature_names)

# Create horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(
    feature_names,
    importances,
    color='skyblue')

# Labels and title
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance - Student Dropout Risk Prediction")

plt.grid(axis='x')
plt.tight_layout()
plt.show()

# Step 10: Actual vs Predicted Visualization
plt.figure(figsize=(8, 5))

# Scatter plot
plt.scatter(
    y_test,
    y_pred,
    color='green')

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Dropout Risk")
plt.grid(True)
plt.show()