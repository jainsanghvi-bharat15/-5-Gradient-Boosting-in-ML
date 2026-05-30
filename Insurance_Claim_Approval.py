# # Insurance Claim Approval Prediction using XGBoost
# # Import required libraries
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt

# # Load dataset
# df = pd.read_csv("D://scikit_data/insurancedata/insurance_claim_approval.csv")
# # Basic dataset information
# print(df.head())
# print("\nMissing Values:\n", df.isnull().sum())
# print("\nDataset Shape:", df.shape)

# # Encode categorical columns into numeric values
# label_encoders = {}
# for col in ["Gender", "Smoking", "PolicyType", "PreExistingCondition"]:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     label_encoders[col] = le
# print("\nEncoded Dataset:\n", df.head())

# # Define independent and dependent variables
# X = df.drop("ClaimApproved", axis=1)   # Input
# y = df["ClaimApproved"]                # Output

# # Split dataset into training and testing data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42)
# print("\nTraining Records:", len(X_train))
# print("Testing Records:", len(X_test))

# # Create and train XGBoost model
# model = XGBClassifier(
#     use_label_encoder=False,
#     eval_metric='logloss')
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# # Model evaluation
# acc = accuracy_score(y_test, y_pred)
# cm = confusion_matrix(y_test, y_pred)
# print("\nModel Accuracy:", round(acc * 100, 2), "%")
# print("\nConfusion Matrix:\n", cm)

# # User input prediction
# print("\nEnter Insurance Details")
# age = int(input("Age: "))
# gender = input("Gender (Male/Female): ")
# bmi = float(input("BMI: "))
# smoking = input("Smoking (Yes/No): ")
# policy = input("Policy Type (Basic/Premium/Gold): ")
# amount = float(input("Claim Amount: "))
# condition = input("Pre-existing Condition (None/Diabetes/Heart Disease/Asthma): ")
# stay = int(input("Hospital Stay Days: "))

# # Encode user inputs
# gender = label_encoders["Gender"].transform([gender])[0]
# smoking = label_encoders["Smoking"].transform([smoking])[0]
# policy = label_encoders["PolicyType"].transform([policy])[0]
# condition = label_encoders["PreExistingCondition"].transform([condition])[0]

# # Create input dataframe
# input_data = pd.DataFrame([{
#     "Age": age,
#     "Gender": gender,
#     "BMI": bmi,
#     "Smoking": smoking,
#     "PolicyType": policy,
#     "ClaimAmount": amount,
#     "PreExistingCondition": condition,
#     "HospitalStayDays": stay}])

# # Predict claim approval
# prediction = model.predict(input_data)[0]

# if prediction == 1:
#     print("\nClaim Status: Approved")
# else:
#     print("\nClaim Status: Not Approved")

# # Feature Importance Graph
# plt.figure(figsize=(8,5))
# plt.barh(
#     X.columns,
#     model.feature_importances_,
#     color='skyblue')

# plt.xlabel("Importance Score")
# plt.ylabel("Features")
# plt.title("Feature Importance - Insurance Claim Approval")
# plt.tight_layout()
# plt.show()