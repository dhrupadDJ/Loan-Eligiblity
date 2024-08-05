import data_preprocessing.read_data as dp
import data_preprocessing.preprocess_data as dc
import models.logistic_regression as lr
import models.random_forest as rf
import models.cross_validation as cv
from sklearn.preprocessing import StandardScaler

# Load the data
file_path = r"C:\Users\jaisw\Desktop\Data Science Final Project\complete project\Loan_Eligibility\Loan_Eligibility\credit.csv"
df = dp.load_data(file_path)

# Preprocess the data
df = dc.handle_missing_values(df)
df = dc.encode_categorical_variables(df)

# Split the data
X_train, X_test, y_train, y_test = dp.split_data(df)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate models
logistic_model = lr.train_logistic_regression(X_train_scaled, y_train)
rf_model = rf.train_random_forest(X_train_scaled, y_train)

logistic_accuracy = lr.evaluate_model(logistic_model, X_test_scaled, y_test)
rf_accuracy = rf.evaluate_model(rf_model, X_test_scaled, y_test)

print(f"Logistic Regression Accuracy: {logistic_accuracy}")
print(f"Random Forest Accuracy: {rf_accuracy}")

# Cross-validation scores
lr_cv_mean, lr_cv_std = cv.cross_validate_model(logistic_model, X_train_scaled, y_train)
rf_cv_mean, rf_cv_std = cv.cross_validate_model(rf_model, X_train_scaled, y_train)

print(f"Logistic Regression CV Mean: {lr_cv_mean}, CV Std: {lr_cv_std}")
print(f"Random Forest CV Mean: {rf_cv_mean}, CV Std: {rf_cv_std}")

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
