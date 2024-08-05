import pandas as pd

def handle_missing_values(df):
    df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    df = df.dropna(subset=['Gender', 'Married', 'Dependents', 'Self_Employed'])
    return df

def encode_categorical_variables(df):
    df = pd.get_dummies(df, drop_first=True)
    return df
