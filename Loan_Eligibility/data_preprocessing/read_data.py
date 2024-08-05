import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def split_data(df):
    target_column = 'Loan_Status_Y'
    
    # Print columns of the DataFrame to ensure 'Loan_Status_Y' is present
    print("Columns in DataFrame before splitting:", df.columns.tolist())
    
    if target_column not in df.columns:
        raise KeyError(f"Column '{target_column}' not found in the DataFrame.")
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

