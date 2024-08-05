# Loan Eligibility Prediction

## Project Overview
This project aims to predict loan eligibility using machine learning algorithms. We utilize logistic regression and random forest models to predict whether individuals qualify for a loan based on various financial attributes. The dataset used is `credit.csv`, which contains pertinent financial details of the applicants.

## Getting Started

### Prerequisites
Ensure you have Python installed on your machine. You can download Python [here](https://www.python.org/downloads/). Additionally, you will need the following libraries:
- pandas
- numpy
- scikit-learn

You can install these packages using pip:
```bash
pip install pandas numpy scikit-learn
```

### Installation
Clone the repository to your local machine:
```bash
git clone https://github.com/dhrupadDJ/DatsScience_Final_Project
cd loan-eligibility
```

### File Structure
- `data_preprocessing/`
  - `read_data.py` - Module to load data.
  - `preprocess_data.py` - Module to preprocess data.
- `models/`
  - `logistic_regression.py` - Logistic regression model.
  - `random_forest.py` - Random forest model.
  - `cross_validation.py` - Cross-validation utilities.
- `Loan_Eligibility/`
  - `credit.csv` - Dataset file.

### Running the Code
To run the prediction script:
```bash
python main.py
```

Replace `main.py` with the actual name of your script file if different.

## Usage
This project can be used to evaluate the eligibility of loan applications using historical data. The models provide a binary output determining the eligibility of each applicant.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

