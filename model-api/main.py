from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import requests
import os

# Define custom transformer
class ParentOccupationRegexCategorizer(BaseEstimator, TransformerMixin):
    def __init__(self, input_columns=None):
        self.input_columns = input_columns
        self.father_patterns = {
            "Dead": r"\b(died|dead|deth|death|die|dade|date|dyed|late|let|lat|lare|leat|leth|lete|laet|laht|lath|ler|laye|মারা|মৃত্যু)\b",
            "Sick": r"\b(illness|ilnesd|ill|ill[- ]?ness|ill-effected|ilness|illnesses|sick|sickness|sick at home|sike|sickly|sikly|sickli|sik)\b",
            "Unemployed": r"\b(unemploy|unemployed|jobless|un employ|unemployment|bekara|no job|out of a job|kormohin|job lose dew to covid|bekar|beker)\b"
        }
        self.mother_patterns = {
            "Housewife": r"\b(hous[ ]?wife|housewif[e]?|home[ ]?wife|homemaker|household|house work|housework|housing|houdewife|houaswife|housewiffe)\b",
            "Separated": r"\b(alada thaka|separated|separate|sep[ar]e?t(ed)?|অন্য জায়গায়|সেখানে থাকে)\b",
            "Died": r"\b(died|dead|deth|death|die|dade|date|dyed|late|let|lat|lare|leat|leth|lete|laet|laht|lath|ler|laye|মারা|মৃত্যু)\b"
        }

    def fit(self, X, y=None):
      # Store column names if X is a DataFrame
        self.input_columns_ = X.columns if isinstance(X, pd.DataFrame) else \
        self.input_columns if self.input_columns else [f"col_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.input_columns_) if isinstance(X, np.ndarray) else X.copy()
        X['Father Occupation'] = X.apply(
            lambda row: self.regex_categorize(row['Father Occupation'], row.get('Father Occupation NA', None), self.father_patterns, True), axis=1
        )
        X['Father Occupation'] = X.apply(
            lambda row: self.regex_categorize(row['Father Occupation'], row.get('Father Occupation others', None), self.father_patterns, False), axis=1
        )
        X['Mother Occupation'] = X.apply(
            lambda row: self.regex_categorize(row['Mother Occupation'], row.get('Mother Occupation NA', None), self.mother_patterns, True), axis=1
        )
        X['Mother Occupation'] = X.apply(
            lambda row: self.regex_categorize(row['Mother Occupation'], row.get('Mother Occupation others', None), self.mother_patterns, False), axis=1
        )
        columns_to_drop = ['Father Occupation NA', 'Father Occupation others', 'Mother Occupation NA', 'Mother Occupation others']
        X.drop(columns=[col for col in columns_to_drop if col in X.columns], inplace=True)
        self.input_columns_ = [col for col in self.input_columns_ if col not in columns_to_drop]
        return X

    def regex_categorize(self, original_val, column_val, patterns, check_na=True):
        if pd.isnull(column_val):
            return original_val  # skip NaN

        original_val = str(original_val).strip().lower()
        column_val = str(column_val).strip().lower()

        # Only process if original_val matches condition
        if check_na and original_val != "not applicable":
            return original_val
        if not check_na and original_val != "others":
            return original_val

        # Match patterns
        for category, pattern in patterns.items():
            if re.search(pattern, column_val, re.IGNORECASE):
                return category

        return original_val  # fallback if no match

class FirstValueExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column="Learner Disadvantage Group", delimiter=";", input_columns=None):
        self.column = column
        self.delimiter = delimiter
        self.input_columns = input_columns
    def fit(self, X, y=None):
        self.input_columns_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else \
                              self.input_columns if self.input_columns else [f"col_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.input_columns_) if isinstance(X, np.ndarray) else X.copy()

        def extract_first(val):
            if pd.isnull(val):  # handle missing values
                return val
            if self.delimiter in str(val):
                return str(val).split(self.delimiter)[0].strip()
            return str(val).strip()

        X[self.column] = X[self.column].apply(extract_first)
        return X


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None, input_columns=None):
        self.columns_to_drop = columns_to_drop or ['Learner: Created Date', 'Main Source Drinking Water', 'Guardian Occupation', 
                                                   'Learner ID', 'Training Start Date', 'Interest of guardian', 
                                                   'Learner DOB', 'SDP Area', 'Cohort Name']
        self.input_columns = input_columns

    def fit(self, X, y=None):
      # Store column names if X is a DataFrame
        self.input_columns_ = X.columns if isinstance(X, pd.DataFrame) else\
        self.input_columns if self.input_columns else [f"col_{i}" for i in range(X.shape[1])]
        return self  # nothing to fit

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.input_columns_) if isinstance(X, np.ndarray) else X.copy()
        X.drop(columns=[col for col in self.columns_to_drop if col in X.columns], inplace=True)
        self.input_columns_ = [col for col in self.input_columns_ if col not in self.columns_to_drop]
        return X
class FloatToIntTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X, np.ndarray):
            # Floor float values to integers
            return np.floor(X).astype(int)
        else:
            # Apply to DataFrame columns
            for col in X.columns:
                X[col] = np.floor(X[col]).astype(int)
            return X



app = FastAPI()

# Function to download file from Google Drive
def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/drive/folders/1F2bW9g6k7v1EXWbbByaFOPgvbyxEEk8b?usp=drive_link" + file_id
    response = requests.get(URL, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

# Download model files
model_files = {
    'selection_preprocessor.joblib': '1Qhkx-MddHDlJzv3gAC8L5kRCZTDJJmzI',
    'selection_model.joblib': '1DjfNA6xBYFax_q6itTmWrt5hMAn2pkrf',
    'status_pipeline.joblib': '1rQcyZLuNgMm5Tte7ar0dhYfYsgEco1RF',
    'graduation_pipeline.joblib': '1NTSIkqEEk3odRJmWXU7fc3PFyJy9PaP3'
}

for file_name, file_id in model_files.items():
    if not os.path.exists(file_name):
        download_file_from_google_drive(file_id, file_name)

# Load models
try:
    selection_preprocessor = joblib.load('selection_preprocessor.joblib')
    selection_model = joblib.load('selection_model.joblib')
    status_pipeline = joblib.load('status_pipeline.joblib')
    graduation_pipeline = joblib.load('graduation_pipeline.joblib')
except Exception as e:
    raise Exception(f"Error loading models: {e}")

# Required columns
required_columns = [
    'Natural Hazard Frequently', 'Disability', 'First Preferred Trade', 'BRAC Beneficiary Type',
    'Male Earning Member Status', 'Learner Disadvantage Group', 'Learner Religion',
    'Marital Status', 'Mother Occupation', 'Father Occupation', 'Indigenous Group', 'Learner Gender',
    'Age During Training', 'Learner Family Members', 'Earning Family Members', 'Travel Time Health Center',
    'Educational Qualification', 'Learner Dropout Year Passed', 'Learner Current Status',
    'Land Amount', 'Family Monthly Savings', 'Family Average Income', 'Damage by Climate',
    'Cyclone Flood Shelter', 'Intensity of these Hazards', 'City Land ownership status',
    'Previous training experience', 'Natural Hazard 10 Year', 'Migrate for Income'
]

class InputData(BaseModel):
    data: list  # List of values in required_columns order

@app.post("/predict")
async def predict(input: InputData):
    try:
        df = pd.DataFrame([input.data], columns=required_columns)
        
        # Selection prediction
        selection_transformed = selection_preprocessor.transform(df)
        selection_prob = selection_model.predict_proba(selection_transformed)[:, 1]
        selection_pred = "Yes" if selection_prob[0] >= 0.5 else "No"
        
        # Status and Graduation Status predictions
        status_pred = "N/A"
        graduation_pred = "N/A"
        if selection_pred == "Yes":
            status_pred = status_pipeline.predict(df)[0]
            graduation_pred = graduation_pipeline.predict(df)[0]
        
        return {
            'Selection_Pred': selection_pred,
            'Status_Pred': status_pred,
            'Graduation_Pred': graduation_pred
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)