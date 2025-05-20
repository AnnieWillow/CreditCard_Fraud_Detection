import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Function to preprocess new transaction data
def preprocess_data(df, label_encoders=None):
    # Drop unnecessary columns, including target variable 'is_fraud'
    df = df.drop(columns=['trans_num', 'merchant', 'is_fraud'], errors='ignore')  # Ignore 'is_fraud' if present

    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['dob_year'] = df['dob'].dt.year
        df['dob_month'] = df['dob'].dt.month
        df['dob_day'] = df['dob'].dt.day
        df = df.drop(columns=['dob'])

    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.day
    df['month'] = df['trans_date_trans_time'].dt.month
    df = df.drop(columns=['trans_date_trans_time'])

    label_encoders = label_encoders or {}
    categorical_cols = ['category', 'state', 'job', 'city']

    for col in categorical_cols:
        le = label_encoders.get(col, LabelEncoder())
        if col not in label_encoders:
            le.fit(df[col].astype(str))  # Fit only if not preloaded
            label_encoders[col] = le
        df[col] = le.transform(df[col].astype(str).where(df[col].isin(le.classes_), le.classes_[0]))

    df = df.fillna(0)  # Handle missing values
    return df

def detect_fraud_xgboost():
    # Load trained model and encoders
    model = joblib.load("xgboost_model.pkl")  # Ensure this is an XGBClassifier
    label_encoders = joblib.load("label_encoders_xgboost.pkl")

    # Load new transaction data
    df_fake = pd.read_csv("fake_transactions.csv")

    # Preprocess the data
    df_fake = preprocess_data(df_fake, label_encoders)

    # 🔥 **FIXED:** Pass DataFrame directly instead of converting to DMatrix
    predictions = model.predict(df_fake)

    # Convert to labels
    prediction_labels = ["Fraud Transaction" if pred > 0.5 else "Normal Transaction" for pred in predictions]

    df_fake['prediction'] = prediction_labels

    # Show results
    print(df_fake[['prediction']].head())  # Display the predictions

    # Save predictions to CSV
    df_fake.to_csv("predictions_XGBoost.csv", index=False)  # Save predictions to CSV file


detect_fraud_xgboost()