import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import joblib
from sklearn.metrics import confusion_matrix


# Function to preprocess data, including label encoding and handling unseen labels
def preprocess_data(df, label_encoders=None):
    # Drop unnecessary columns
    df = df.drop(columns=['trans_num', 'merchant','is_fraud'], errors='ignore')  # Avoid errors if columns are not present

    # Convert 'dob' column (if present) to datetime, and extract relevant features
    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')  # Convert to datetime, errors='coerce' will turn invalid dates into NaT
        df['dob_year'] = df['dob'].dt.year
        df['dob_month'] = df['dob'].dt.month
        df['dob_day'] = df['dob'].dt.day
        df = df.drop(columns=['dob'])  # Drop original 'dob' column

    # Convert 'trans_date_trans_time' to datetime and extract features
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')  # Handling invalid dates
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.day
    df['month'] = df['trans_date_trans_time'].dt.month
    df = df.drop(columns=['trans_date_trans_time'])  # Drop original date column

    # Encode categorical variables
    label_encoders = label_encoders or {}  # If no label encoders provided, initialize an empty dictionary
    categorical_cols = ['category', 'state', 'job', 'city']

    for col in categorical_cols:
        le = label_encoders.get(col, LabelEncoder())  # Get existing encoder or create a new one
        if col not in label_encoders:  # Fit encoder only if not done before (for training data)
            le.fit(df[col].astype(str))  # Fit on training data
            label_encoders[col] = le  # Save encoder for future use

        # Transform with handling for unseen labels
        df[col] = df[col].astype(str)  # Ensure all values are strings
        # Handle unseen labels by replacing with the most frequent value
        df[col] = le.transform(df[col].where(df[col].isin(le.classes_), le.classes_[0]))

    # Check for missing values after conversion and handle them if any
    df = df.fillna(0)  # Fill missing values with 0 or other appropriate values

    return df, label_encoders

def train_model():
    # Load real transactions
    df_real = pd.read_csv("credit_card_fraud.csv")

    # Preprocess data
    df_train, label_encoders = preprocess_data(df_real)

    # Train Isolation Forest
    model = IsolationForest(n_estimators=500, contamination=0.005, random_state=42)
    model.fit(df_train)

    # Save trained model and label encoders
    joblib.dump(model, "isolation_forest.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")

    print("✅ Model trained and saved!")
#
# Run the training function
train_model()

# Train Isolation Forest
# def train_isolation_forest():
#     df = pd.read_csv("credit_card_fraud.csv")
#     df, label_encoders = preprocess_data(df)
#
#     model = IsolationForest(n_estimators=500, contamination=0.008, max_samples=0.8, random_state=42)
#     model.fit(df)
#
#     joblib.dump(model, "isolation_forest.pkl")
#     joblib.dump(label_encoders, "label_encoders.pkl")
#     print("✅ Model trained and saved!")
#
#     # y_pred = model.predict(df)
#     # y_pred = [1 if x == -1 else 0 for x in y_pred]
#     # tn, fp, fn, tp = confusion_matrix(df['is_fraud'], y_pred).ravel()
#     # print(f"Isolation Forest Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
#
#
# train_isolation_forest()
#
#
