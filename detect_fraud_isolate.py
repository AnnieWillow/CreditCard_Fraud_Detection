# import pandas as pd
# import joblib
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import IsolationForest
#
# # Function to preprocess the new data, including label encoding and handling unseen labels
# def preprocess_data(df, label_encoders=None):
#     # Drop unnecessary columns
#     df = df.drop(columns=['trans_num', 'merchant'], errors='ignore')  # Avoid errors if columns are not present
#
#     # Convert 'dob' column (if present) to datetime, and extract relevant features
#     if 'dob' in df.columns:
#         df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
#         df['dob_year'] = df['dob'].dt.year
#         df['dob_month'] = df['dob'].dt.month
#         df['dob_day'] = df['dob'].dt.day
#         df = df.drop(columns=['dob'])
#
#     # Convert 'trans_date_trans_time' to datetime and extract features
#     df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
#     df['hour'] = df['trans_date_trans_time'].dt.hour
#     df['day'] = df['trans_date_trans_time'].dt.day
#     df['month'] = df['trans_date_trans_time'].dt.month
#     df = df.drop(columns=['trans_date_trans_time'])
#
#     # Encode categorical variables
#     label_encoders = label_encoders or {}
#     categorical_cols = ['category', 'state', 'job', 'city']
#
#     for col in categorical_cols:
#         le = label_encoders.get(col, LabelEncoder())
#         if col not in label_encoders:
#             le.fit(df[col].astype(str))  # Fit encoder only if not done before (for new data)
#             label_encoders[col] = le
#
#         df[col] = df[col].astype(str)
#         df[col] = le.transform(df[col].where(df[col].isin(le.classes_), le.classes_[0]))
#
#     # Fill missing values
#     df = df.fillna(0)
#
#     return df
#
# def detect_fraud():
#     # Load the trained model and label encoders
#     model = joblib.load("isolation_forest.pkl")
#     label_encoders = joblib.load("label_encoders.pkl")
#
#     # Load new/fake transaction data
#     df_fake = pd.read_csv("fake_transactions.csv")
#
#     # Preprocess the data the same way as during training
#     df_fake = preprocess_data(df_fake, label_encoders)
#
#     # Make predictions using the trained model
#     predictions = model.predict(df_fake)
#
#     # Add the predictions to the dataframe (1 = normal, -1 = fraud)
#     df_fake['prediction'] = predictions
#
#     # Show or save the results
#     print(df_fake[['prediction']].head())  # Print predictions
#     df_fake.to_csv("predictions.csv", index=False)  # Save predictions to CSV file
#
# # Run the detection function
# detect_fraud()

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

# Function to preprocess the new data, including label encoding and handling unseen labels
def preprocess_data(df, label_encoders=None):
    # Drop unnecessary columns
    df = df.drop(columns=['trans_num', 'merchant','is_fraud'], errors='ignore')  # Avoid errors if columns are not present

    # Convert 'dob' column (if present) to datetime, and extract relevant features
    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['dob_year'] = df['dob'].dt.year
        df['dob_month'] = df['dob'].dt.month
        df['dob_day'] = df['dob'].dt.day
        df = df.drop(columns=['dob'])

    # Convert 'trans_date_trans_time' to datetime and extract features
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.day
    df['month'] = df['trans_date_trans_time'].dt.month
    df = df.drop(columns=['trans_date_trans_time'])

    # Encode categorical variables
    label_encoders = label_encoders or {}
    categorical_cols = ['category', 'state', 'job', 'city']

    for col in categorical_cols:
        le = label_encoders.get(col, LabelEncoder())
        if col not in label_encoders:
            le.fit(df[col].astype(str))  # Fit encoder only if not done before (for new data)
            label_encoders[col] = le

        df[col] = df[col].astype(str)
        df[col] = le.transform(df[col].where(df[col].isin(le.classes_), le.classes_[0]))

    # Fill missing values
    df = df.fillna(0)

    return df

def detect_fraud():
    # Load the trained model and label encoders
    model = joblib.load("isolation_forest.pkl")
    label_encoders = joblib.load("label_encoders.pkl")

    # Load new/fake transaction data
    df_fake = pd.read_csv("fake_transactions.csv")

    # Preprocess the data the same way as during training
    df_fake = preprocess_data(df_fake, label_encoders)

    # Make predictions using the trained model
    predictions = model.predict(df_fake)

    # Map predictions (-1 = Fraud, 1 = Normal)
    prediction_labels = ["Fraud Transaction" if pred == -1 else "Normal Transaction" for pred in predictions]

    # Add the predictions to the dataframe
    df_fake['prediction'] = prediction_labels

    # Print results in the desired format
    print(df_fake[['prediction']].head())  # Display prediction column

    # Optionally, save predictions to CSV file
    df_fake.to_csv("predictions.csv", index=False)  # Save predictions to CSV file

# Run the detection function
detect_fraud()
