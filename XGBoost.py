# import pandas as pd
# import joblib
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#
# # Function to preprocess data
# def preprocess_data(df, label_encoders=None):
#     df = df.drop(columns=['trans_num', 'merchant'], errors='ignore')
#
#     if 'dob' in df.columns:
#         df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
#         df['dob_year'] = df['dob'].dt.year
#         df['dob_month'] = df['dob'].dt.month
#         df['dob_day'] = df['dob'].dt.day
#         df = df.drop(columns=['dob'])
#
#     df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
#     df['hour'] = df['trans_date_trans_time'].dt.hour
#     df['day'] = df['trans_date_trans_time'].dt.day
#     df['month'] = df['trans_date_trans_time'].dt.month
#     df = df.drop(columns=['trans_date_trans_time'])
#
#     label_encoders = label_encoders or {}
#     categorical_cols = ['category', 'state', 'job', 'city']
#
#     for col in categorical_cols:
#         le = label_encoders.get(col, LabelEncoder())
#         if col not in label_encoders:
#             df[col] = df[col].astype(str)
#             le.fit(df[col])
#             label_encoders[col] = le
#
#         df[col] = le.transform(df[col].astype(str))
#
#     df = df.fillna(0)
#     return df, label_encoders
#
# # Train XGBoost model
# def train_xgboost():
#     df = pd.read_csv("balanced_credit_card_fraud.csv")
#     df, label_encoders = preprocess_data(df)
#
#     X = df.drop(columns=['is_fraud'])
#     y = df['is_fraud']
#
#     # Train-Test Split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
#     model = xgb.XGBClassifier(
#         n_estimators=100,
#         learning_rate=0.1,
#         max_depth=6,
#         random_state=42,
#         use_label_encoder=False,
#         eval_metric='logloss'
#     )
#     model.fit(X_train, y_train)
#
#     # Save model and encoders
#     joblib.dump(model, "xgboost_model.pkl")
#     joblib.dump(label_encoders, "label_encoders_xgboost.pkl")
#
#     # Evaluate model
#     y_pred = model.predict(X_test)
#     print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
#     print(f"Precision: {precision_score(y_test, y_pred):.4f}")
#     print(f"Recall: {recall_score(y_test, y_pred):.4f}")
#     print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
#     print("✅ XGBoost model training complete!")
#
# # Run training
# if __name__ == "__main__":
#     train_xgboost()

import pandas as pd
import joblib
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Function to preprocess data
def preprocess_data(df, label_encoders=None):
    df = df.drop(columns=['trans_num', 'merchant'], errors='ignore')

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
            df[col] = df[col].astype(str)
            le.fit(df[col])
            label_encoders[col] = le

        df[col] = le.transform(df[col].astype(str))

    df = df.fillna(0)
    return df, label_encoders

# Train XGBoost model with SMOTE
def train_xgboost_with_smote():
    df = pd.read_csv("credit_card_fraud.csv")  # Original imbalanced dataset
    df, label_encoders = preprocess_data(df)

    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']

    # Apply SMOTE to balance the dataset
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        scale_pos_weight=y_train.value_counts()[0] / y_train.value_counts()[1],
        random_state=42,
        eval_metric = 'logloss'
    )
    model.fit(X_train, y_train)

#     # Save model and encoders
#     joblib.dump(model, "xgboost_model.pkl")
#     joblib.dump(label_encoders, "label_encoders_xgboost.pkl")
#
#
#     print("✅ XGBoost model training with SMOTE complete!")
#
# # Run training
# if __name__ == "__main__":
#     train_xgboost_with_smote()

    model.fit(X_train, y_train)
    joblib.dump(model, "xgboost_model.pkl")
    joblib.dump(label_encoders, "label_encoders_xgboost.pkl")
    print("✅ XGBoost model training with SMOTE complete!")

    # Evaluate the model
    y_pred = model.predict(X_test)



# Run training
if __name__ == "__main__":
    train_xgboost_with_smote()
