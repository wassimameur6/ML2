"""
Train churn prediction model and save artifacts.
"""
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

# Paths
DATA_PATH = "../data/churn2.csv"
ARTIFACTS_PATH = "../artifacts"

def load_and_preprocess_data():
    """Load and preprocess the churn dataset."""
    df = pd.read_csv(DATA_PATH)

    # Drop empty column if exists (trailing comma in CSV)
    df = df.dropna(axis=1, how='all')

    # Target variable
    df['Churn'] = (df['Attrition_Flag'] == 'Attrited Customer').astype(int)

    # Generate synthetic email for demo
    df['Email'] = df['CLIENTNUM'].apply(lambda x: f"customer_{x}@example.com")

    # Features to use
    categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    numerical_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
                      'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit',
                      'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
                      'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Prepare features
    feature_cols = categorical_cols + numerical_cols
    X = df[feature_cols]
    y = df['Churn']

    # Store customer info for agent use
    customer_info = df[['CLIENTNUM', 'Email', 'Customer_Age', 'Gender', 'Income_Category',
                        'Card_Category', 'Months_on_book', 'Total_Trans_Amt']].copy()

    return X, y, label_encoders, feature_cols, customer_info, df

def train_model(X, y):
    """Train the churn prediction model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print("Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

    return model, scaler

def save_artifacts(model, scaler, label_encoders, feature_cols):
    """Save trained model and preprocessing artifacts."""
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)

    with open(f"{ARTIFACTS_PATH}/model.pickle", 'wb') as f:
        pickle.dump(model, f)

    with open(f"{ARTIFACTS_PATH}/scaler.pickle", 'wb') as f:
        pickle.dump(scaler, f)

    with open(f"{ARTIFACTS_PATH}/label_encoders.pickle", 'wb') as f:
        pickle.dump(label_encoders, f)

    with open(f"{ARTIFACTS_PATH}/feature_cols.pickle", 'wb') as f:
        pickle.dump(feature_cols, f)

    print(f"\nArtifacts saved to {ARTIFACTS_PATH}/")

def main():
    print("Loading and preprocessing data...")
    X, y, label_encoders, feature_cols, customer_info, df = load_and_preprocess_data()

    print(f"Dataset: {len(df)} customers")
    print(f"Churn rate: {y.mean()*100:.2f}%")
    print(f"Features: {len(feature_cols)}")

    print("\nTraining model...")
    model, scaler = train_model(X, y)

    print("\nSaving artifacts...")
    save_artifacts(model, scaler, label_encoders, feature_cols)

    # Save reference data for monitoring
    df.to_csv(f"../data/ref_data.csv", index=False)
    print("Reference data saved to ../data/ref_data.csv")

if __name__ == "__main__":
    main()
