import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train_and_save_model():
    print("--- Starting Model Training ---")

    # 1. Load the data we just created
    try:
        df = pd.read_csv('hackathon_data.csv')
    except FileNotFoundError:
        print("Error: csv file not found. Run generate_data.py first!")
        return

    # 2. Preprocessing
    # We convert 'Tech_Stack_Diversity' (Words) into Numbers
    le = LabelEncoder()
    df['Tech_Stack_Diversity'] = le.fit_transform(df['Tech_Stack_Diversity'])
    
    # 3. Separate Features (X) and Target (y)
    X = df.drop(columns=['Target'])
    y = df['Target']

    # 4. Split data (80% for training, 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Initialize and Train the XGBoost Model
    # This is the "Learning" phase
    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        use_label_encoder=False, 
        eval_metric='logloss'
    )
    
    print("Training model...")
    model.fit(X_train, y_train)

    # 6. Check Accuracy
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model Accuracy: {acc * 100:.2f}%")

    # 7. Save the Model and Encoder for the App
    joblib.dump(model, 'xgb_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("SUCCESS: Model files saved (.pkl).")

if __name__ == "__main__":
    train_and_save_model()