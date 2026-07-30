import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    print("Step 1: Loading ML_READY_DATA...")
    data_path = r"C:\Users\Jayanth B\PycharmProjects\PythonProject\data\ML_READY_DATA.csv"
    
    if not os.path.exists(data_path):
        print(f"ERROR: Could not find {data_path}. Please run feature engineering first.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Successfully loaded {len(df)} rows.")

    print("Step 2: Preparing Features and Target...")
    # Drop identifiers and raw time from features
    # 'pms_eta_seconds_remaining' is the target.
    drop_cols = ['icao_address', 'r_callsign', 'abstime_sec', 'pms_eta_seconds_remaining']
    
    X = df.drop(columns=drop_cols)
    y = df['pms_eta_seconds_remaining']
    
    # 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training on {len(X_train)} rows...")
    print(f"Testing on {len(X_test)} rows...")
    print(f"Features being used ({len(X.columns)} total): {list(X.columns)}")

    print("\nStep 3: Initializing XGBoost Regressor AI...")
    # Using 'hist' tree method for extremely fast training on large datasets
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=7,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )

    print("Step 4: Training the AI Model (this may take a minute)...")
    model.fit(X_train, y_train)

    print("\nStep 5: Evaluating AI Accuracy on Unseen Test Data...")
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("==================================================")
    print(f"MAE (Mean Absolute Error) : {mae:.2f} seconds")
    print(f"RMSE (Root Mean Square)   : {rmse:.2f} seconds")
    print(f"R-Squared (Accuracy %)    : {r2*100:.2f} %")
    print("==================================================")

    print("\nStep 6: Saving Production Model...")
    models_dir = r"C:\Users\Jayanth B\PycharmProjects\PythonProject\models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    model_path = os.path.join(models_dir, "xgboost_eta_model.json")
    model.save_model(model_path)
    print(f"SUCCESS! AI Model safely exported to: {model_path}")

if __name__ == "__main__":
    main()
