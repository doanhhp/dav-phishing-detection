import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from src.features.structural import StructuralProcessor
from src.models.structural_rf import Structural_RF
from src.models.structural_xgb import Structural_XGB

def main():
    print("--- 1. Loading ONLY the New Zero-Day Dataset ---")
    df_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_html = pd.read_excel('data/raw/OOD_html.xlsx')
    
    y = df_url['Category'].map({'ham': 0, 'spam': 1}).values

    df_raw = df_url[['Data']].copy()
    df_raw['html'] = df_html['Data']

    print("--- 2. Performing 80/20 Train/Test Split ---")
    df_train, df_test, y_train, y_test = train_test_split(
        df_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    print("--- 3. Extracting Structural Features (No Leakage) ---")
    processor = StructuralProcessor({})
    # Fit on train, transform on test
    X_train = processor.fit_transform(df_train)
    X_test = processor.transform(df_test)

    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")

    print("\n--- 4. Training Models In-Domain ---")
    rf = Structural_RF(config={'n_estimators': 100})
    rf.fit(X_train, y_train)

    xgb = Structural_XGB(config={})
    xgb.fit(X_train, y_train)

    print("\n--- 5. Evaluating Models ---")
    
    # RF
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    print(f"Random Forest (In-Domain) -> Acc: {accuracy_score(y_test, rf_preds):.4f} | AUC: {roc_auc_score(y_test, rf_probs):.4f}")
    print(classification_report(y_test, rf_preds))

    # XGBoost
    xgb_preds = xgb.predict(X_test)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    print(f"XGBoost (In-Domain)       -> Acc: {accuracy_score(y_test, xgb_preds):.4f} | AUC: {roc_auc_score(y_test, xgb_probs):.4f}")
    print(classification_report(y_test, xgb_preds))

if __name__ == "__main__":
    main()
