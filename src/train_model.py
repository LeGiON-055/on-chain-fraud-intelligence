"""
train_model.py
--------------
Trains an XGBoost fraud detection model with GridSearchCV hyperparameter
tuning, 5-fold Stratified Cross-Validation, and SHAP explainability.

"""

import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from pathlib import Path
from typing import Tuple, Dict, Any

from xgboost import XGBClassifier
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score
)
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURES_PATH = Path("data/processed/features.csv")
MODEL_SAVE_PATH = Path("models/fraud_detector.joblib")
SCALER_SAVE_PATH = Path("models/scaler.joblib")
FEATURE_NAMES_PATH = Path("models/feature_names.joblib")
DOCS_PATH = Path("docs")


def load_features(filepath: Path = FEATURES_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the engineered feature set and split into X and y.

    Args:
        filepath: Path to the features CSV file.

    Returns:
        Tuple of (X features DataFrame, y target Series).

    Raises:
        FileNotFoundError: If features file doesn't exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Features not found at {filepath}. "
            "Please run src/feature_extractor.py first."
        )

    print(f"Loading features from {filepath}...")
    df = pd.read_csv(filepath)

    X = df.drop(columns=['FLAG'])
    y = df['FLAG']

    print(f"Features shape: {X.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    return X, y


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to balance classes in the training set.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_resampled, y_resampled) as numpy arrays.
    """
    print("\nApplying SMOTE to balance classes...")
    print(f"  Before: {dict(pd.Series(y).value_counts())}")

    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X, y)

    print(f"  After:  {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list
) -> XGBClassifier:
    """
    Tune XGBoost hyperparameters using GridSearchCV with 5-fold
    Stratified Cross-Validation.

    WHY STRATIFIED K-FOLD?
    Regular K-Fold might put all fraud cases in one fold.
    Stratified ensures each fold has the same fraud/legit ratio
    as the overall dataset — critical for imbalanced data.

    Args:
        X_train: Training features as numpy array.
        y_train: Training labels as numpy array.
        feature_names: List of feature names for the model.

    Returns:
        Best XGBoost model found by GridSearchCV.
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
    print("="*60)

    # Define the hyperparameter grid to search
    # Each combination will be evaluated with 5-fold CV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }

    # Base XGBoost model
    base_model = XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        feature_names_in=feature_names,
    )

    # 5-fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\nSearching {3*3*3*2*2} parameter combinations...")
    print("This will take 3-5 minutes — this is normal.\n")

    # GridSearchCV with ROC-AUC scoring
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,          # Use all CPU cores
        verbose=1,          # Show progress
        refit=True          # Retrain best model on full training data
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest CV ROC-AUC: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def evaluate_model(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list
) -> Dict[str, float]:
    """
    Evaluate the trained model on the test set.
    Prints a full classification report and saves evaluation plots.

    Args:
        model: Trained XGBoost model.
        X_test: Test features.
        y_test: True test labels.
        feature_names: List of feature names.

    Returns:
        Dictionary of evaluation metrics.
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION ON TEST SET")
    print("="*60)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")

    if roc_auc >= 0.95:
        print("  ✓ TARGET ACHIEVED: ROC-AUC >= 0.95")
    else:
        print(f"  ⚠ ROC-AUC below 0.95 target (got {roc_auc:.4f})")

    # Full classification report
    print(f"\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Legitimate', 'Fraud']
    ))

    # Cross-validation scores on full dataset
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_test, y_test,
        cv=cv, scoring='roc_auc'
    )
    print(f"5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # Save evaluation plots
    DOCS_PATH.mkdir(exist_ok=True)
    _save_confusion_matrix(y_test, y_pred)
    _save_roc_curve(model, X_test, y_test)

    metrics = {
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    return metrics


def _save_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """Save confusion matrix plot to docs folder."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Legitimate', 'Fraud']
    )
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix — Fraud Detection Model', fontsize=13)
    plt.tight_layout()
    save_path = DOCS_PATH / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def _save_roc_curve(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> None:
    """Save ROC curve plot to docs folder."""
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title('ROC Curve — Fraud Detection Model', fontsize=13)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.legend()
    plt.tight_layout()
    save_path = DOCS_PATH / 'roc_curve.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path}")


def generate_shap_plots(
    model: XGBClassifier,
    X_test: np.ndarray,
    feature_names: list
) -> None:
    """
    Generate and save SHAP explainability plots.

    WHY SHAP?
    SHAP (SHapley Additive exPlanations) tells us WHY the model
    made each prediction. For fraud detection, this is critical —
    you can't just say 'this wallet is fraudulent', you need to
    explain WHICH features made it look suspicious.

    This is your biggest resume differentiator — most students
    just train models, you explain them.

    Plots generated:
        - shap_beeswarm.png: Global feature importance across all samples
        - shap_waterfall.png: Local explanation for one high-risk prediction

    Args:
        model: Trained XGBoost model.
        X_test: Test features as numpy array.
        feature_names: List of feature names.
    """
    print("\n" + "="*60)
    print("GENERATING SHAP EXPLAINABILITY PLOTS")
    print("="*60)

    print("\nCalculating SHAP values (this takes ~1 minute)...")

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for test set
    # Use a sample of 500 for speed — still representative
    sample_size = min(500, len(X_test))
    X_sample = X_test[:sample_size]

    shap_values = explainer.shap_values(X_sample)

    # ── Plot 1: Beeswarm plot (global feature importance) ─────────────────────
    print("\nGenerating beeswarm plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        max_display=20
    )
    plt.title('SHAP Beeswarm — Global Feature Importance', fontsize=13)
    plt.tight_layout()
    beeswarm_path = DOCS_PATH / 'shap_beeswarm.png'
    plt.savefig(beeswarm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Beeswarm plot saved to {beeswarm_path}")

    # ── Plot 2: Waterfall plot (local explanation for highest risk wallet) ────
    print("\nGenerating waterfall plot...")

    # Find the sample with highest fraud probability
    # This is the most interesting case to explain
    fraud_probs = model.predict_proba(X_sample)[:, 1]
    highest_risk_idx = np.argmax(fraud_probs)

    # Create SHAP explanation object
    explanation = shap.Explanation(
        values=shap_values[highest_risk_idx],
        base_values=explainer.expected_value,
        data=X_sample[highest_risk_idx],
        feature_names=feature_names
    )

    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(explanation, show=False, max_display=15)
    plt.title(
        f'SHAP Waterfall — Highest Risk Wallet '
        f'(Fraud Probability: {fraud_probs[highest_risk_idx]:.1%})',
        fontsize=13
    )
    plt.tight_layout()
    waterfall_path = DOCS_PATH / 'shap_waterfall.png'
    plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Waterfall plot saved to {waterfall_path}")

    # Print top 5 most important features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True
    )
    print(f"\nTop 10 Most Important Features (by SHAP):")
    for i, (feat, importance) in enumerate(feature_importance[:10], 1):
        print(f"  {i:2}. {feat:<35} SHAP: {importance:.4f}")


def save_model_artifacts(
    model: XGBClassifier,
    scaler: StandardScaler,
    feature_names: list
) -> None:
    """
    Save all model artifacts needed for the Flask API.

    We save 3 files:
    - fraud_detector.joblib: The trained XGBoost model
    - scaler.joblib: The StandardScaler (must use same scaler in API)
    - feature_names.joblib: List of feature names in correct order

    Args:
        model: Trained XGBoost model.
        scaler: Fitted StandardScaler.
        feature_names: Ordered list of feature names.
    """
    MODEL_SAVE_PATH.parent.mkdir(exist_ok=True)

    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Scaler saved to {SCALER_SAVE_PATH}")

    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    print(f"Feature names saved to {FEATURE_NAMES_PATH}")


def run_training_pipeline() -> None:
    """
    Run the complete model training pipeline:
    1. Load features
    2. Apply SMOTE
    3. Train/test split
    4. Hyperparameter tuning with GridSearchCV
    5. Model evaluation
    6. SHAP explainability plots
    7. Save model artifacts
    """
    print("Starting model training pipeline...")
    print("="*60)

    # Step 1: Load features
    X, y = load_features()

    # Step 2: Train/test split BEFORE SMOTE
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")

    # Step 3: Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    feature_names = list(X.columns)

    # Step 4: Apply SMOTE only to training set
    X_train_balanced, y_train_balanced = apply_smote(
        pd.DataFrame(X_train_scaled, columns=feature_names),
        y_train
    )

    # Step 5: Hyperparameter tuning
    best_model = tune_hyperparameters(
        X_train_balanced.values,
        y_train_balanced.values,
        feature_names
    )

    # Step 6: Evaluate on test set
    metrics = evaluate_model(
        best_model,
        X_test_scaled,
        y_test.values,
        feature_names
    )

    # Step 7: SHAP plots
    generate_shap_plots(
        best_model,
        X_test_scaled,
        feature_names
    )

    # Step 8: Save everything
    print("\n" + "="*60)
    print("SAVING MODEL ARTIFACTS")
    print("="*60)
    save_model_artifacts(best_model, scaler, feature_names)

    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE ✓")
    print("="*60)
    print(f"Final ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print("Next step: Run api/app.py")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_training_pipeline()