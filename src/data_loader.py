"""
data_loader.py
--------------
Handles loading, merging, cleaning, and exploring the Ethereum fraud dataset.
This is the first step in the ML pipeline — garbage in, garbage out.

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_PATH = Path("data/raw/transaction_dataset.csv")
PROCESSED_DATA_PATH = Path("data/processed/cleaned_transactions.csv")
FIGURES_PATH = Path("docs")


def load_raw_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Ethereum fraud dataset from CSV.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        Raw DataFrame with all original columns.

    Raises:
        FileNotFoundError: If the CSV file doesn't exist at the given path.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Please download it from: "
            "https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset "
            "and place transaction_dataset.csv in data/raw/"
        )

    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """
    Print a full exploratory data analysis summary.
    This is always the FIRST thing you do with any new dataset.

    Args:
        df: Raw DataFrame to explore.
    """
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)

    # Basic shape
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Column names
    print(f"\nColumns:\n{list(df.columns)}")

    # Data types
    print(f"\nData Types:\n{df.dtypes}")

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    print(f"\nMissing Values:\n{missing_df[missing_df['Missing Count'] > 0]}")

    # Class distribution — this is critical for fraud detection
    if 'FLAG' in df.columns:
        print(f"\nClass Distribution (FLAG column):")
        counts = df['FLAG'].value_counts()
        pcts = df['FLAG'].value_counts(normalize=True) * 100
        for label, count, pct in zip(counts.index, counts.values, pcts.values):
            label_str = "FRAUD" if label == 1 else "LEGITIMATE"
            print(f"  {label_str} ({label}): {count:,} ({pct:.1f}%)")

    # Basic statistics
    print(f"\nNumerical Summary:\n{df.describe().round(2)}")
    print("="*60)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
    - Drop irrelevant columns
    - Handle missing values
    - Fix data types
    - Remove duplicates

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    print("\nCleaning data...")
    original_len = len(df)

    # Make a copy — never modify the original
    df = df.copy()

    # Drop the address column — it's an identifier, not a feature
    # Also drop unnamed index columns that sometimes appear in CSVs
    cols_to_drop = [col for col in df.columns if
                    'Unnamed' in str(col) or
                    col.lower() == 'address' or
                    col.lower() == 'index']

    if cols_to_drop:
        print(f"  Dropping identifier columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop, errors='ignore')

    # Remove duplicate rows
    df = df.drop_duplicates()
    print(f"  Removed {original_len - len(df):,} duplicate rows")

    # Handle missing values
    # For numerical columns: fill with median (robust to outliers)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Don't fill the target column FLAG
    fill_cols = [c for c in numerical_cols if c != 'FLAG']

    for col in fill_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Filled {col} nulls with median: {median_val:.4f}")

    # Ensure FLAG column is integer (0 or 1)
    if 'FLAG' in df.columns:
        df['FLAG'] = df['FLAG'].astype(int)

    print(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("Data cleaning complete ✓")
    return df


def plot_class_distribution(df: pd.DataFrame) -> None:
    """
    Plot and save the class distribution chart.
    Visualizing imbalance is important before deciding how to handle it.

    Args:
        df: Cleaned DataFrame with FLAG column.
    """
    FIGURES_PATH.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Ethereum Fraud Dataset — Class Distribution', fontsize=14)

    counts = df['FLAG'].value_counts()
    labels = ['Legitimate (0)', 'Fraud (1)']
    colors = ['#3B8BD4', '#E24B4A']

    # Bar chart
    axes[0].bar(labels, counts.values, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_title('Transaction Count by Class')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')

    # Pie chart
    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Class Percentage Split')

    plt.tight_layout()
    save_path = FIGURES_PATH / 'class_distribution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nClass distribution plot saved to {save_path}")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Plot and save a correlation heatmap of all numerical features.
    Helps identify which features are most correlated with fraud.

    Args:
        df: Cleaned DataFrame.
    """
    FIGURES_PATH.mkdir(exist_ok=True)

    # Select only numerical columns
    num_df = df.select_dtypes(include=[np.number])

    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
    sns.heatmap(
        num_df.corr(),
        mask=mask,
        annot=False,
        cmap='coolwarm',
        center=0,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8}
    )
    plt.title('Feature Correlation Heatmap', fontsize=14)
    plt.tight_layout()

    save_path = FIGURES_PATH / 'correlation_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Correlation heatmap saved to {save_path}")
    plt.show()


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.

    WHY SMOTE?
    Fraud datasets are heavily imbalanced (e.g. 90% legitimate, 10% fraud).
    If we train on this as-is, the model learns to always predict 'legitimate'
    and still gets 90% accuracy — but misses all fraud. SMOTE fixes this by
    synthetically generating new fraud examples.

    Args:
        X: Feature DataFrame.
        y: Target Series (FLAG column).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_resampled, y_resampled) with balanced classes.
    """
    print(f"\nApplying SMOTE...")
    print(f"  Before SMOTE: {y.value_counts().to_dict()}")

    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"  After SMOTE:  {pd.Series(y_resampled).value_counts().to_dict()}")
    print("SMOTE complete ✓")

    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)


def get_train_test_split(
    df: pd.DataFrame,
    target_col: str = 'FLAG',
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train/test sets and optionally apply SMOTE to training set.

    IMPORTANT: We apply SMOTE ONLY to the training set.
    Never apply SMOTE to the test set — it would give fake evaluation results.

    Args:
        df: Cleaned DataFrame.
        target_col: Name of the target column.
        test_size: Fraction of data for testing (0.2 = 20%).
        random_state: Seed for reproducibility.
        use_smote: Whether to apply SMOTE to the training set.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split BEFORE applying SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Ensures same class ratio in train and test
    )

    print(f"\nTrain/Test Split:")
    print(f"  Training set:  {len(X_train):,} samples")
    print(f"  Test set:      {len(X_test):,} samples")

    # Apply SMOTE only to training data
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train, random_state)

    return X_train, X_test, y_train, y_test


def save_cleaned_data(df: pd.DataFrame) -> None:
    """
    Save the cleaned DataFrame to the processed data folder.

    Args:
        df: Cleaned DataFrame to save.
    """
    PROCESSED_DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\nCleaned data saved to {PROCESSED_DATA_PATH}")


def run_pipeline() -> pd.DataFrame:
    """
    Run the complete data loading and cleaning pipeline.
    Call this function to execute all steps in order.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    print("Starting data pipeline...")

    # Step 1: Load
    df = load_raw_data()

    # Step 2: Explore
    explore_data(df)

    # Step 3: Clean
    df = clean_data(df)

    # Step 4: Visualize
    plot_class_distribution(df)
    plot_correlation_heatmap(df)

    # Step 5: Save
    save_cleaned_data(df)

    print("\nData pipeline complete! ✓")
    print(f"Next step: Run src/feature_extractor.py")
    return df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = run_pipeline()