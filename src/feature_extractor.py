"""
feature_extractor.py
--------------------
Engineers 25+ features from raw Ethereum wallet transaction data.
These features capture behavioral patterns that distinguish fraudulent
wallets from legitimate ones.

"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

# ── Paths ─────────────────────────────────────────────────────────────────────
CLEANED_DATA_PATH = Path("data/processed/cleaned_transactions.csv")
FEATURES_DATA_PATH = Path("data/processed/features.csv")


def load_cleaned_data(filepath: Path = CLEANED_DATA_PATH) -> pd.DataFrame:
    """
    Load the cleaned dataset produced by data_loader.py.

    Args:
        filepath: Path to the cleaned CSV file.

    Returns:
        Cleaned DataFrame ready for feature extraction.

    Raises:
        FileNotFoundError: If cleaned data doesn't exist yet.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {filepath}. "
            "Please run src/data_loader.py first."
        )
    print(f"Loading cleaned data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df


def feature_transaction_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Group 1: Transaction frequency and volume features.

    These capture HOW OFTEN a wallet transacts — fraudsters tend to
    send many small transactions quickly (smurfing) or very few large ones.

    Features added:
        - total_transactions: Total number of transactions
        - sent_received_ratio: Ratio of sent to received transactions
        - transaction_density: Transactions per unit of active time

    Args:
        df: Input DataFrame with raw transaction columns.

    Returns:
        DataFrame with new frequency features added.
    """
    print("  Engineering transaction frequency features...")
    df = df.copy()

    # Total transactions
    df['total_transactions'] = df['Sent tnx'] + df['Received Tnx']

    # Sent to received ratio — fraudsters often send much more than they receive
    # Add small epsilon to avoid division by zero
    df['sent_received_ratio'] = df['Sent tnx'] / (df['Received Tnx'] + 1e-9)

    # Transaction density — how busy is this wallet per minute of activity?
    active_time = df['Time Diff between first and last (Mins)'].replace(0, 1)
    df['transaction_density'] = df['total_transactions'] / active_time

    return df


def feature_value_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Group 2: Statistical features about transaction values (in Ether).

    Fraudulent wallets often have unusual value distributions —
    either very consistent small amounts (automated) or extreme outliers.

    Features added:
        - value_range_sent: Difference between max and min sent value
        - value_range_received: Difference between max and min received value
        - avg_value_ratio: Ratio of avg sent to avg received value
        - total_ether_moved: Total ether flowing through the wallet
        - balance_ratio: Balance relative to total ether moved

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with new value statistical features added.
    """
    print("  Engineering value statistics features...")
    df = df.copy()

    # Range of sent values — low range = automated/scripted behavior
    df['value_range_sent'] = df['max val sent'] - df['min val sent']

    # Range of received values
    df['value_range_received'] = (
        df['max value received '] - df['min value received']
    )

    # Ratio of average sent to average received
    df['avg_value_ratio'] = (
        df['avg val sent'] / (df['avg val received'] + 1e-9)
    )

    # Total ether moved through wallet (in + out)
    df['total_ether_moved'] = (
        df['total Ether sent'] + df['total ether received']
    )

    # Balance ratio — what fraction of moved ether is retained?
    # Fraudsters often drain wallets quickly (low balance ratio)
    df['balance_ratio'] = (
        df['total ether balance'] / (df['total_ether_moved'] + 1e-9)
    )

    return df


def feature_network_graph(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Group 3: Network/graph features based on wallet connectivity.

    In graph terms, each wallet is a NODE and each transaction is an EDGE.
    Fraudulent wallets have unusual connectivity patterns.

    Features added:
        - fan_in: Number of unique addresses sending TO this wallet
        - fan_out: Number of unique addresses this wallet sends TO
        - fan_in_out_ratio: Ratio of fan_in to fan_out
        - unique_address_ratio: Total unique counterparties / total transactions
        - is_hub: Boolean — does this wallet connect to many unique addresses?

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with new network graph features added.
    """
    print("  Engineering network graph features...")
    df = df.copy()

    # Fan-in: unique senders to this wallet
    df['fan_in'] = df['Unique Received From Addresses']

    # Fan-out: unique receivers from this wallet
    df['fan_out'] = df['Unique Sent To Addresses']

    # Fan-in/out ratio — money mules have high fan-in (receive from many)
    # Scammers have high fan-out (send to many victims)
    df['fan_in_out_ratio'] = df['fan_in'] / (df['fan_out'] + 1e-9)

    # Unique address ratio — what fraction of txns are to/from unique addresses?
    total_unique = df['fan_in'] + df['fan_out']
    df['unique_address_ratio'] = total_unique / (df['total_transactions'] + 1e-9)

    # Hub detection — wallets with unusually many connections
    # Using median as threshold (robust to outliers)
    hub_threshold = total_unique.median()
    df['is_hub'] = (total_unique > hub_threshold).astype(int)

    return df


def feature_time_behavior(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Group 4: Time-based behavioral features.

    Fraudulent wallets often have very short active lifespans —
    they're created, used for fraud, then abandoned.

    Features added:
        - active_lifespan_hours: How long the wallet has been active (hours)
        - avg_time_between_sent: Average minutes between outgoing transactions
        - avg_time_between_received: Average minutes between incoming transactions
        - time_regularity: How regular/automated the transaction timing is
        - is_short_lived: Flag for wallets active less than 24 hours

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with new time-based features added.
    """
    print("  Engineering time-based behavioral features...")
    df = df.copy()

    # Active lifespan in hours
    df['active_lifespan_hours'] = (
        df['Time Diff between first and last (Mins)'] / 60
    )

    # Average time between sent transactions
    df['avg_time_between_sent'] = df['Avg min between sent tnx'].fillna(0)

    # Average time between received transactions
    df['avg_time_between_received'] = (
        df['Avg min between received tnx'].fillna(0)
    )

    # Time regularity — low std in timing suggests automated/bot behavior
    # Approximated as ratio of avg sent time to avg received time
    df['time_regularity'] = (
        df['avg_time_between_sent'] /
        (df['avg_time_between_received'] + 1e-9)
    )

    # Short-lived wallet flag — active less than 24 hours (1440 minutes)
    df['is_short_lived'] = (
        df['Time Diff between first and last (Mins)'] < 1440
    ).astype(int)

    return df


def feature_contract_behavior(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Group 5: Smart contract interaction features.

    Fraudulent wallets often interact with contracts differently —
    either heavily (for token theft) or not at all (simple scams).

    Features added:
        - contract_interaction_rate: Fraction of txns involving contracts
        - contract_value_ratio: Fraction of ether sent to contracts
        - created_contracts_flag: Whether this wallet creates contracts
        - erc20_activity_rate: ERC20 token transaction rate

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with new contract behavior features added.
    """
    print("  Engineering contract behavior features...")
    df = df.copy()

    # Contract interaction rate
    total_txns = df['total_transactions'].replace(0, 1)
    df['contract_interaction_rate'] = (
        df['Number of Created Contracts'] / total_txns
    )

    # What fraction of sent ether goes to contracts?
    total_sent = df['total Ether sent'].replace(0, 1e-9)
    df['contract_value_ratio'] = (
        df['total ether sent contracts'] / total_sent
    )

    # Binary flag: does this wallet deploy contracts?
    df['created_contracts_flag'] = (
        df['Number of Created Contracts'] > 0
    ).astype(int)

    # ERC20 activity — token transfers (DeFi fraud often uses ERC20)
    df['erc20_activity_rate'] = (
        df[' Total ERC20 tnxs'] / total_txns
    )

    return df


def feature_erc20_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Group 6: ERC20 token-specific patterns.

    ERC20 features capture token-based fraud patterns like
    rug pulls, token dumping, and wash trading.

    Features added:
        - erc20_sent_received_ratio: Token send/receive balance
        - erc20_unique_tokens_sent: Diversity of tokens sent
        - erc20_unique_tokens_received: Diversity of tokens received
        - erc20_token_diversity: Total unique tokens interacted with
        - erc20_avg_value_ratio: Ratio of avg ERC20 sent to received value

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with new ERC20 pattern features added.
    """
    print("  Engineering ERC20 token pattern features...")
    df = df.copy()

    # ERC20 sent vs received ratio
    erc20_received = df[' ERC20 total Ether received'].replace(0, 1e-9)
    df['erc20_sent_received_ratio'] = (
        df[' ERC20 total ether sent'] / erc20_received
    )

    # Token diversity — scammers often use many different tokens
    df['erc20_unique_tokens_sent'] = df[' ERC20 uniq sent token name'].fillna(0)
    df['erc20_unique_tokens_received'] = (
        df[' ERC20 uniq rec token name'].fillna(0)
    )
    df['erc20_token_diversity'] = (
        df['erc20_unique_tokens_sent'] + df['erc20_unique_tokens_received']
    )

    # ERC20 value ratio
    erc20_avg_rec = df[' ERC20 avg val rec'].replace(0, 1e-9)
    df['erc20_avg_value_ratio'] = (
        df[' ERC20 avg val sent'] / erc20_avg_rec
    )

    return df


def feature_risk_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Group 7: Composite risk indicator features.

    These combine multiple signals into higher-level risk scores
    that directly capture known fraud patterns.

    Features added:
        - high_value_low_frequency: Big transactions but rarely active
        - rapid_drain_indicator: Sends much more than receives
        - dormancy_score: Long gaps between transactions
        - anomaly_score: Combined zscore-based anomaly indicator

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with composite risk indicator features added.
    """
    print("  Engineering composite risk indicator features...")
    df = df.copy()

    # High value but low frequency — could be a hack/theft
    df['high_value_low_frequency'] = (
        df['avg val sent'] * (1 / (df['transaction_density'] + 1e-9))
    )
    # Log-scale to handle extreme values
    df['high_value_low_frequency'] = np.log1p(
        df['high_value_low_frequency']
    )

    # Rapid drain indicator — sends way more than it receives
    df['rapid_drain_indicator'] = (
        df['total Ether sent'] /
        (df['total ether received'] + 1e-9)
    ).clip(0, 100)  # Cap at 100 to handle extreme outliers

    # Dormancy score — long average time between transactions
    df['dormancy_score'] = np.log1p(
        df['avg_time_between_sent'] + df['avg_time_between_received']
    )

    # Anomaly score — z-score based composite
    # Measures how far from "normal" this wallet's behavior is
    key_features = ['sent_received_ratio', 'transaction_density',
                    'fan_in_out_ratio', 'balance_ratio']
    zscores = pd.DataFrame()
    for feat in key_features:
        mean = df[feat].mean()
        std = df[feat].std()
        if std > 0:
            zscores[feat] = ((df[feat] - mean) / std).abs()
        else:
            zscores[feat] = 0
    df['anomaly_score'] = zscores.mean(axis=1)

    return df


def drop_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop original raw columns that have been replaced by engineered features.
    Keep only the engineered features and the target FLAG column.

    Args:
        df: DataFrame with both raw and engineered features.

    Returns:
        DataFrame with only engineered features + FLAG column.
    """
    print("  Selecting final feature set...")

    # Columns to keep — our 25+ engineered features + target
    engineered_features = [
        # Group 1: Transaction frequency
        'total_transactions',
        'sent_received_ratio',
        'transaction_density',
        # Group 2: Value statistics
        'value_range_sent',
        'value_range_received',
        'avg_value_ratio',
        'total_ether_moved',
        'balance_ratio',
        # Group 3: Network graph
        'fan_in',
        'fan_out',
        'fan_in_out_ratio',
        'unique_address_ratio',
        'is_hub',
        # Group 4: Time behavior
        'active_lifespan_hours',
        'avg_time_between_sent',
        'avg_time_between_received',
        'time_regularity',
        'is_short_lived',
        # Group 5: Contract behavior
        'contract_interaction_rate',
        'contract_value_ratio',
        'created_contracts_flag',
        'erc20_activity_rate',
        # Group 6: ERC20 patterns
        'erc20_sent_received_ratio',
        'erc20_unique_tokens_sent',
        'erc20_unique_tokens_received',
        'erc20_token_diversity',
        'erc20_avg_value_ratio',
        # Group 7: Risk indicators
        'high_value_low_frequency',
        'rapid_drain_indicator',
        'dormancy_score',
        'anomaly_score',
        # Target column — always keep last
        'FLAG',
    ]

    # Only keep columns that exist in the dataframe
    available = [c for c in engineered_features if c in df.columns]
    missing = [c for c in engineered_features if c not in df.columns]

    if missing:
        print(f"  Warning: These features not found: {missing}")

    return df[available]


def validate_features(df: pd.DataFrame) -> None:
    """
    Validate the final feature set for quality issues.
    Prints warnings for any features with problems.

    Args:
        df: Final feature DataFrame to validate.
    """
    print("\n" + "="*60)
    print("FEATURE VALIDATION REPORT")
    print("="*60)

    print(f"\nTotal features: {len(df.columns) - 1} (excluding FLAG target)")
    print(f"Total samples:  {len(df):,}")

    issues_found = False

    for col in df.columns:
        if col == 'FLAG':
            continue

        null_count = df[col].isnull().sum()
        inf_count = np.isinf(df[col]).sum()

        if null_count > 0:
            print(f"  WARNING: {col} has {null_count} null values")
            issues_found = True

        if inf_count > 0:
            print(f"  WARNING: {col} has {inf_count} infinite values")
            issues_found = True

    if not issues_found:
        print("\n  All features passed validation ✓")
        print("  No nulls, no infinite values")

    # Final class distribution after cleaning
    print(f"\nClass distribution in final feature set:")
    counts = df['FLAG'].value_counts()
    for label, count in counts.items():
        pct = count / len(df) * 100
        label_str = "FRAUD" if label == 1 else "LEGITIMATE"
        print(f"  {label_str}: {count:,} ({pct:.1f}%)")

    print("="*60)


def run_feature_pipeline() -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.
    Applies all 7 feature groups in sequence.

    Returns:
        Final feature DataFrame with 30+ engineered features + FLAG column.
    """
    print("Starting feature engineering pipeline...")
    print("="*60)

    # Load cleaned data
    df = load_cleaned_data()

    # Apply all feature groups in sequence
    print("\nEngineering features:")
    df = feature_transaction_frequency(df)
    df = feature_value_statistics(df)
    df = feature_network_graph(df)
    df = feature_time_behavior(df)
    df = feature_contract_behavior(df)
    df = feature_erc20_patterns(df)
    df = feature_risk_indicators(df)

    # Keep only engineered features
    df = drop_raw_columns(df)

    # Fix any remaining infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # Validate
    validate_features(df)

    # Save
    FEATURES_DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(FEATURES_DATA_PATH, index=False)
    print(f"\nFeature set saved to {FEATURES_DATA_PATH}")
    print("\nFeature engineering complete! ✓")
    print("Next step: Run src/train_model.py")

    return df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = run_feature_pipeline()