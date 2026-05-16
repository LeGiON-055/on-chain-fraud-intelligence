"""
api/app.py
----------
Flask REST API for the On-Chain Fraud Intelligence Network.
Exposes 3 endpoints:
    POST /analyze/wallet  — fetch transactions, run ML model, return risk score
    POST /explain         — call Groq LLaMA 3 70B for plain English explanation
    GET  /health          — health check

"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

# Add project root to path so we can import config
sys.path.append(str(Path(__file__).parent.parent))
import config

# ── Flask app setup ───────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = config.FLASK_SECRET_KEY

# Allow requests from any origin (needed for MetaMask frontend)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Load model artifacts at startup ───────────────────────────────────────────
print("Loading model artifacts...")
try:
    MODEL = joblib.load(config.MODEL_PATH)
    SCALER = joblib.load("models/scaler.joblib")
    FEATURE_NAMES = joblib.load(config.FEATURES_PATH)
    print(f"Model loaded ✓ ({len(FEATURE_NAMES)} features)")
except Exception as e:
    print(f"ERROR loading model: {e}")
    MODEL = None
    SCALER = None
    FEATURE_NAMES = []

# ── Groq client setup ─────────────────────────────────────────────────────────
try:
    groq_client = Groq(api_key=config.GROQ_API_KEY)
    print("Groq client initialized ✓")
except Exception as e:
    print(f"WARNING: Groq client failed: {e}")
    groq_client = None

# ── Recent scans store (simple in-memory + JSON file) ─────────────────────────
RECENT_SCANS_FILE = Path("data/recent_scans.json")
MAX_RECENT_SCANS = 10


def load_recent_scans() -> list:
    """Load recent scans from JSON file."""
    if RECENT_SCANS_FILE.exists():
        try:
            with open(RECENT_SCANS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_recent_scan(address: str, risk_score: float, verdict: str) -> None:
    """
    Save a scan result to the recent scans JSON file.
    Keeps only the last MAX_RECENT_SCANS entries.

    Args:
        address: Ethereum wallet address scanned.
        risk_score: Model fraud probability (0-1).
        verdict: Risk verdict string.
    """
    scans = load_recent_scans()
    scans.insert(0, {
        'address': address,
        'risk_score': round(risk_score, 4),
        'verdict': verdict,
        'timestamp': datetime.utcnow().isoformat()
    })
    # Keep only the most recent scans
    scans = scans[:MAX_RECENT_SCANS]
    RECENT_SCANS_FILE.parent.mkdir(exist_ok=True)
    with open(RECENT_SCANS_FILE, 'w') as f:
        json.dump(scans, f, indent=2)


# ── Helper functions ──────────────────────────────────────────────────────────

def validate_ethereum_address(address: str) -> bool:
    """
    Validate that a string is a valid Ethereum address.
    Must start with 0x and be 42 characters long.

    Args:
        address: String to validate.

    Returns:
        True if valid Ethereum address, False otherwise.
    """
    if not address:
        return False
    if not address.startswith('0x'):
        return False
    if len(address) != 42:
        return False
    # Check all characters after 0x are valid hex
    try:
        int(address[2:], 16)
        return True
    except ValueError:
        return False


def fetch_transactions_from_etherscan(address: str) -> list:
    """
    Fetch the last 100 transactions for an Ethereum address
    from the Etherscan API V2.

    Args:
        address: Valid Ethereum wallet address.

    Returns:
        List of transaction dictionaries from Etherscan.

    Raises:
        Exception: If Etherscan API call fails.
    """
    import requests

    # Etherscan V2 API endpoint
    url = f"https://api.etherscan.io/v2/api"

    params = {
        'chainid': 1,           # Ethereum mainnet
        'module': 'account',
        'action': 'txlist',
        'address': address,
        'startblock': 0,
        'endblock': 99999999,
        'page': 1,
        'offset': 100,
        'sort': 'desc',
        'apikey': config.ETHERSCAN_API_KEY
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )
    response.raise_for_status()
    data = response.json()

    if data['status'] == '0':
        if data['message'] == 'No transactions found':
            return []
        raise Exception(f"Etherscan API error: {data['message']}")

    return data['result']


def extract_features_from_transactions(
    transactions: list,
    address: str
) -> pd.DataFrame:
    """
    Extract the same 31 features from live transaction data
    that were used during model training.

    This mirrors the logic in feature_extractor.py but works
    on raw Etherscan API response data.

    Args:
        transactions: List of transaction dicts from Etherscan API.
        address: The wallet address being analyzed.

    Returns:
        Single-row DataFrame with all 31 features.
    """
    address_lower = address.lower()

    if not transactions:
        # Return zero features for wallets with no transactions
        return pd.DataFrame(
            [dict.fromkeys(FEATURE_NAMES, 0.0)]
        )

    # Convert to DataFrame for easier processing
    df = pd.DataFrame(transactions)

    # Convert value from Wei to Ether (1 ETH = 10^18 Wei)
    df['value_eth'] = pd.to_numeric(df['value'], errors='coerce') / 1e18
    df['timeStamp'] = pd.to_numeric(df['timeStamp'], errors='coerce')
    df['gasUsed'] = pd.to_numeric(df['gasUsed'], errors='coerce')

    # Identify sent and received transactions
    sent_mask = df['from'].str.lower() == address_lower
    recv_mask = df['to'].str.lower() == address_lower

    sent_txns = df[sent_mask]
    recv_txns = df[recv_mask]

    # ── Calculate all features ────────────────────────────────────────────────

    # Time features
    if len(df) > 1:
        time_diff_mins = (
            df['timeStamp'].max() - df['timeStamp'].min()
        ) / 60
    else:
        time_diff_mins = 0

    active_lifespan_hours = time_diff_mins / 60

    # Transaction counts
    sent_count = len(sent_txns)
    recv_count = len(recv_txns)
    total_txns = len(df)

    # Value statistics
    sent_values = sent_txns['value_eth'].dropna()
    recv_values = recv_txns['value_eth'].dropna()

    avg_val_sent = sent_values.mean() if len(sent_values) > 0 else 0
    avg_val_recv = recv_values.mean() if len(recv_values) > 0 else 0
    min_val_sent = sent_values.min() if len(sent_values) > 0 else 0
    max_val_sent = sent_values.max() if len(sent_values) > 0 else 0
    min_val_recv = recv_values.min() if len(recv_values) > 0 else 0
    max_val_recv = recv_values.max() if len(recv_values) > 0 else 0

    total_sent = sent_values.sum()
    total_recv = recv_values.sum()
    total_moved = total_sent + total_recv
    balance = total_recv - total_sent

    # Network features
    unique_senders = df[recv_mask]['from'].nunique()
    unique_receivers = df[sent_mask]['to'].nunique()
    fan_in = unique_senders
    fan_out = unique_receivers

    # Time between transactions
    if len(sent_txns) > 1:
        sent_times = sent_txns['timeStamp'].sort_values()
        avg_time_between_sent = sent_times.diff().mean() / 60
    else:
        avg_time_between_sent = 0

    if len(recv_txns) > 1:
        recv_times = recv_txns['timeStamp'].sort_values()
        avg_time_between_recv = recv_times.diff().mean() / 60
    else:
        avg_time_between_recv = 0

    # ── Build feature dictionary matching training features exactly ───────────
    eps = 1e-9  # Avoid division by zero

    features = {
        # Group 1: Transaction frequency
        'total_transactions': total_txns,
        'sent_received_ratio': sent_count / (recv_count + eps),
        'transaction_density': total_txns / (time_diff_mins + eps),

        # Group 2: Value statistics
        'value_range_sent': max_val_sent - min_val_sent,
        'value_range_received': max_val_recv - min_val_recv,
        'avg_value_ratio': avg_val_sent / (avg_val_recv + eps),
        'total_ether_moved': total_moved,
        'balance_ratio': balance / (total_moved + eps),

        # Group 3: Network graph
        'fan_in': fan_in,
        'fan_out': fan_out,
        'fan_in_out_ratio': fan_in / (fan_out + eps),
        'unique_address_ratio': (fan_in + fan_out) / (total_txns + eps),
        'is_hub': int((fan_in + fan_out) > 10),

        # Group 4: Time behavior
        'active_lifespan_hours': active_lifespan_hours,
        'avg_time_between_sent': avg_time_between_sent,
        'avg_time_between_received': avg_time_between_recv,
        'time_regularity': avg_time_between_sent / (avg_time_between_recv + eps),
        'is_short_lived': int(time_diff_mins < 1440),

        # Group 5: Contract behavior
        'contract_interaction_rate': 0.0,
        'contract_value_ratio': 0.0,
        'created_contracts_flag': 0,
        'erc20_activity_rate': 0.0,

        # Group 6: ERC20 patterns
        'erc20_sent_received_ratio': 0.0,
        'erc20_unique_tokens_sent': 0.0,
        'erc20_unique_tokens_received': 0.0,
        'erc20_token_diversity': 0.0,
        'erc20_avg_value_ratio': 0.0,

        # Group 7: Risk indicators
        'high_value_low_frequency': float(
            np.log1p(avg_val_sent * (1 / (total_txns / (time_diff_mins + eps) + eps)))
        ),
        'rapid_drain_indicator': min(total_sent / (total_recv + eps), 100),
        'dormancy_score': float(
            np.log1p(avg_time_between_sent + avg_time_between_recv)
        ),
        'anomaly_score': 0.0,
    }

    return pd.DataFrame([features])


def get_risk_verdict(risk_score: float) -> dict:
    """
    Convert a fraud probability score to a human-readable verdict.

    Args:
        risk_score: Model output probability of fraud (0.0 to 1.0).

    Returns:
        Dictionary with verdict, color, and description.
    """
    if risk_score >= config.RISK_HIGH:
        return {
            'verdict': 'HIGH RISK',
            'color': 'red',
            'emoji': '🔴',
            'description': 'This wallet shows strong indicators of fraudulent activity.'
        }
    elif risk_score >= config.RISK_MEDIUM:
        return {
            'verdict': 'SUSPICIOUS',
            'color': 'amber',
            'emoji': '🟡',
            'description': 'This wallet has some unusual patterns. Proceed with caution.'
        }
    else:
        return {
            'verdict': 'SAFE',
            'color': 'green',
            'emoji': '🟢',
            'description': 'This wallet appears to be legitimate.'
        }


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health_check():
    """
    GET /health
    Health check endpoint — used by monitoring systems and the frontend
    to verify the API is running correctly.

    Returns:
        JSON with status, model loaded flag, and timestamp.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'feature_count': len(FEATURE_NAMES),
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }), 200


@app.route('/analyze/wallet', methods=['POST'])
def analyze_wallet():
    """
    POST /analyze/wallet
    Main fraud detection endpoint.

    Request body:
        {
            "address": "0x1234...abcd"
        }

    Response:
        {
            "address": "0x1234...abcd",
            "risk_score": 0.87,
            "risk_percentage": 87,
            "verdict": "HIGH RISK",
            "color": "red",
            "top_shap_signals": [...],
            "transaction_count": 45,
            "timestamp": "2024-01-01T00:00:00"
        }
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()
    address = data.get('address', '').strip()

    if not address:
        return jsonify({'error': 'address field is required'}), 400

    if not validate_ethereum_address(address):
        return jsonify({
            'error': f'Invalid Ethereum address: {address}. '
                     'Must start with 0x and be 42 characters.'
        }), 400

    if MODEL is None:
        return jsonify({
            'error': 'Model not loaded. Please check server logs.'
        }), 500

    try:
        # ── Step 1: Fetch transactions from Etherscan ─────────────────────────
        print(f"\nAnalyzing wallet: {address}")
        transactions = fetch_transactions_from_etherscan(address)
        print(f"  Fetched {len(transactions)} transactions")

        # ── Step 2: Extract features ──────────────────────────────────────────
        features_df = extract_features_from_transactions(
            transactions, address
        )

        # Ensure columns match training order exactly
        features_df = features_df.reindex(columns=FEATURE_NAMES, fill_value=0.0)

        # ── Step 3: Scale features ────────────────────────────────────────────
        features_scaled = SCALER.transform(features_df)

        # ── Step 4: Get prediction ────────────────────────────────────────────
        risk_score = float(MODEL.predict_proba(features_scaled)[0][1])
        verdict_info = get_risk_verdict(risk_score)

        # ── Step 5: Get top SHAP signals ──────────────────────────────────────
        import shap
        explainer = shap.TreeExplainer(MODEL)
        shap_values = explainer.shap_values(features_scaled)

        # Get top 5 features driving this prediction
        feature_shap = list(zip(FEATURE_NAMES, shap_values[0]))
        feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)
        top_shap_signals = [
            {
                'feature': feat,
                'shap_value': round(float(val), 4),
                'feature_value': round(float(features_df[feat].iloc[0]), 4),
                'direction': 'increases fraud risk' if val > 0 else 'decreases fraud risk'
            }
            for feat, val in feature_shap[:5]
        ]

        # ── Step 6: Save to recent scans ──────────────────────────────────────
        save_recent_scan(address, risk_score, verdict_info['verdict'])

        # ── Step 7: Build response ────────────────────────────────────────────
        response = {
            'address': address,
            'risk_score': round(risk_score, 4),
            'risk_percentage': round(risk_score * 100, 1),
            'verdict': verdict_info['verdict'],
            'color': verdict_info['color'],
            'emoji': verdict_info['emoji'],
            'description': verdict_info['description'],
            'top_shap_signals': top_shap_signals,
            'transaction_count': len(transactions),
            'timestamp': datetime.now(timezone.utc).isoformat()  
        }

        print(f"  Risk score: {risk_score:.4f} → {verdict_info['verdict']}")
        return jsonify(response), 200

    except Exception as e:
        print(f"  ERROR: {e}")
        return jsonify({
            'error': str(e),
            'address': address
        }), 500


@app.route('/explain', methods=['POST'])
def explain_prediction():
    """
    POST /explain
    Calls Groq + LLaMA 3 70B to generate a plain English explanation
    of why a wallet was flagged as fraudulent.

    Request body:
        {
            "address": "0x1234...abcd",
            "risk_score": 0.87,
            "verdict": "HIGH RISK",
            "top_shap_signals": [...],
            "transaction_count": 45
        }

    Response:
        {
            "explanation": "This wallet shows several red flags..."
        }
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()
    required_fields = ['address', 'risk_score', 'verdict', 'top_shap_signals']

    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400

    if groq_client is None:
        return jsonify({
            'explanation': 'LLM explanation unavailable — Groq API not configured.'
        }), 200

    try:
        # ── Build prompt for LLaMA 3 ──────────────────────────────────────────
        shap_summary = "\n".join([
            f"  - {s['feature']}: {s['direction']} "
            f"(value: {s['feature_value']}, SHAP: {s['shap_value']})"
            for s in data['top_shap_signals']
        ])

        prompt = f"""You are a blockchain security analyst explaining fraud detection results to a non-technical user.

Wallet Analysis Results:
- Address: {data['address']}
- Fraud Risk Score: {data['risk_score']:.1%}
- Verdict: {data['verdict']}
- Transactions Analyzed: {data.get('transaction_count', 'unknown')}

Top risk signals detected by our ML model:
{shap_summary}

Please explain in 3-4 clear sentences:
1. What this verdict means for the user
2. Which specific behaviors made this wallet suspicious
3. What the user should do (e.g. avoid transacting, proceed with caution)

Use simple language. Do not use technical jargon. Be direct and helpful."""

        # ── Call Groq API ─────────────────────────────────────────────────────
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful blockchain security analyst. "
                        "Explain fraud detection results clearly and concisely "
                        "to non-technical users. Always be accurate and helpful."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=config.GROQ_MODEL,
            max_tokens=300,
            temperature=0.3,   # Low temperature = more consistent, factual responses
        )

        explanation = chat_completion.choices[0].message.content.strip()
        print(f"  LLM explanation generated ({len(explanation)} chars)")

        return jsonify({'explanation': explanation}), 200

    except Exception as e:
        print(f"  Groq API error: {e}")
        return jsonify({
            'explanation': f'Explanation unavailable: {str(e)}'
        }), 200


@app.route('/recent-scans', methods=['GET'])
def get_recent_scans():
    """
    GET /recent-scans
    Returns the last 10 wallet scans globally.
    Used by the Streamlit dashboard feed.

    Returns:
        JSON list of recent scan results.
    """
    scans = load_recent_scans()
    return jsonify({'scans': scans}), 200


# ── Run the app ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*60)
    print("On-Chain Fraud Intelligence Network — Flask API")
    print("="*60)
    print(f"Starting server on http://localhost:{config.FLASK_PORT}")
    print("Endpoints:")
    print("  GET  /health")
    print("  POST /analyze/wallet")
    print("  POST /explain")
    print("  GET  /recent-scans")
    print("="*60 + "\n")

    app.run(
    host='0.0.0.0',
    port=config.FLASK_PORT,
    debug=False,
    use_reloader=False
)