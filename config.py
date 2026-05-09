"""
config.py
---------
Central configuration for the On-Chain Fraud Intelligence Network.
Loads all settings from environment variables via python-dotenv.
Never hardcode API keys — always use .env file.
"""

import os
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()

# ── Etherscan ─────────────────────────────────────────────────────────────────
ETHERSCAN_API_KEY: str = os.getenv("ETHERSCAN_API_KEY", "")
ETHERSCAN_BASE_URL: str = "https://api.etherscan.io/api"

# ── Groq / LLaMA ──────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama3-70b-8192"

# ── Alchemy ───────────────────────────────────────────────────────────────────
ALCHEMY_API_KEY: str = os.getenv("ALCHEMY_API_KEY", "")

# ── Flask ─────────────────────────────────────────────────────────────────────
FLASK_ENV: str = os.getenv("FLASK_ENV", "development")
FLASK_SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
FLASK_PORT: int = 5000

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH: str = "models/fraud_detector.joblib"
FEATURES_PATH: str = "models/feature_names.joblib"

# ── Risk thresholds ───────────────────────────────────────────────────────────
RISK_HIGH: float = 0.75      # Above this → HIGH RISK (red)
RISK_MEDIUM: float = 0.40    # Above this → SUSPICIOUS (amber)
                              # Below 0.40 → SAFE (green)