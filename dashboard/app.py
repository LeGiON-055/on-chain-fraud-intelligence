"""
dashboard/app.py
----------------
Streamlit dashboard for the On-Chain Fraud Intelligence Network.
Provides a visual interface for wallet fraud analysis with:
    - Risk score gauge
    - SHAP waterfall chart
    - Transaction history table
    - LLM explanation box
    - Recent flagged wallets feed

What you'll learn from this file:
    - Streamlit components and layout
    - st.session_state for managing app state
    - Plotly charts in Streamlit
    - Calling your own Flask API from Python
    - Deploying a Python app to the cloud
"""

import json
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pathlib import Path
from datetime import datetime

# ── Page config — must be first Streamlit command ─────────────────────────────
st.set_page_config(
    page_title="On-Chain Fraud Intelligence",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────────────────────
import os
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:5000")
RECENT_SCANS_FILE = Path("data/recent_scans.json")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .risk-high {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffaa00, #cc8800);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .risk-safe {
        background: linear-gradient(135deg, #00cc66, #009944);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #2d3250;
        text-align: center;
    }
    .shap-positive { color: #ff6b6b; font-weight: bold; }
    .shap-negative { color: #51cf66; font-weight: bold; }
    .explanation-box {
        background: #1e2130;
        border-left: 4px solid #4dabf7;
        padding: 16px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .scan-item {
        background: #1e2130;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 4px solid #444;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def call_analyze_api(address: str) -> dict:
    """
    Call the Flask /analyze/wallet endpoint.

    Args:
        address: Ethereum wallet address to analyze.

    Returns:
        API response dictionary or error dict.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze/wallet",
            json={"address": address},
            timeout=30
        )
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "error": "Cannot connect to Flask API. Make sure api/app.py is running."
        }
    except Exception as e:
        return {"error": str(e)}


def call_explain_api(analysis_result: dict) -> str:
    """
    Call the Flask /explain endpoint to get LLM explanation.

    Args:
        analysis_result: Result from /analyze/wallet endpoint.

    Returns:
        Plain English explanation string.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/explain",
            json=analysis_result,
            timeout=30
        )
        data = response.json()
        return data.get('explanation', 'Explanation unavailable.')
    except Exception as e:
        return f"Could not generate explanation: {str(e)}"


def load_recent_scans() -> list:
    """Load recent scans from JSON file."""
    if RECENT_SCANS_FILE.exists():
        try:
            with open(RECENT_SCANS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def create_risk_gauge(risk_score: float, verdict: str) -> go.Figure:
    """
    Create a Plotly gauge chart showing the fraud risk score.

    Args:
        risk_score: Fraud probability between 0 and 1.
        verdict: Risk verdict string.

    Returns:
        Plotly Figure object.
    """
    # Color based on risk level
    if risk_score >= 0.75:
        color = "#ff4444"
    elif risk_score >= 0.40:
        color = "#ffaa00"
    else:
        color = "#00cc66"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(risk_score * 100, 1),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"Fraud Risk Score<br><span style='font-size:0.8em'>{verdict}</span>",
            'font': {'size': 18, 'color': 'white'}
        },
        number={
            'suffix': "%",
            'font': {'size': 36, 'color': color}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': "white",
                'tickfont': {'color': 'white'}
            },
            'bar': {'color': color},
            'bgcolor': "#1e2130",
            'borderwidth': 2,
            'bordercolor': "#2d3250",
            'steps': [
                {'range': [0, 40], 'color': '#1a3a1a'},
                {'range': [40, 75], 'color': '#3a2a00'},
                {'range': [75, 100], 'color': '#3a0000'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': risk_score * 100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def create_shap_waterfall(shap_signals: list) -> go.Figure:
    """
    Create a SHAP waterfall chart from the top risk signals.

    Args:
        shap_signals: List of SHAP signal dicts from API response.

    Returns:
        Plotly Figure object.
    """
    features = [s['feature'] for s in shap_signals]
    values = [s['shap_value'] for s in shap_signals]
    colors = ['#ff6b6b' if v > 0 else '#51cf66' for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition='outside',
        textfont={'color': 'white', 'size': 12}
    ))

    fig.update_layout(
        title={
            'text': 'SHAP Feature Importance — Why this verdict?',
            'font': {'color': 'white', 'size': 14}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis={
            'title': 'SHAP Value (red = increases fraud risk)',
            'color': 'white',
            'gridcolor': '#2d3250',
            'zerolinecolor': '#4d5270'
        },
        yaxis={
            'color': 'white',
            'gridcolor': '#2d3250'
        },
        height=300,
        margin=dict(l=20, r=80, t=50, b=40)
    )

    return fig


def display_verdict_banner(result: dict) -> None:
    """
    Display a colored verdict banner based on risk level.

    Args:
        result: API response dictionary.
    """
    color = result.get('color', 'green')
    emoji = result.get('emoji', '🟢')
    verdict = result.get('verdict', 'SAFE')
    pct = result.get('risk_percentage', 0)
    desc = result.get('description', '')

    css_class = {
        'red': 'risk-high',
        'amber': 'risk-medium',
        'green': 'risk-safe'
    }.get(color, 'risk-safe')

    st.markdown(f"""
    <div class="{css_class}">
        {emoji} {verdict} — {pct}% Fraud Risk<br>
        <span style="font-size: 14px; font-weight: normal;">{desc}</span>
    </div>
    """, unsafe_allow_html=True)


def display_shap_table(shap_signals: list) -> None:
    """
    Display SHAP signals as a formatted table.

    Args:
        shap_signals: List of SHAP signal dicts.
    """
    rows = []
    for s in shap_signals:
        direction = "🔴 Increases risk" if s['shap_value'] > 0 else "🟢 Decreases risk"
        rows.append({
            'Feature': s['feature'],
            'Value': round(s['feature_value'], 4),
            'SHAP': round(s['shap_value'], 4),
            'Impact': direction
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    """Main Streamlit app function."""

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🔗 On-Chain Fraud Intelligence Network")
    st.markdown(
        "Real-time Ethereum wallet fraud detection using "
        "XGBoost + SHAP + LLaMA 3 70B"
    )
    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔍 Analyze Wallet")

        address_input = st.text_input(
            "Ethereum Wallet Address",
            placeholder="0x1234...abcd",
            help="Enter any Ethereum wallet address to check for fraud risk"
        )

        analyze_btn = st.button(
            "🚀 Analyze Wallet",
            type="primary",
            use_container_width=True
        )

        st.divider()

        # Quick test addresses
        st.subheader("🧪 Quick Test Addresses")
        st.caption("Click to copy and paste above")

        test_addresses = [
            ("Known wallet", "0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe"),
            ("Vitalik's wallet", "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"),
        ]

        for label, addr in test_addresses:
            st.code(addr, language=None)
            st.caption(label)

        st.divider()

        # API Status
        st.subheader("⚙️ API Status")
        try:
            health = requests.get(f"{API_BASE_URL}/health", timeout=3)
            if health.status_code == 200:
                st.success("Flask API: Online ✓")
            else:
                st.error("Flask API: Error")
        except Exception:
            st.error("Flask API: Offline ✗")
            st.caption("Run: python api/app.py")

    # ── Main content ──────────────────────────────────────────────────────────

    # Initialize session state
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'explanation' not in st.session_state:
        st.session_state.explanation = None

    # Run analysis when button clicked
    if analyze_btn and address_input:
        with st.spinner(f"Analyzing wallet {address_input[:10]}..."):
            result = call_analyze_api(address_input)

            if 'error' in result:
                st.error(f"Error: {result['error']}")
            else:
                st.session_state.analysis_result = result

                # Get LLM explanation
                with st.spinner("Generating AI explanation..."):
                    explanation = call_explain_api(result)
                    st.session_state.explanation = explanation

    elif analyze_btn and not address_input:
        st.warning("Please enter an Ethereum wallet address.")

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result

        # Verdict banner
        st.subheader("📊 Analysis Result")
        display_verdict_banner(result)
        st.markdown("")

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Risk Score", f"{result['risk_percentage']}%")
        with col2:
            st.metric("Transactions Analyzed", result['transaction_count'])
        with col3:
            st.metric("Verdict", result['verdict'])
        with col4:
            ts = result['timestamp'][:10]
            st.metric("Scan Date", ts)

        st.divider()

        # Charts row
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("🎯 Risk Gauge")
            gauge_fig = create_risk_gauge(
                result['risk_score'],
                result['verdict']
            )
            st.plotly_chart(
                gauge_fig,
                use_container_width=True,
                key="gauge"
            )

        with col_right:
            st.subheader("📈 SHAP Feature Impact")
            shap_fig = create_shap_waterfall(result['top_shap_signals'])
            st.plotly_chart(
                shap_fig,
                use_container_width=True,
                key="shap"
            )

        st.divider()

        # SHAP details table
        st.subheader("🔬 Feature Analysis Details")
        display_shap_table(result['top_shap_signals'])

        st.divider()

        # LLM Explanation
        if st.session_state.explanation:
            st.subheader("🤖 AI Explanation (LLaMA 3 70B via Groq)")
            st.markdown(f"""
            <div class="explanation-box">
                {st.session_state.explanation}
            </div>
            """, unsafe_allow_html=True)

            # Regenerate button
            if st.button("🔄 Regenerate Explanation"):
                with st.spinner("Regenerating..."):
                    explanation = call_explain_api(result)
                    st.session_state.explanation = explanation
                    st.rerun()

        st.divider()

        # Wallet address display
        st.subheader("🔗 Wallet Details")
        st.code(result['address'])
        etherscan_url = f"https://etherscan.io/address/{result['address']}"
        st.markdown(f"[View on Etherscan ↗]({etherscan_url})")

    else:
        # Landing state — show instructions
        st.info(
            "👈 Enter an Ethereum wallet address in the sidebar and click "
            "**Analyze Wallet** to get started."
        )

        # Feature highlights
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            ### 🧠 ML Powered
            XGBoost model trained on 9,000+ wallets
            with ROC-AUC of 0.98
            """)
        with col2:
            st.markdown("""
            ### 💡 Explainable AI
            SHAP values show exactly which
            transaction patterns triggered the alert
            """)
        with col3:
            st.markdown("""
            ### 🤖 LLM Explanations
            LLaMA 3 70B converts technical signals
            into plain English for anyone to understand
            """)

    # ── Recent Scans Feed ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("🕐 Recent Wallet Scans")

    recent_scans = load_recent_scans()

    if recent_scans:
        for scan in recent_scans[:5]:
            color_map = {
                'HIGH RISK': '#ff4444',
                'SUSPICIOUS': '#ffaa00',
                'SAFE': '#00cc66'
            }
            border_color = color_map.get(scan['verdict'], '#444')
            short_addr = scan['address'][:10] + '...' + scan['address'][-6:]
            ts = scan['timestamp'][:16].replace('T', ' ')

            st.markdown(f"""
            <div class="scan-item" style="border-left-color: {border_color}">
                <strong>{short_addr}</strong> —
                <span style="color: {border_color}">{scan['verdict']}</span>
                ({round(scan['risk_score']*100, 1)}% risk) &nbsp;·&nbsp;
                <span style="color: #888; font-size: 12px;">{ts} UTC</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No scans yet — analyze a wallet to see it appear here.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()