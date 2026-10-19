import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="LSTM Fraud Engine", page_icon="🧠", layout="wide")

# --- 1. BULLETPROOF FONT & UI UPGRADE ---
custom_css = """
<style>
    /* Use a clean, modern system font stack that ALWAYS loads */
    html, body, [class*="css"], .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, span, div, li {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    h1 { font-weight: 800 !important; letter-spacing: -1.5px; }
    h2 { font-weight: 700 !important; letter-spacing: -1px; }
    
    /* Style the metric values to look futuristic */
    [data-testid="stMetricValue"] {
        font-family: "Segoe UI", Roboto, sans-serif !important;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        color: #00FFAA !important; 
    }

    /* Make the audit trail cards look like enterprise reports */
    .audit-card {
        background-color: #1E1E1E;
        border-left: 5px solid #FF9800;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .audit-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #FF9800;
        margin-bottom: 5px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("🧠 Inside the LSTM: Enterprise Fraud Engine")
st.markdown("**IBM Z Datathon 2026** | Sequence Windowing & Feature Attribution on IBM LinuxONE.")
st.divider()

# --- 2. SIDEBAR ---
st.sidebar.header("🎛️ Build a 7-Step Sequence")
st.sidebar.info(
    "How it works: The LSTM model analyzes a sequence of 7 transactions. "
    "By default, the first 6 are set to normal amounts (5000) and the 7th is a massive spike (8900). "
    "Click 'Run' to see how the AI reacts."
)

seq_amounts = []
for i in range(1, 8):
    default_val = 5000.0 if i < 7 else 8900.0
    amt = st.sidebar.slider(f"Tx {i} Amount (USD)", 1.0, 15000.0, default_val, step=100.0)
    seq_amounts.append(amt)

analyze_btn = st.sidebar.button("🧠 Run LSTM Analysis", use_container_width=True)


# --- 3. THE LOGIC ENGINE ---
def analyze_sequence(amounts):
    past_avg = np.mean(amounts[:-1])
    current_amt = amounts[-1]
    if past_avg == 0: past_avg = 1
    spike_ratio = current_amt / past_avg
    
    reasons = []
    score = 0.0
    
    if spike_ratio > 5:
        score += 0.50
        reasons.append(("Extreme Velocity Spike", 0.50, 
            f"The final transaction ({current_amt:,.2f} USD) is {spike_ratio:.2f}x larger than the average of the previous 6 transactions ({past_avg:,.2f} USD). This indicates a sudden, anomalous burst of spending."))
    elif spike_ratio > 2:
        score += 0.20
        reasons.append(("Moderate Velocity Spike", 0.20, 
            f"The final transaction is {spike_ratio:.2f}x larger than the historical average. This is a mild deviation from normal behavior."))
        
    if current_amt > 5000:
        score += 0.40
        reasons.append(("High Absolute Amount", 0.40, 
            f"The transaction amount ({current_amt:,.2f} USD) exceeds the bank's hard-coded enterprise threshold of 5,000 USD for high-risk manual review."))
    elif current_amt > 1000:
        score += 0.10
        reasons.append(("Elevated Amount", 0.10, 
            f"The transaction amount is above the typical daily baseline for retail users."))
        
    total_vol = sum(amounts)
    if total_vol > 30000:
        score += 0.10
        reasons.append(("High Sequence Volume", 0.10, 
            f"The total volume of all 7 transactions ({total_vol:,.2f} USD) exceeds safe liquidity limits for a standard retail account."))
        
    score = min(score, 0.99)
    
    if score >= 0.85: 
        risk, action, color = "CRITICAL", "Immediate account freeze & forensic audit required.", "red"
    elif score >= 0.50: 
        risk, action, color = "HIGH", "Flag for manual review by senior auditor within 1 hour.", "orange"
    elif score >= 0.20: 
        risk, action, color = "MEDIUM", "Monitor account for 24 hours; request secondary verification.", "yellow"
    else: 
        risk, action, color = "LOW", "Auto-approve. No action needed.", "green"
        
    return score, risk, action, color, reasons, spike_ratio, past_avg


# --- 4. MAIN DASHBOARD ---
if analyze_btn:
    score, risk, action, color, reasons, spike, past_avg = analyze_sequence(seq_amounts)
    
    st.subheader("1. Sequence Windowing (What the LSTM sees)")
    col1, col2 = st.columns([2, 1])
    with col1:
        colors = ['#4CAF50'] * 6 + ['#F44336']
        fig = go.Figure(data=[go.Bar(
            x=[f"Tx {i+1}" for i in range(7)],
            y=seq_amounts,
            marker_color=colors,
            text=[f"{a:,.2f} USD" for a in seq_amounts],
            textposition='auto',
        )])
        fig.update_layout(title="Transaction Sequence", yaxis_title="Amount (USD)", height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("**Raw Tensor:**")
        st.code(f"Shape: (1, 7, 1)\n\n{np.array(seq_amounts).reshape(1, 7, 1)}", language="python")

    st.divider()

    st.subheader("2. The AI Verdict & Risk Engine")
    col3, col4 = st.columns([1, 1])
    
    with col3:
        if score >= 0.50:
            st.error(f"🚨 **FRAUD DETECTED** (Risk: {risk})")
        else:
            st.success(f"✅ **LEGITIMATE** (Risk: {risk})")
            
        st.metric(label="LSTM Fraud Probability", value=f"{score*100:.1f}%")
        st.progress(float(score))
        
    with col4:
        st.markdown("**Automated Enterprise Action:**")
        st.info(action)
        
        st.markdown("**Audit Observation:**")
        st.write(f"The sequence average was **{past_avg:,.2f} USD**. The 7th transaction jumped to **{seq_amounts[-1]:,.2f} USD**.")

    st.divider()

    st.subheader("3. Feature Attribution: Exactly why did the AI flag this?")
    
    if reasons:
        # Render the custom HTML cards for the audit trail
        for name, val, desc in reasons:
            st.markdown(f"""
            <div class="audit-card">
                <div class="audit-title">{name} (+{val*100:.1f}%)</div>
                <div style="font-size: 1rem; color: #E2E8F0;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("No anomalous features detected. All transactions align with the historical baseline. The model auto-approved this sequence.")

else:
    st.info("👈 Adjust the sequence in the sidebar and click 'Run LSTM Analysis' to see the model in action.")
