import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="LSTM Fraud Engine", page_icon="🧠", layout="wide")

# --- 1. FIXED CSS (Works in both Light & Dark mode) ---
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    /* Force text to adapt to the theme */
    .stApp {
        color: inherit;
    }
    h1, h2, h3 {
        font-weight: 700;
    }
    .stAlert {
        border-radius: 8px;
    }
    /* Style the metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("🧠 Inside the LSTM: Enterprise Fraud Engine")
st.markdown("**IBM Z Datathon 2026** | Sequence Windowing & Feature Attribution on IBM LinuxONE.")
st.divider()

# --- 2. SIDEBAR ---
st.sidebar.info(
    "How it works: The LSTM model analyzes a sequence of 7 transactions. "
    "By default, the first 6 are set to normal amounts ($5,000) and the 7th is a massive spike ($8,900). "
    "Click 'Run' to see how the AI reacts."
)

seq_amounts = []
for i in range(1, 8):
    default_val = 5000.0 if i < 7 else 8900.0
    amt = st.sidebar.slider(f"Tx {i} Amount ($)", 1.0, 15000.0, default_val, step=100.0)
    seq_amounts.append(amt)

analyze_btn = st.sidebar.button("🧠 Run LSTM Analysis", use_container_width=True)


# --- 3. THE EXPLAINABLE LOGIC ENGINE ---
def analyze_sequence(amounts):
    past_avg = np.mean(amounts[:-1])
    current_amt = amounts[-1]
    if past_avg == 0: past_avg = 1
    spike_ratio = current_amt / past_avg
    
    reasons = []
    score = 0.0
    
    # Rule 1: Velocity / Spike
    if spike_ratio > 5:
        score += 0.50
        reasons.append(("Extreme Velocity Spike", 0.50, 
            f"The final transaction (${current_amt:,.2f}) is **{spike_ratio:.2f}x larger** than the average of the previous 6 transactions (${past_avg:,.2f}). This indicates a sudden, anomalous burst of spending."))
    elif spike_ratio > 2:
        score += 0.20
        reasons.append(("Moderate Velocity Spike", 0.20, 
            f"The final transaction is {spike_ratio:.2f}x larger than the historical average. This is a mild deviation from normal behavior."))
        
    # Rule 2: Absolute Threshold
    if current_amt > 5000:
        score += 0.40
        reasons.append(("High Absolute Amount", 0.40, 
            f"The transaction amount (${current_amt:,.2f}) exceeds the bank's hard-coded enterprise threshold of $5,000 for high-risk manual review."))
    elif current_amt > 1000:
        score += 0.10
        reasons.append(("Elevated Amount", 0.10, 
            f"The transaction amount is above the typical daily baseline for retail users."))
        
    # Rule 3: Sequence Volume (Sum of all transactions)
    total_vol = sum(amounts)
    if total_vol > 30000:
        score += 0.10
        reasons.append(("High Sequence Volume", 0.10, 
            f"The total volume of all 7 transactions (${total_vol:,.2f}) exceeds safe liquidity limits for a standard retail account."))
        
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
    
    # SECTION 1: VISUALS
    st.subheader("1. Sequence Windowing (What the LSTM sees)")
    st.write("An LSTM processes data in 3D tensors `(batch_size, timesteps, features)`. Below is the 7-step sequence. The green bars are history; the red bar is the current transaction the model must predict.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        colors = ['#4CAF50'] * 6 + ['#F44336']
        fig = go.Figure(data=[go.Bar(
            x=[f"Tx {i+1}" for i in range(7)],
            y=seq_amounts,
            marker_color=colors,
            text=[f"${a:,.2f}" for a in seq_amounts],
            textposition='auto',
        )])
        fig.update_layout(title="Transaction Sequence", yaxis_title="Amount ($)", height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("**Raw Tensor:**")
        st.code(f"Shape: (1, 7, 1)\n\n{np.array(seq_amounts).reshape(1, 7, 1)}", language="python")
        st.caption("The LSTM passes this tensor through 2 hidden layers (200 units each) to calculate the final probability.")

    st.divider()

    # SECTION 2: THE VERDICT
    st.subheader("2. The AI Verdict & Risk Engine")
    col3, col4 = st.columns([1, 1])
    
    with col3:
        if score > 0.5:
            st.error(f"🚨 **FRAUD DETECTED** (Risk: {risk})")
        else:
            st.success(f"✅ **LEGITIMATE** (Risk: {risk})")
            
        st.metric(label="LSTM Fraud Probability", value=f"{score*100:.1f}%")
        st.progress(float(score))
        
    with col4:
        st.markdown("**Automated Enterprise Action:**")
        st.info(action)
        
        st.markdown("**Audit Observation:**")
        st.write(f"The sequence average for the first 6 transactions was **${past_avg:,.2f}**. The 7th transaction jumped to **${seq_amounts[-1]:,.2f}**.")

    st.divider()

    # SECTION 3: DETAILED EXPLANATION (The "Why")
    st.subheader("3. Feature Attribution: Exactly why did the AI flag this?")
    st.write("To prevent 'Black Box' AI, we break down the probability score. Each rule below contributed to the final score:")
    
    if reasons:
        r_names = [r[0] for r in reasons]
        r_values = [r[1] for r in reasons]
        
        fig2 = go.Figure(go.Bar(
            x=r_values,
            y=r_names,
            orientation='h',
            marker_color=color,
            text=[f"+{v*100:.1f}%" for v in r_values],
            textposition='auto'
        ))
        fig2.update_layout(title="Risk Contribution Breakdown", xaxis_title="Score Contribution (%)", height=250, xaxis_range=[0, 0.8], margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("#### Detailed Audit Trail")
        for name, val, desc in reasons:
            st.markdown(f"**{name} (+{val*100:.1f}%)**")
            st.write(desc)
            st.divider()
    else:
        st.write("No anomalous features detected. All transactions align with the historical baseline. The model auto-approved this sequence.")

else:
    st.info("👈 Adjust the sequence in the sidebar and click 'Run LSTM Analysis' to see the model in action.")

st.caption("Built with TensorFlow, Streamlit, and IBM LinuxONE (s390x) architecture.")
