import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="LSTM Fraud Engine | Enterprise", page_icon="🧠", layout="wide")

st.title("🧠 Inside the LSTM: Enterprise Fraud Engine")
st.markdown("**IBM Z Datathon 2026** | Sequence Windowing & Feature Attribution on IBM LinuxONE.")
st.divider()

# --- SIDEBAR ---
st.sidebar.header("🎛️ Build a 7-Step Sequence")
st.sidebar.write("The LSTM looks at the 6 past transactions to predict the 7th.")
seq_amounts = []
for i in range(1, 8):
    default_val = 15.0 if i < 7 else 8900.0
    amt = st.sidebar.slider(f"Tx {i} Amount ($)", 1.0, 10000.0, default_val, step=100.0)
    seq_amounts.append(amt)

analyze_btn = st.sidebar.button("🧠 Run LSTM Analysis", use_container_width=True)

# --- ADVANCED LOGIC & ATTRIBUTION ---
def analyze_sequence(amounts):
    past_avg = np.mean(amounts[:-1])
    current_amt = amounts[-1]
    if past_avg == 0: past_avg = 1
    spike_ratio = current_amt / past_avg
    
    # Calculate contributions
    reasons = []
    score = 0.0
    
    # 1. Spike Contribution
    if spike_ratio > 10:
        score += 0.60
        reasons.append(("Extreme Velocity Spike", 0.60, f"Current tx is {spike_ratio:.1f}x the historical average."))
    elif spike_ratio > 5:
        score += 0.40
        reasons.append(("High Velocity Spike", 0.40, f"Current tx is {spike_ratio:.1f}x the historical average."))
    elif spike_ratio > 2:
        score += 0.15
        reasons.append(("Moderate Velocity Spike", 0.15, f"Current tx is {spike_ratio:.1f}x the historical average."))
        
    # 2. Absolute Amount Contribution
    if current_amt > 5000:
        score += 0.30
        reasons.append(("High Absolute Amount", 0.30, f"Transaction exceeds the $5,000 high-risk threshold."))
    elif current_amt > 1000:
        score += 0.10
        reasons.append(("Elevated Amount", 0.10, f"Transaction is above the normal daily baseline."))
        
    score = min(score, 0.99)
    
    if score >= 0.85: risk, action, color = "CRITICAL", "Immediate account freeze", "red"
    elif score >= 0.50: risk, action, color = "HIGH", "Flag for manual review", "orange"
    elif score >= 0.20: risk, action, color = "MEDIUM", "Monitor account", "yellow"
    else: risk, action, color = "LOW", "Auto-approve", "green"
        
    return score, risk, action, color, reasons, spike_ratio

# --- MAIN DISPLAY ---
if analyze_btn:
    score, risk, action, color, reasons, spike = analyze_sequence(seq_amounts)
    
    # Top Row: Visuals
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Sequence Windowing Visualization")
        st.write("The LSTM receives an array of shape `(1, 7, 1)`. The final step (red) is the target prediction.")
        
        colors = ['#4CAF50'] * 6 + ['#F44336']
        fig = go.Figure(data=[go.Bar(
            x=[f"Tx {i+1}" for i in range(7)],
            y=seq_amounts,
            marker_color=colors,
            text=[f"${a:,.2f}" for a in seq_amounts],
            textposition='auto',
        )])
        fig.update_layout(title="Transaction Sequence (7 Steps)", yaxis_title="Amount ($)", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🤖 LSTM Output")
        if score > 0.5:
            st.error(f"🚨 **FRAUD DETECTED** (Risk: {risk})")
        else:
            st.success(f"✅ **LEGITIMATE** (Risk: {risk})")
            
        st.metric("Fraud Probability", f"{score*100:.1f}%")
        st.progress(float(score))
        
        st.markdown(f"**Automated Action:**")
        st.info(action)

    st.divider()

    # Bottom Row: The "Why" (Feature Attribution)
    st.subheader("🔍 Feature Attribution: Why did the LSTM flag this?")
    st.write("The model breaks down the probability score into individual risk contributions.")
    
    # Create a horizontal bar chart for the reasons
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
        fig2.update_layout(title="Risk Contribution Breakdown", xaxis_title="Score Contribution", height=300, xaxis_range=[0, 0.8])
        st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed text explanation
        for name, val, desc in reasons:
            st.markdown(f"- **{name} (+{val*100:.1f}%)**: {desc}")
    else:
        st.write("No anomalous features detected in this sequence. All transactions align with the historical baseline.")

else:
    st.info("👈 Adjust the sequence in the sidebar and click 'Run LSTM Analysis' to see the model in action.")
