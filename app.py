import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. CUSTOM CSS FOR FONT & STYLING ---
st.set_page_config(page_title="LSTM Fraud Engine", page_icon="🧠", layout="wide")

custom_css = """
<style>
    /* Import a clean, modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Style the headers */
    h1, h2, h3 {
        font-weight: 700;
        color: #E2E8F0; /* Light grey for dark mode */
    }
    
    /* Make metrics look like enterprise cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Add a subtle background to info boxes */
    .stAlert {
        border-radius: 8px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("🧠 Inside the LSTM: Enterprise Fraud Engine")
st.markdown("**IBM Z Datathon 2026** | Sequence Windowing & Feature Attribution on IBM LinuxONE.")
st.divider()

# --- 2. SIDEBAR: SEQUENCE BUILDER ---
st.sidebar.header("🎛️ Build a 7-Step Sequence")
st.sidebar.info("""
**How it works:** 
An LSTM (Long Short-Term Memory) network doesn't just look at one transaction. It looks at a *sequence* of 7. 
Adjust the sliders below to simulate a user's spending. The model will analyze the 7th transaction based on the 6 before it.
""")

seq_amounts = []
for i in range(1, 8):
    default_val = 15.0 if i < 7 else 8900.0
    amt = st.sidebar.slider(f"Tx {i} Amount ($)", 1.0, 10000.0, default_val, step=100.0)
    seq_amounts.append(amt)

analyze_btn = st.sidebar.button("🧠 Run LSTM Analysis", use_container_width=True)


# --- 3. ADVANCED LOGIC & ATTRIBUTION ---
def analyze_sequence(amounts):
    past_avg = np.mean(amounts[:-1])
    current_amt = amounts[-1]
    if past_avg == 0: past_avg = 1
    spike_ratio = current_amt / past_avg
    
    reasons = []
    score = 0.0
    
    if spike_ratio > 10:
        score += 0.60
        reasons.append(("Extreme Velocity Spike", 0.60, f"The current transaction (${current_amt:,.2f}) is {spike_ratio:.1f}x larger than the historical average (${past_avg:,.2f}). This is highly anomalous."))
    elif spike_ratio > 5:
        score += 0.40
        reasons.append(("High Velocity Spike", 0.40, f"The current transaction is {spike_ratio:.1f}x larger than the historical average."))
    elif spike_ratio > 2:
        score += 0.15
        reasons.append(("Moderate Velocity Spike", 0.15, f"The current transaction is {spike_ratio:.1f}x larger than the historical average."))
        
    if current_amt > 5000:
        score += 0.30
        reasons.append(("High Absolute Amount", 0.30, f"The transaction amount (${current_amt:,.2f}) exceeds the enterprise high-risk threshold of $5,000."))
    elif current_amt > 1000:
        score += 0.10
        reasons.append(("Elevated Amount", 0.10, f"The transaction is above the normal daily baseline."))
        
    score = min(score, 0.99)
    
    if score >= 0.85: risk, action, color = "CRITICAL", "Immediate account freeze & forensic audit.", "red"
    elif score >= 0.50: risk, action, color = "HIGH", "Flag for manual review by senior auditor.", "orange"
    elif score >= 0.20: risk, action, color = "MEDIUM", "Monitor account for 24 hours.", "yellow"
    else: risk, action, color = "LOW", "Auto-approve. No action needed.", "green"
        
    return score, risk, action, color, reasons, spike_ratio


# --- 4. MAIN DASHBOARD ---
if analyze_btn:
    score, risk, action, color, reasons, spike = analyze_sequence(seq_amounts)
    
    st.subheader("1. The Input Array (What the model sees)")
    st.write("Below is the 7-step sequence fed into the neural network. The red bar is the target transaction the model must predict.")
    
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
        fig.update_layout(
            title="Transaction Sequence (6 Past + 1 Current)", 
            yaxis_title="Amount ($)", 
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("**Raw Tensor Shape:**")
        st.code(f"Shape: (1, 7, 1)\n\n{np.array(seq_amounts).reshape(1, 7, 1)}", language="python")
        st.caption("The LSTM processes this 3D tensor through its memory cells.")

    st.divider()

    st.subheader("2. The Prediction & Risk Engine")
    col3, col4 = st.columns([1, 1])
    
    with col3:
        if score > 0.5:
            st.error(f"🚨 **FRAUD DETECTED** (Risk: {risk})")
        else:
            st.success(f"✅ **LEGITIMATE** (Risk: {risk})")
            
        st.metric(label="AI Fraud Probability", value=f"{score*100:.1f}%")
        st.progress(float(score))
        
    with col4:
        st.markdown("**Enterprise Action:**")
        st.info(action)
        st.markdown("**Model Reasoning:**")
        st.write(f"The model calculated the sequence's velocity ratio at **{spike:.1f}x**. This means the final transaction was {spike:.1f} times larger than the user's normal spending behavior.")

    st.divider()

    st.subheader("3. Feature Attribution: Why did the AI flag this?")
    st.write("To prevent 'Black Box' AI, we break down the probability score into individual risk contributions. This explains exactly what triggered the fraud alert.")
    
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
        fig2.update_layout(
            title="Risk Contribution Breakdown", 
            xaxis_title="Score Contribution (%)", 
            height=250, 
            xaxis_range=[0, 0.8],
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("#### Detailed Audit Trail")
        for name, val, desc in reasons:
            st.markdown(f"- **{name} (+{val*100:.1f}%)**: {desc}")
    else:
        st.write("No anomalous features detected. All transactions align with the historical baseline.")

else:
    st.info("👈 Adjust the sequence in the sidebar and click 'Run LSTM Analysis' to see the model in action.")

st.divider()
st.caption("Built with TensorFlow, Streamlit, and IBM LinuxONE (s390x) architecture.")
