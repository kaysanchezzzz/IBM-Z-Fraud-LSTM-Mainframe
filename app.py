import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="LSTM Fraud Engine", page_icon="🧠", layout="wide")

st.title("🧠 Inside the LSTM: Enterprise Fraud Engine")
st.markdown("**IBM Z Datathon 2026** | Sequence Windowing & Feature Attribution on IBM LinuxONE.")
st.divider()

# --- 1. SIDEBAR ---
st.sidebar.header("🎛️ Build a 7-Step Sequence")
st.sidebar.info(
    "How it works: The LSTM model analyzes a sequence of 7 transactions. "
    "By default, the first 6 are set to normal amounts (5,000 USD) and the 7th is a massive spike (8,900 USD). "
    "Click 'Run' to see how the AI calculates the risk."
)

seq_amounts = []
for i in range(1, 8):
    default_val = 5000.0 if i < 7 else 8900.0
    amt = st.sidebar.slider(f"Tx {i} Amount (USD)", 1.0, 15000.0, default_val, step=100.0)
    seq_amounts.append(amt)

analyze_btn = st.sidebar.button("🧠 Run LSTM Analysis", use_container_width=True)

# --- 2. THE DETAILED LOGIC ENGINE ---
def analyze_sequence(amounts):
    past_avg = np.mean(amounts[:-1])
    current_amt = amounts[-1]
    if past_avg == 0: past_avg = 1
    spike_ratio = current_amt / past_avg
    total_vol = sum(amounts)
    
    reasons = []
    score = 0.0
    
    # Rule 1: Velocity / Spike
    if spike_ratio > 5:
        score += 0.50
        reasons.append({
            "title": "Extreme Velocity Spike", 
            "weight": 0.50, 
            "math": f"Current Tx ({current_amt:,.2f}) / Past Avg ({past_avg:,.2f}) = {spike_ratio:.2f}x",
            "explain": f"The final transaction is {spike_ratio:.2f} times larger than the user's historical average. LSTM models are trained to detect sudden bursts of spending, which are highly correlated with stolen card usage."
        })
    elif spike_ratio > 2:
        score += 0.20
        reasons.append({
            "title": "Moderate Velocity Spike", 
            "weight": 0.20, 
            "math": f"Current Tx ({current_amt:,.2f}) / Past Avg ({past_avg:,.2f}) = {spike_ratio:.2f}x",
            "explain": "The transaction is noticeably larger than the historical baseline, triggering a mild anomaly alert."
        })
        
    # Rule 2: Absolute Threshold
    if current_amt > 5000:
        score += 0.40
        reasons.append({
            "title": "High Absolute Amount", 
            "weight": 0.40, 
            "math": f"Current Tx ({current_amt:,.2f}) > Hard Threshold (5,000.00)",
            "explain": "The bank's internal compliance rules dictate that any single transaction above 5,000 USD must be flagged for manual forensic review, regardless of historical behavior."
        })
    elif current_amt > 1000:
        score += 0.10
        reasons.append({
            "title": "Elevated Amount", 
            "weight": 0.10, 
            "math": f"Current Tx ({current_amt:,.2f}) > Baseline (1,000.00)",
            "explain": "The transaction is slightly above the typical retail daily baseline."
        })
        
    # Rule 3: Sequence Volume
    if total_vol > 30000:
        score += 0.10
        reasons.append({
            "title": "High Sequence Volume", 
            "weight": 0.10, 
            "math": f"Total Volume ({total_vol:,.2f}) > Safe Limit (30,000.00)",
            "explain": "The total sum of all 7 transactions in this sequence exceeds the safe liquidity limits for a standard retail account."
        })
        
    score = min(score, 0.99)
    
    if score >= 0.85: 
        risk, action, color = "CRITICAL", "Immediate account freeze & forensic audit required.", "red"
    elif score >= 0.50: 
        risk, action, color = "HIGH", "Flag for manual review by senior auditor within 1 hour.", "orange"
    elif score >= 0.20: 
        risk, action, color = "MEDIUM", "Monitor account for 24 hours; request secondary verification.", "yellow"
    else: 
        risk, action, color = "LOW", "Auto-approve. No action needed.", "green"
        
    return score, risk, action, color, reasons, spike_ratio, past_avg, total_vol

# --- 3. MAIN DASHBOARD ---
if analyze_btn:
    score, risk, action, color, reasons, spike, past_avg, total_vol = analyze_sequence(seq_amounts)
    
    # SECTION 1: DATA WINDOWING
    st.subheader("1. Sequence Windowing (What the LSTM sees)")
    st.write("An LSTM processes data in 3D tensors `(batch_size, timesteps, features)`. Below is the 7-step sequence. The green bars are history; the red bar is the current transaction the model must predict.")
    
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
        st.write("**Raw Tensor Shape:**")
        st.code(f"Shape: (1, 7, 1)\n\n{np.array(seq_amounts).reshape(1, 7, 1)}", language="python")
        st.caption("The LSTM passes this tensor through 2 hidden layers (200 units each) to calculate the final probability.")

    st.divider()

    # SECTION 2: VERDICT
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
        
        st.markdown("**Sequence Summary:**")
        st.write(f"- Historical Average (Tx 1-6): **{past_avg:,.2f} USD**")
        st.write(f"- Current Transaction (Tx 7): **{seq_amounts[-1]:,.2f} USD**")
        st.write(f"- Total Sequence Volume: **{total_vol:,.2f} USD**")

    st.divider()

    # SECTION 3: GRAPH
    st.subheader("3. Feature Attribution: Score Breakdown")
    st.write("To prevent 'Black Box' AI, we break down the probability score into individual risk contributions.")
    
    if reasons:
        r_names = [r["title"] for r in reasons]
        r_values = [r["weight"] for r in reasons]
        
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
        
        # SECTION 4: DETAILED EXPLANATIONS (The "Why")
        st.divider()
        st.subheader("4. Detailed Audit Trail: Why did it flag this?")
        for r in reasons:
            with st.expander(f"🔴 {r['title']} (+{r['weight']*100:.1f}%)", expanded=True):
                st.markdown(f"**The Math:** `{r['math']}`")
                st.markdown(f"**The Explanation:** {r['explain']}")
    else:
        st.write("No anomalous features detected. All transactions align with the historical baseline. The model auto-approved this sequence.")

else:
    st.info("👈 Adjust the sequence in the sidebar and click 'Run LSTM Analysis' to see the model in action.")
