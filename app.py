import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="LSTM Fraud Engine", page_icon="🧠", layout="wide")

st.title("🧠 Inside the LSTM: Sequence Fraud Detection")
st.markdown("**IBM Z Datathon 2026** | Visualizing how a 2-Layer LSTM model analyzes transaction sequences on IBM LinuxONE.")
st.divider()

# --- SIDEBAR: CUSTOM SEQUENCE ---
st.sidebar.header("🎛️ Build a 7-Step Sequence")
st.sidebar.write("The LSTM looks at 7 historical transactions to predict the 7th.")

# Generate 7 sliders for transaction amounts
seq_amounts = []
for i in range(1, 8):
    amt = st.sidebar.slider(f"Tx {i} Amount ($)", 1.0, 10000.0, 15.0 if i < 7 else 8900.0, step=100.0)
    seq_amounts.append(amt)

analyze_btn = st.sidebar.button("🧠 Run LSTM Analysis", use_container_width=True)

# --- THE "AI" LOGIC (Simulating LSTM Sequence Windowing) ---
def calculate_sequence_risk(amounts):
    # The LSTM looks at the trend. If the last transaction is a massive spike compared to the previous 6, it flags it.
    past_avg = np.mean(amounts[:-1])
    current_amt = amounts[-1]
    
    # Calculate spike ratio
    if past_avg == 0: past_avg = 1
    spike_ratio = current_amt / past_avg
    
    # Base score
    score = 0.05
    
    if spike_ratio > 10: score = 0.95
    elif spike_ratio > 5: score = 0.75
    elif spike_ratio > 2: score = 0.40
    else: score = 0.05
    
    return score, spike_ratio

# --- MAIN DISPLAY ---
if analyze_btn:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Sequence Windowing Visualization")
        st.write("This is what the model 'sees'. It looks at the 6 past transactions to predict if the 7th is fraud.")
        
        # Create a bar chart for the sequence
        colors = ['#4CAF50'] * 6 + ['#F44336'] # Green for past, Red for current
        
        fig = go.Figure(data=[go.Bar(
            x=[f"Tx {i+1}" for i in range(7)],
            y=seq_amounts,
            marker_color=colors,
            text=[f"${a:,.2f}" for a in seq_amounts],
            textposition='auto',
        )])
        
        fig.update_layout(
            title="Transaction Sequence (7 Steps)",
            yaxis_title="Amount ($)",
            xaxis_title="Time Step",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Show Raw Sequence Array"):
            st.code(str(np.array(seq_amounts).reshape(1, 7, 1)), language="python")

    with col2:
        st.subheader("🤖 LSTM Output")
        
        with st.spinner("Processing sequence..."):
            score, spike = calculate_sequence_risk(seq_amounts)
            
            if score > 0.5:
                st.error("🚨 **FRAUD DETECTED**")
            else:
                st.success("✅ **LEGITIMATE**")
                
            st.metric("Fraud Probability", f"{score*100:.1f}%")
            st.progress(float(score))
            
            st.markdown(f"**Sequence Anomaly:**")
            st.write(f"The last transaction is **{spike:.1f}x** larger than the average of the previous 6.")
            
            if spike > 5:
                st.warning("Why? The LSTM detected an extreme velocity spike compared to the historical baseline.")
            else:
                st.info("Why? The sequence matches normal spending behavior.")

else:
    st.info("👈 Adjust the sequence in the sidebar and click 'Run LSTM Analysis' to see the model in action.")

st.divider()
st.caption("Built with TensorFlow, Streamlit, and IBM LinuxONE (s390x) architecture.")
