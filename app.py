"""
Fraud Detection Studio - a Streamlit front-end for the LSTM credit-card fraud model.

Run:  streamlit run app.py
Needs the files produced by your notebook (same folder, or set paths in the sidebar):
  shared/fraud_lstm/fitted_mapper.pkl
  shared/fraud_lstm/model.weights.h5
  test_220_40k.csv  and  test_220_40k.indices
"""
import os
import sys
import math

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

TIMESTEPS = 7
FONT = "Inter, 'Segoe UI', system-ui, sans-serif"

# --------------------------------------------------------------------------
# Encoder functions. They MUST exist under these exact names because the
# pickled mapper from the notebook refers to them (notebook defined them in
# __main__). We re-create them here and register them on __main__.
# --------------------------------------------------------------------------
def timeEncoder(X):
    X_hm = X["Time"].str.split(":", expand=True)
    d = pd.to_datetime(dict(year=X["Year"], month=X["Month"], day=X["Day"],
                            hour=X_hm[0], minute=X_hm[1])).astype("int64")
    return pd.DataFrame(d)


def amtEncoder(X):
    amt = X.apply(lambda x: x[1:]).astype(float).map(lambda amt: max(1, amt)).map(math.log)
    return pd.DataFrame(amt)


def decimalEncoder(X, length=5):
    dnew = pd.DataFrame()
    for i in range(length):
        dnew[i] = np.mod(X, 10)
        X = np.floor_divide(X, 10)
    return dnew


def fraudEncoder(X):
    return np.where(X == "Yes", 1, 0).astype(int)


import __main__  # noqa: E402
for _f in (timeEncoder, amtEncoder, decimalEncoder, fraudEncoder):
    setattr(__main__, _f.__name__, _f)

# --------------------------------------------------------------------------
# Page setup + one consistent font everywhere
# --------------------------------------------------------------------------
st.set_page_config(page_title="Fraud Detection Studio", page_icon="🛡️", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, p, li, label, button, input, textarea,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], h1, h2, h3, h4, h5, th, td {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}
h1 { font-weight: 800 !important; letter-spacing: -0.02em; }
h2, h3 { font-weight: 700 !important; letter-spacing: -0.01em; }
.block-container { padding-top: 2rem; max-width: 1200px; }
.why-box {
    border-left: 5px solid #4f46e5; background: rgba(79,70,229,0.07);
    padding: 14px 18px; border-radius: 8px; margin: 10px 0 22px 0; line-height: 1.6;
}
.why-box b.title { color: #4f46e5; display:block; margin-bottom:4px; }
.warn-box {
    border-left: 5px solid #dc2626; background: rgba(220,38,38,0.07);
    padding: 14px 18px; border-radius: 8px; margin: 10px 0 22px 0; line-height: 1.6;
}
.ok-box {
    border-left: 5px solid #16a34a; background: rgba(22,163,74,0.07);
    padding: 14px 18px; border-radius: 8px; margin: 10px 0 22px 0; line-height: 1.6;
}
.hero {
    background: linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%); color:white;
    padding: 28px 32px; border-radius: 14px; margin-bottom: 24px;
}
.hero h1 { color:white; margin:0 0 6px 0; }
.hero p { margin:0; opacity:.92; font-size:1.05rem; }
.small { opacity:.75; font-size:.9rem; }
</style>
""",
    unsafe_allow_html=True,
)


def why(title, text, kind="why"):
    cls = {"why": "why-box", "warn": "warn-box", "ok": "ok-box"}[kind]
    st.markdown(f'<div class="{cls}"><b class="title">{title}</b>{text}</div>', unsafe_allow_html=True)


def style_fig(fig, height=380):
    fig.update_layout(font=dict(family=FONT, size=14), height=height,
                      margin=dict(l=20, r=20, t=50, b=40),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# --------------------------------------------------------------------------
# Sidebar: navigation and file locations
# --------------------------------------------------------------------------
st.sidebar.markdown("## 🛡️ Fraud Detection Studio")
PAGE = st.sidebar.radio(
    "Go to",
    ["🏠 Start here", "🧠 Model & data", "📊 Performance", "🔍 Inspect a transaction",
     "📤 Score your own file", "📖 Glossary"],
)
st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ File locations", expanded=False):
    SAVE_DIR = st.text_input("Model folder", "shared/fraud_lstm")
    TEST_CSV = st.text_input("Test sample CSV", "test_220_40k.csv")
    TEST_IDX = st.text_input("Test sample indices", "test_220_40k.indices")
st.sidebar.caption("Everything runs locally. No data leaves your computer.")


# --------------------------------------------------------------------------
# Loading (cached so the app stays fast)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading preprocessing mapper...")
def load_mapper(save_dir):
    return joblib.load(os.path.join(save_dir, "fitted_mapper.pkl"))


@st.cache_resource(show_spinner="Loading LSTM model...")
def load_model(save_dir, input_size):
    import tensorflow as tf
    m = tf.keras.models.Sequential([
        tf.keras.Input(shape=(TIMESTEPS, input_size)),
        tf.keras.layers.LSTM(200, return_sequences=True),
        tf.keras.layers.LSTM(200, return_sequences=True),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.load_weights(os.path.join(save_dir, "model.weights.h5"))
    return m


@st.cache_data(show_spinner="Reading test sample...")
def load_test(csv_path, idx_path):
    df = pd.read_csv(csv_path, dtype={"Merchant Name": "str"}, index_col="Index")
    idx = np.loadtxt(idx_path).astype(int)
    return df, idx


def assets_ok():
    need = [os.path.join(SAVE_DIR, "fitted_mapper.pkl"), os.path.join(SAVE_DIR, "model.weights.h5")]
    return [p for p in need if not os.path.exists(p)]


def build_windows(df, indices, mapper):
    """Turn target-row indices into (n, 7, features) arrays: 6 past + the current transaction."""
    idx = np.asarray(indices).astype(int)
    arr = np.zeros((len(idx), TIMESTEPS), dtype=int)
    for i in range(TIMESTEPS):
        arr[:, i] = idx + 1 - TIMESTEPS + i
    mapped = mapper.transform(df.loc[arr.flatten()])
    y = mapped["Is Fraud?"].to_numpy().reshape(len(idx), TIMESTEPS)[:, -1]
    X = mapped.drop(columns=["Is Fraud?"]).to_numpy().reshape(len(idx), TIMESTEPS, -1).astype("float32")
    return X, y, arr


def predict(model, X):
    return model.predict(X, batch_size=512, verbose=0)[:, -1, 0]


@st.cache_data(show_spinner="Scoring transactions with the LSTM...")
def score_sample(save_dir, csv_path, idx_path, n):
    mapper = load_mapper(save_dir)
    df, idx = load_test(csv_path, idx_path)
    idx = idx[:n]
    X, y, arr = build_windows(df, idx, mapper)
    model = load_model(save_dir, X.shape[-1])
    return X, y, predict(model, X), idx


def metrics_at(y, s, thr):
    p = (s >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0
    acc = (tp + tn) / max(1, len(y))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1, far=far, acc=acc)


# --------------------------------------------------------------------------
# Plain-language "why" for one transaction window (rule-based facts)
# --------------------------------------------------------------------------
def describe_window(w):
    """w = raw DataFrame of 7 consecutive transactions of the same card. Returns list of (icon, text)."""
    w = w.copy()
    w["_amt"] = w["Amount"].astype(str).str.replace("$", "", regex=False).astype(float)
    w["_ts"] = pd.to_datetime(dict(year=w["Year"], month=w["Month"], day=w["Day"],
                                   hour=w["Time"].str.split(":").str[0].astype(int),
                                   minute=w["Time"].str.split(":").str[1].astype(int)))
    last, prev = w.iloc[-1], w.iloc[:-1]
    out = []
    med = prev["_amt"].median()
    if med > 0 and last["_amt"] >= 3 * med:
        out.append(("⚠️", f"The amount (${last['_amt']:,.2f}) is {last['_amt']/med:.1f}× the card's typical recent "
                          f"amount (median ${med:,.2f}). Sudden large spends are a classic fraud signal."))
    elif last["_amt"] <= 0:
        out.append(("⚠️", "The amount is zero or negative (a refund or test charge). Fraudsters sometimes test cards with tiny charges."))
    else:
        out.append(("✅", f"The amount (${last['_amt']:,.2f}) is in line with the card's recent behaviour (median ${med:,.2f})."))

    if str(last["Use Chip"]).startswith("Online") and not prev["Use Chip"].astype(str).str.startswith("Online").any():
        out.append(("⚠️", "This is an online purchase but none of the previous 6 were online. A change in payment channel is suspicious."))
    else:
        out.append(("✅", f"Payment method: {last['Use Chip']}. This matches the card's recent habits or is not unusual by itself."))

    if pd.notna(last["Errors?"]):
        out.append(("⚠️", f"The transaction carried an error flag: \"{last['Errors?']}\" (for example a bad PIN or CVV). Repeated errors often come from someone guessing details."))
    if str(last["Merchant City"]) not in set(prev["Merchant City"].astype(str)):
        out.append(("⚠️", f"Merchant city \"{last['Merchant City']}\" has not appeared in the card's previous 6 transactions."))
    else:
        out.append(("✅", f"Merchant city \"{last['Merchant City']}\" was already visited recently."))
    if str(last["Merchant Name"]) not in set(prev["Merchant Name"].astype(str)):
        out.append(("ℹ️", "This merchant is new for this card in the recent window. Not suspicious alone, but it adds up."))
    if last["MCC"] not in set(prev["MCC"]):
        out.append(("ℹ️", f"Merchant category code {last['MCC']} is new for this card in the recent window."))
    gap = (last["_ts"] - prev.iloc[-1]["_ts"]).total_seconds() / 60
    if gap < 5:
        out.append(("⚠️", f"Only {gap:.0f} minute(s) since the previous transaction. Rapid-fire purchases can mean automated misuse."))
    else:
        out.append(("✅", f"{gap/60:.1f} hours since the previous transaction. That is a normal-looking gap." if gap >= 60
                    else f"{gap:.0f} minutes since the previous transaction."))
    return out


def raw_window_from(df, arr_row):
    return df.loc[arr_row].reset_index(drop=False)


# ==========================================================================
# PAGES
# ==========================================================================
def hero(title, sub):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


missing = assets_ok()

# ---------------------------- START HERE ----------------------------------
if PAGE == "🏠 Start here":
    hero("Fraud Detection Studio",
         "An AI that reads a card's last 7 transactions and tells you how suspicious the newest one looks, "
         "and explains every number it shows you.")
    st.markdown("### How this app works in 60 seconds")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. It looks at a sequence**\n\nA single purchase rarely proves fraud. The model reads the "
                "6 earlier purchases on the same card *plus* the new one, so it knows what 'normal' looks like for that card.")
    c2.markdown("**2. It gives a score from 0 to 1**\n\nClose to 0 means 'looks normal'. Close to 1 means "
                "'looks like fraud'. The score is a degree of suspicion, not a certainty.")
    c3.markdown("**3. A threshold turns it into a decision**\n\nYou choose the cut-off. Scores above it are "
                "flagged. A lower cut-off catches more fraud but annoys more honest customers.")
    why("Why a sequence model (LSTM)?",
        "Fraud is about <i>change</i>: a purchase in a new city, at an odd hour, right after a failed PIN. "
        "An LSTM (Long Short-Term Memory network) is built to remember what happened earlier in a sequence, "
        "so it can notice that the newest transaction does not fit the card's recent pattern.")
    st.markdown("### Where to go next")
    st.markdown("- **📊 Performance**: how good is the model overall, and how to pick a threshold.\n"
                "- **🔍 Inspect a transaction**: open one case and see *why* it scored high or low.\n"
                "- **📤 Score your own file**: upload transactions in the same format and get scores.\n"
                "- **📖 Glossary**: every term explained in plain words.")
    if missing:
        why("Setup needed",
            "These files from your notebook were not found:<br><code>" + "</code><br><code>".join(missing) +
            "</code><br>Run the notebook first (it creates them) or fix the paths in the sidebar under "
            "<i>File locations</i>.", "warn")
    else:
        why("Ready", "The model files were found. Use the menu on the left.", "ok")

# ---------------------------- MODEL & DATA ---------------------------------
elif PAGE == "🧠 Model & data":
    hero("Model & data", "What the AI is made of, and how raw transactions become numbers it can read.")
    st.markdown("### The data")
    st.markdown("Each row is one credit-card transaction with: user, card, date and time, amount, how the card was used "
                "(swipe, chip or online), merchant name/city/state, zip, merchant category code (MCC), an error flag, "
                "and the label *Is Fraud?* (Yes/No). The notebook uses about **245,000** transactions after removing the "
                "first 6 of every card (they have no full history).")
    why("Why remove the first 6 transactions of each card?",
        "The model needs 7 transactions in a row (6 past + 1 current). The first transactions of a card do not have "
        "6 predecessors, so there is nothing to compare them against.")
    st.markdown("### From raw columns to numbers")
    steps = pd.DataFrame({
        "Column": ["Amount", "Date + Time", "Merchant name / city / state, MCC, Zip", "Use Chip, Errors?", "Is Fraud?"],
        "What is done": ["'$' removed, log taken, scaled 0 to 1", "Converted to one timestamp, scaled 0 to 1",
                         "Turned into a code, split into decimal digits, one-hot encoded",
                         "Missing filled, converted to 0/1 flags", "Yes → 1, No → 0 (the answer the model learns)"],
        "Why": ["Log stops huge amounts from dominating; scaling keeps all inputs comparable",
                "Lets the model compare time gaps between purchases",
                "Computers cannot read text; digit splitting keeps the input small even for thousands of merchants",
                "Neural networks need numbers, not words", "This is the target we want to predict"],
    })
    st.dataframe(steps, use_container_width=True, hide_index=True)
    st.markdown("### The network")
    fig = go.Figure(go.Sankey(
        node=dict(label=["7 transactions × 189 features", "LSTM layer 1 (200 units)", "LSTM layer 2 (200 units)",
                         "Dense + sigmoid", "Fraud score 0–1"],
                  color=["#94a3b8", "#4f46e5", "#7c3aed", "#0ea5e9", "#16a34a"], pad=30),
        link=dict(source=[0, 1, 2, 3], target=[1, 2, 3, 4], value=[1, 1, 1, 1])))
    st.plotly_chart(style_fig(fig, 260), use_container_width=True)
    st.markdown("**Parameters:** 312,000 + 320,800 + 201 = **633,001** (about 2.4 MB). The model is small on purpose so it "
                "can score a card payment in milliseconds.")
    why("What does each part do?",
        "<b>LSTM layer 1</b> reads the 7 transactions one by one and builds a memory of the pattern. "
        "<b>LSTM layer 2</b> refines that memory into higher-level ideas (for example 'spending speeds up while location changes'). "
        "<b>Dense + sigmoid</b> squeezes everything into a single number between 0 and 1, the fraud score.")
    st.markdown("### How it was trained")
    why("Why training used a 50/50 mix of fraud and normal cases",
        "Real fraud is extremely rare (roughly 1 in a few thousand). If the model saw real proportions it could reach "
        "99.9% accuracy by always answering 'not fraud'. So each training batch used all fraud examples plus an equal "
        "number of random normal ones. <b>Side effect:</b> the model learns that fraud is common, so its raw scores are "
        "pushed upwards, which is why you will see false alarms on real data and why the threshold matters (see Performance).")

# ---------------------------- PERFORMANCE ----------------------------------
elif PAGE == "📊 Performance":
    hero("Performance", "How well does the model separate fraud from normal behaviour on transactions it never trained on?")
    if missing or not (os.path.exists(TEST_CSV) and os.path.exists(TEST_IDX)):
        why("Files missing", "Need model files plus the test sample (<code>test_220_40k.csv</code> and "
            "<code>.indices</code>). Check the sidebar paths.", "warn")
        st.stop()
    _, _idx = load_test(TEST_CSV, TEST_IDX)
    n = st.slider("How many test transactions to score", 500, int(len(_idx)), min(3000, int(len(_idx))), step=500,
                  help="More transactions give a more reliable picture, but take longer to process.")
    why("Why a slider?", "Scoring runs the full model on every transaction. Start small for speed; go to the maximum for the most trustworthy numbers.")
    X, y, s, idx = score_sample(SAVE_DIR, TEST_CSV, TEST_IDX, n)
    n_fraud = int(y.sum())

    thr = st.slider("Decision threshold (flag as fraud if score ≥ this)", 0.01, 0.99, 0.50, 0.01)
    m = metrics_at(y, s, thr)
    st.markdown(f"### Results at threshold **{thr:.2f}**")
    c = st.columns(5)
    c[0].metric("Accuracy", f"{m['acc']*100:.2f}%")
    c[1].metric("Precision", f"{m['precision']*100:.1f}%")
    c[2].metric("Recall", f"{m['recall']*100:.1f}%")
    c[3].metric("F1", f"{m['f1']:.3f}")
    c[4].metric("False-alarm rate", f"{m['far']*100:.2f}%")

    why("What do these five numbers mean?",
        f"Out of <b>{len(y):,}</b> transactions, <b>{n_fraud}</b> were truly fraud.<br>"
        f"• <b>Accuracy {m['acc']*100:.2f}%</b>: share of all decisions that were right. "
        f"<i>Misleading here</i>: since fraud is so rare, answering 'not fraud' every time would already score "
        f"{(1-n_fraud/len(y))*100:.2f}%. Do not judge the model by this alone.<br>"
        f"• <b>Precision {m['precision']*100:.1f}%</b>: when the model says 'fraud', how often it is correct. "
        f"Low precision = many honest customers get blocked.<br>"
        f"• <b>Recall {m['recall']*100:.1f}%</b>: of all real frauds, how many the model caught. "
        f"Low recall = fraud slips through.<br>"
        f"• <b>F1 {m['f1']:.3f}</b>: one balanced number combining precision and recall (1 is perfect).<br>"
        f"• <b>False-alarm rate {m['far']*100:.2f}%</b>: share of honest transactions wrongly flagged.")

    st.markdown("### Confusion matrix")
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted normal", "Predicted fraud"], y=["Actually normal", "Actually fraud"],
                               text=[[f"{v:,}" for v in r] for r in z], texttemplate="%{text}",
                               textfont=dict(size=22), colorscale="Purples", showscale=False))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    why("How to read the 2×2 box",
        f"<b>Top-left ({m['tn']:,})</b>: honest transactions correctly allowed. "
        f"<b>Top-right ({m['fp']:,})</b>: honest ones wrongly flagged (false alarms, annoyed customers). "
        f"<b>Bottom-left ({m['fn']:,})</b>: frauds that got through (the costly mistake). "
        f"<b>Bottom-right ({m['tp']:,})</b>: frauds correctly caught. A perfect model has numbers only on the diagonal "
        f"(top-left and bottom-right).")
    if m["fp"] > m["tp"] and m["tp"] > 0:
        why("Why are there more false alarms than caught frauds?",
            "Because honest transactions outnumber fraud by thousands to one. Even a tiny error rate on the honest side "
            "creates many false alarms in absolute terms. Training on a 50/50 mix also makes the model a bit "
            "over-suspicious. Raising the threshold (slider above) reduces false alarms, usually at the cost of some recall.", "warn")

    st.markdown("### Score distribution")
    fig = go.Figure()
    fig.add_histogram(x=s[y == 0], name="Normal", marker_color="#4f46e5", opacity=0.7, nbinsx=50)
    fig.add_histogram(x=s[y == 1], name="Fraud", marker_color="#dc2626", opacity=0.8, nbinsx=50)
    fig.add_vline(x=thr, line_dash="dash", line_color="#f59e0b", annotation_text=f"threshold {thr:.2f}")
    fig.update_layout(barmode="overlay", xaxis_title="Fraud score", yaxis_title="Number of transactions (log scale)")
    fig.update_yaxes(type="log")
    st.plotly_chart(style_fig(fig), use_container_width=True)
    why("What am I looking at?",
        "Each bar counts transactions with a given score. Blue = truly normal, red = truly fraud. A good model piles blue "
        "on the far left and red on the far right with little overlap. The dashed line is your threshold: everything to "
        "its right gets flagged. The vertical axis is logarithmic so the few red cases remain visible next to the huge blue pile.")

    st.markdown("### Trade-off curve: precision vs. recall for every threshold")
    ths = np.linspace(0.02, 0.98, 49)
    rows = [metrics_at(y, s, t) for t in ths]
    fig = go.Figure()
    fig.add_scatter(x=ths, y=[r["precision"] for r in rows], name="Precision", line=dict(color="#4f46e5", width=3))
    fig.add_scatter(x=ths, y=[r["recall"] for r in rows], name="Recall", line=dict(color="#dc2626", width=3))
    fig.add_scatter(x=ths, y=[r["f1"] for r in rows], name="F1", line=dict(color="#16a34a", width=3, dash="dot"))
    fig.add_vline(x=thr, line_dash="dash", line_color="#f59e0b")
    fig.update_layout(xaxis_title="Threshold", yaxis_title="Value (0–1)")
    st.plotly_chart(style_fig(fig), use_container_width=True)
    best = ths[int(np.argmax([r["f1"] for r in rows]))]
    why("How do I pick a threshold?",
        f"As the threshold rises, precision usually goes up (fewer false alarms) and recall goes down (more fraud missed). "
        f"On this sample the best balance (highest F1) is around <b>{best:.2f}</b>. Choose lower if missing fraud is "
        f"very costly; choose higher if blocking honest customers is very costly.")

# ---------------------------- INSPECT --------------------------------------
elif PAGE == "🔍 Inspect a transaction":
    hero("Inspect a transaction", "Open one case, see the 7 transactions the model saw, its score, and exactly what drove it.")
    if missing or not (os.path.exists(TEST_CSV) and os.path.exists(TEST_IDX)):
        why("Files missing", "Model files or the test sample were not found. Check the sidebar paths.", "warn")
        st.stop()
    mapper = load_mapper(SAVE_DIR)
    df, _ = load_test(TEST_CSV, TEST_IDX)
    X, y, s, idx = score_sample(SAVE_DIR, TEST_CSV, TEST_IDX, 3000)
    thr = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01)
    mode = st.radio("Which cases?", ["Truly fraudulent", "Flagged by the model", "Highest scores", "Any (browse)"], horizontal=True)
    if mode == "Truly fraudulent":
        pool = np.where(y == 1)[0]
    elif mode == "Flagged by the model":
        pool = np.where(s >= thr)[0]
    elif mode == "Highest scores":
        pool = np.argsort(-s)[:50]
    else:
        pool = np.arange(len(y))
    if len(pool) == 0:
        why("Nothing to show", "No cases match this filter. Try another option or lower the threshold.", "warn")
        st.stop()
    pick = st.selectbox("Case", pool, format_func=lambda i: f"Case #{i}  |  score {s[i]:.3f}  |  actual: {'FRAUD' if y[i] else 'normal'}")

    score = float(s[pick])
    flagged = score >= thr
    left, right = st.columns([1, 1.4])
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score * 100, number=dict(suffix="%", font=dict(family=FONT)),
            title=dict(text="Fraud suspicion score"),
            gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#dc2626" if flagged else "#16a34a"),
                       threshold=dict(line=dict(color="#f59e0b", width=4), value=thr * 100))))
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)
    with right:
        verdict = "🚩 FLAGGED as suspicious" if flagged else "✅ Allowed (looks normal)"
        truth = "Actually FRAUD" if y[pick] else "Actually normal"
        st.markdown(f"### {verdict}")
        st.markdown(f"**Ground truth:** {truth}")
        if flagged and y[pick]:
            outcome = "Correct catch (true positive)."
        elif flagged and not y[pick]:
            outcome = "False alarm (false positive): an honest customer would have been blocked."
        elif (not flagged) and y[pick]:
            outcome = "Missed fraud (false negative): the costly kind of mistake."
        else:
            outcome = "Correct pass (true negative)."
        st.markdown(f"**Outcome:** {outcome}")
    why("What does the gauge mean?",
        f"The needle shows how suspicious the model finds the <b>latest</b> transaction: <b>{score*100:.1f}%</b>. "
        f"The orange line is your threshold ({thr*100:.0f}%). Needle past the line = flagged. "
        f"This is a degree of suspicion, <i>not</i> a literal probability, because the model was trained on a 50/50 fraud mix.")

    arr = idx[pick] + 1 - TIMESTEPS + np.arange(TIMESTEPS)
    w = df.loc[arr]
    st.markdown("### The 7 transactions the model read (oldest → newest)")
    show = w[["Year", "Month", "Day", "Time", "Amount", "Use Chip", "Merchant Name", "Merchant City",
              "Merchant State", "MCC", "Errors?"]].reset_index(drop=True)
    show.index = [f"t-{TIMESTEPS-1-i}" for i in range(TIMESTEPS)]
    show.index = show.index.map(lambda v: "t (NEWEST)" if v == "t-0" else v)
    st.dataframe(show, use_container_width=True)
    why("Why 7?", "The model was trained on windows of 7: six past purchases give context, the seventh is the one being judged. "
        "Only the last row is scored; the first six are the card's 'normal' reference.")

    st.markdown("### Plain-English reasons")
    try:
        for icon, txt in describe_window(w):
            st.markdown(f"{icon} {txt}")
    except Exception as e:
        st.caption(f"Could not build the readable summary ({e}).")
    why("How to use these bullets",
        "⚠️ = something unusual compared with the card's recent history, ✅ = looks ordinary, ℹ️ = mild note. "
        "These are simple human-readable checks; the model weighs hundreds of signals at once, so it may disagree "
        "with the bullets. That is normal and exactly why it is useful.")

    st.markdown("### Which transaction in the window mattered most?")
    model = load_model(SAVE_DIR, X.shape[-1])
    base = X.mean(axis=(0, 1))
    variants = np.repeat(X[pick:pick + 1], TIMESTEPS, axis=0)
    for i in range(TIMESTEPS):
        variants[i, i, :] = base
    new = model.predict(variants, verbose=0)[:, -1, 0]
    delta = (score - new) * 100
    labels = [f"t-{TIMESTEPS-1-i}" if i < TIMESTEPS - 1 else "t (newest)" for i in range(TIMESTEPS)]
    fig = go.Figure(go.Bar(x=labels, y=delta, marker_color=["#dc2626" if d > 0 else "#4f46e5" for d in delta]))
    fig.update_layout(yaxis_title="Score change when replaced by an 'average' transaction (points)")
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    top = labels[int(np.argmax(np.abs(delta)))]
    why("What is this chart?",
        f"For each bar we swapped one of the 7 transactions for an 'average' one and re-scored the case. "
        f"A <b style='color:#dc2626'>red bar</b> means that transaction was <i>pushing the score up</i> (removing it lowers suspicion by that many points). "
        f"A <b style='color:#4f46e5'>blue bar</b> means it was <i>calming the score</i>. The biggest effect here comes from <b>{top}</b>. "
        f"This is an approximation (an 'occlusion test'), not an exact explanation of the network's inner workings.")

# ---------------------------- UPLOAD ---------------------------------------
elif PAGE == "📤 Score your own file":
    hero("Score your own file", "Upload transactions in the same format as the training data and get a fraud score for each one.")
    if missing:
        why("Model files missing", "<code>" + "</code><br><code>".join(missing) + "</code>", "warn")
        st.stop()
    why("What format do I need?",
        "A CSV with these columns: <code>User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name, "
        "Merchant City, Merchant State, Zip, MCC, Errors?</code> (and optionally <code>Is Fraud?</code>). "
        "<i>Time</i> looks like 14:35 and <i>Amount</i> like $23.50. Each card needs at least 7 transactions, because "
        "the first 6 only provide history; scores start from the 7th.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    thr = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01)
    if up is not None:
        try:
            raw = pd.read_csv(up, dtype={"Merchant Name": "str"})
            raw["Merchant Name"] = raw["Merchant Name"].astype(str)
            if "Is Fraud?" not in raw.columns:
                raw["Is Fraud?"] = "No"
                has_truth = False
            else:
                has_truth = True
            raw = raw.sort_values(["User", "Card"], kind="stable").reset_index(drop=True)
            pos = raw.groupby(["User", "Card"]).cumcount().to_numpy()
            targets = np.where(pos >= TIMESTEPS - 1)[0]
            if len(targets) == 0:
                why("Not enough history", "No card has 7 or more transactions, so nothing can be scored.", "warn")
                st.stop()
            mapper = load_mapper(SAVE_DIR)
            with st.spinner("Scoring..."):
                Xu, yu, _ = build_windows(raw, targets, mapper)
                model = load_model(SAVE_DIR, Xu.shape[-1])
                su = predict(model, Xu)
            out = raw.loc[targets, ["User", "Card", "Year", "Month", "Day", "Time", "Amount", "Merchant Name", "Merchant City"]].copy()
            out["Fraud score"] = su.round(4)
            out["Verdict"] = np.where(su >= thr, "FLAGGED", "ok")
            out["Reason hint"] = ""
            for k, ridx in enumerate(targets):
                if su[k] >= thr:
                    try:
                        warns = [t for ic, t in describe_window(raw.loc[ridx - TIMESTEPS + 1:ridx]) if ic == "⚠️"]
                        out.iloc[k, out.columns.get_loc("Reason hint")] = warns[0][:140] if warns else "Pattern across the sequence"
                    except Exception:
                        pass
            c = st.columns(3)
            c[0].metric("Transactions scored", f"{len(out):,}")
            c[1].metric("Flagged", f"{int((su >= thr).sum()):,}")
            c[2].metric("Flag rate", f"{(su >= thr).mean()*100:.2f}%")
            why("What am I seeing?",
                f"Out of {len(out):,} scorable transactions, <b>{int((su >= thr).sum()):,}</b> scored at or above {thr:.2f}. "
                f"The first 6 transactions of every card were used only as history, so they are not in the table. "
                f"'Reason hint' gives the first unusual fact about a flagged case; open the full explanation by "
                f"using the Inspect page with the same style of data.")
            st.dataframe(out.sort_values("Fraud score", ascending=False), use_container_width=True, hide_index=True)
            if has_truth:
                mm = metrics_at(yu, su, thr)
                why("Your file had an 'Is Fraud?' column, so here is how the model did on it",
                    f"Precision {mm['precision']*100:.1f}%, recall {mm['recall']*100:.1f}%, F1 {mm['f1']:.3f} "
                    f"(caught {mm['tp']} of {mm['tp']+mm['fn']} frauds with {mm['fp']} false alarms).")
            st.download_button("⬇️ Download scored CSV", out.to_csv(index=False).encode(), "scored_transactions.csv", "text/csv")
        except ValueError as e:
            why("The model could not read this file",
                f"A value it has never seen before (such as a new merchant or MCC) breaks the pre-trained encoders. "
                f"The preprocessing was fitted on the original dataset only. Technical message: <code>{e}</code>", "warn")
        except Exception as e:
            why("Something went wrong", f"<code>{e}</code><br>Check that the column names and formats match the instructions above.", "warn")

# ---------------------------- GLOSSARY -------------------------------------
else:
    hero("Glossary", "Every term used in this app, in everyday language.")
    terms = {
        "Fraud score": "A number from 0 to 1 produced by the model. Higher = more suspicious.",
        "Threshold": "The cut-off you choose. Scores at or above it are flagged as fraud.",
        "LSTM": "Long Short-Term Memory: a neural network that reads things in order and remembers important earlier steps.",
        "Sequence / window": "The 7 consecutive transactions of one card given to the model (6 past + 1 current).",
        "True positive (TP)": "Fraud that the model correctly flagged.",
        "False positive (FP)": "A normal transaction wrongly flagged (false alarm).",
        "False negative (FN)": "Fraud the model missed.",
        "True negative (TN)": "A normal transaction correctly allowed.",
        "Accuracy": "Share of all decisions that were right. Misleading when fraud is very rare.",
        "Precision": "When the model says fraud, how often it is right: TP / (TP + FP).",
        "Recall": "How much of the real fraud was caught: TP / (TP + FN).",
        "F1 score": "Single balance of precision and recall; 1 is perfect, 0 is useless.",
        "False-alarm rate": "Share of honest transactions that were wrongly flagged: FP / (FP + TN).",
        "MCC": "Merchant Category Code: a 4-digit number describing the type of business (e.g. groceries, fuel).",
        "One-hot encoding": "Turning a category into a row of 0/1 switches so a neural network can use it.",
        "Occlusion test": "Replace one input with a neutral value and see how much the score changes. Shows which input mattered.",
        "Imbalanced data": "When one outcome (fraud) is far rarer than the other. Needs special training and careful metrics.",
        "NNPA / zDNN": "IBM Z hardware accelerator for AI. Used in your notebook for fast inference; this app runs on any computer.",
    }
    q = st.text_input("Search a term")
    for k, v in terms.items():
        if q.lower() in k.lower() or q.lower() in v.lower():
            st.markdown(f"**{k}**: {v}")
    why("Still unsure about a number?", "Every page has a purple 'why' box under each result. Read it first; it explains what the number "
        "means for *this* data right now.")
