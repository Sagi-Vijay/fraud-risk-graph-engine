import streamlit as st

st.set_page_config(page_title="Fraud Risk Engine", layout="wide")

st.title("Fraud Risk Graph Engine")
st.markdown("---")

st.subheader("Project Status")
st.info("Phase 1 initialized. Upcoming: synthetic data + fraud modeling.")

st.subheader("Demo (placeholder)")
transaction_amount = st.number_input("Transaction Amount", value=100.0)
num_prev_txn = st.number_input("Transactions in last hour", value=1)

if st.button("Score Transaction"):
    # placeholder logic
    risk_score = min(1.0, (transaction_amount / 1000) + (num_prev_txn * 0.05))

    if risk_score < 0.3:
        decision = "APPROVE"
    elif risk_score < 0.7:
        decision = "REVIEW"
    else:
        decision = "DECLINE"

    st.metric("Risk Score", round(risk_score, 3))
    st.success(f"Decision: {decision}")
