import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Hackathon Analytics", layout="wide", initial_sidebar_state="expanded")

# --- 2. FORCE SCROLL TO TOP ---
components.html("""<script>window.parent.document.querySelector('section.main').scrollTo(0, 0);</script>""", height=0)

# --- 3. CSS STYLING ---
st.markdown("""
<style>
    .section-header { font-size: 1.4rem !important; font-weight: 700 !important; border-bottom: 2px solid #3498db !important; padding-bottom: 10px; margin-top: 40px; margin-bottom: 20px; }
    .footer-card { background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 25px; margin-top: 60px; margin-bottom: 20px; }
    .footer-title { font-weight: 700; font-size: 1.1rem; margin-bottom: 10px; color: #3498db; }
    .footer-text { font-size: 0.95rem; opacity: 0.9; line-height: 1.6; }
    code { color: #e83e8c; background-color: rgba(255, 255, 255, 0.1); padding: 2px 4px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# --- 4. LOAD SYSTEM ---
@st.cache_resource
def load_system():
    try:
        model = joblib.load('xgb_model.pkl')
        encoder = joblib.load('label_encoder.pkl')
        return model, encoder
    except FileNotFoundError:
        st.error("System Error: Model artifacts not found."); st.stop()

model, encoder = load_system()

# --- 5. SIDEBAR ---
st.sidebar.header("Simulation Parameters")
experience = st.sidebar.slider("Average Experience (Years)", 1, 10, 4)
tech_stack = st.sidebar.selectbox("Tech Stack Diversity", ("Low", "Medium", "High"), index=1)
commit_vel = st.sidebar.slider("Commit Velocity (Commits/Hr)", 5, 50, 25)
comm_freq = st.sidebar.slider("Communication Freq (Msgs/Hr)", 10, 100, 50)
sleep = st.sidebar.slider("Sleep Schedule (Hours)", 2.0, 8.0, 6.0)
wins = st.sidebar.slider("Prior Hackathon Wins", 0, 5, 1)
st.sidebar.markdown("---")
st.sidebar.caption("Analytics Dashboard v5.1")

# --- 6. ENGINE ---
input_df = pd.DataFrame({ 'Team_Experience_Avg': [experience], 'Tech_Stack_Diversity': [tech_stack], 'Commit_Velocity': [commit_vel], 'Communication_Freq': [comm_freq], 'Sleep_Hours': [sleep], 'Prior_Wins': [wins] })
input_df_processed = input_df.copy()
input_df_processed['Tech_Stack_Diversity'] = encoder.transform(input_df['Tech_Stack_Diversity'])
prob = model.predict_proba(input_df_processed)[0][1]

# --- 7. TIER LOGIC ---
def get_tier_info(p):
    if p < 0.20:
        return "Critical Failure", "The model detects a near-zero probability of success. Major structural changes required."
    elif p < 0.45:
        return "At Risk", "Performance metrics are significantly below the winning threshold. Immediate intervention needed."
    elif p < 0.60:
        return "Average / Bubble", "Your team is on the borderline. You are performing adequately but lack a competitive edge."
    elif p < 0.75:
        return "Contender", "Strong performance detected. You are in the upper percentile but not yet dominant."
    elif p < 0.90:
        return "Top Tier", "Excellent dynamics. Your team exhibits the traits of a podium-finisher."
    else:
        return "Elite / Dominating", "Exceptional stats. The model predicts a definitive win based on current telemetry."

status_label, status_message = get_tier_info(prob)

# --- 8. DASHBOARD ---
st.title("Hackathon Performance Analytics")
st.markdown("### Predictive Modeling & Telemetry Analysis")

# SECTION: KPIs
st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

# [FIX]: Increased width of col2 and col3 to prevent text cutoff
col1, col2, col3 = st.columns([1.3, 2.5, 3])

with col1:
    st.metric("Win Probability", f"{prob*100:.1f}%", delta=f"{(prob-0.5)*100:.1f}%")
with col2:
    st.metric("Classification", status_label)
with col3:
    st.write("**Primary Constraint**")
    if sleep < 5: st.error("⚠️ **CRITICAL: Sleep Deprivation Detected**")
    elif commit_vel < 20: st.warning("⚠️ **WARNING: Low Commit Velocity**")
    elif comm_freq < 40: st.warning("⚠️ **WARNING: Communication Silos**")
    elif prob < 0.5: st.warning("⚠️ **General Performance Lag:** Stats are too average.")
    else: st.success("✅ **OPTIMAL: No Constraints Detected**")

# SECTION: STRATEGIC ANALYSIS
st.markdown('<div class="section-header">Strategic Analysis & Feature Impact</div>', unsafe_allow_html=True)
explainer = shap.TreeExplainer(model)
shap_values = explainer(input_df_processed)
shap_vals = shap_values[0].values
feature_names = ["Experience", "Stack Diversity", "Commit Velocity", "Communication", "Sleep", "Prior Wins"]

fig, ax = plt.subplots(figsize=(12, 6))
df_shap = pd.DataFrame({'feature': feature_names, 'value': shap_vals})
df_shap = df_shap.sort_values('value', ascending=True)
colors = ['#d62728' if x < 0 else '#1f77b4' for x in df_shap['value']]
ax.barh(df_shap['feature'], df_shap['value'], color=colors, height=0.6)
ax.axvline(0, color='black', linewidth=0.8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False); ax.spines['bottom'].set_color('#888')
ax.tick_params(axis='x', colors='#888'); ax.tick_params(axis='y', colors='#888')
fig.patch.set_alpha(0); ax.patch.set_alpha(0)
st.pyplot(fig, use_container_width=True)

# Assessment Logic
st.markdown("##### **AI Strategic Assessment**")
st.caption("Based on the feature impact visualized above, the system recommends:")

issues_found = False
if sleep < 5.0: st.error(f"❌ **Cognitive Decline:** Sleep ({sleep}h) is critically low."); issues_found = True
if commit_vel < 25: st.warning(f"⚠️ **Velocity Lag:** Commit rate ({commit_vel}/hr) is low."); issues_found = True
if comm_freq < 40: st.info(f"ℹ️ **Comm Gap:** Msgs/hr ({comm_freq}) is below target."); issues_found = True

if not issues_found:
    if prob >= 0.5:
        st.success(f"✅ **System Status:** {status_message}")
    else:
        st.warning(f"⚠️ **System Status:** {status_message}")
else:
    st.write(f"📝 **Model Context:** {status_message}")

# SECTION: TRANSPARENCY
footer_html = """
<div class="footer-card">
<h4 style="margin-top:0; color:#3498db;">ℹ️ Model & Data Transparency</h4>
<div style="display: flex; flex-wrap: wrap; gap: 40px;">
<div style="flex: 1; min-width: 300px;">
<div class="footer-title">Where did the data come from?</div>
<div class="footer-text">The dataset was curated from <b>Open Source Online Repositories</b> specializing in developer productivity.<br>Volume: ~2,000 historical records.</div>
</div>
<div style="flex: 1; min-width: 300px;">
<div class="footer-title">Where was the model trained?</div>
<div class="footer-text">The model is an <b>XGBoost Classifier</b> trained locally via <code>train_model.py</code>.<br><b>Note:</b> This web view operates via a live Python runtime.</div>
</div>
</div>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)