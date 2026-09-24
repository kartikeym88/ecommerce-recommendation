import streamlit as st
import pandas as pd
import os

if not os.path.exists("data/raw/users.csv"):
    from src.recsys.data.generator import SyntheticDataGenerator
    from src.recsys.config import settings
    gen = SyntheticDataGenerator(n_users=1000, n_products=200, n_days=180, seed=settings.seed)
    gen.write(settings.paths.data_dir)

from dashboard.api_client import get_users, get_recommendations, get_activity
from dashboard.components import inject_css, render_activity_table, render_recommendation_row

st.set_page_config(page_title="Recommendation System", layout="wide")

inject_css()

st.title("Recommendation System")

try:
    users = get_users()
except Exception as e:
    st.error("No data found. Ensure 'make data' has been run.")
    st.stop()

st.sidebar.header("Controls")
user_id = st.sidebar.selectbox("Select User", options=[u["user_id"] for u in users], format_func=lambda x: f"User {x}")
n_recs = st.sidebar.slider("Number of recommendations", 3, 20, 10)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader(f"Activity (User {user_id})")
    activity = get_activity(user_id)
    if activity:
        render_activity_table(activity)
    else:
        st.write("No activity history.")

with col2:
    st.subheader("Recommendations")
    with st.spinner("Loading recommendations..."):
        recs, strategy = get_recommendations(user_id, n_recs)
    
    st.markdown(f"<span class='badge'>STRATEGY: {strategy.upper()}</span>", unsafe_allow_html=True)
    st.write("")
    
    for i, rec in enumerate(recs, 1):
        render_recommendation_row(i, rec)
