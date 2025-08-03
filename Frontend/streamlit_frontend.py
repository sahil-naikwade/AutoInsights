import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from utils.config import Config

# --- Helper: Get user_id from session state only ---
def get_user_id():
    # Only allow user_id from session state (set at login)
    return st.session_state.get('user_id', None)

# --- Database Connection (cached) ---
@st.cache_resource
def get_db_engine():
    try:
        db_url = Config.get_db_url()
        engine = create_engine(db_url)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.warning("Please ensure your database is running and credentials in `.env` are correct.")
        return None

engine = get_db_engine()
if engine is None:
    st.stop()

# --- Data Loading: User-specific ---
def load_data_from_mysql(table, user_id):
    if not user_id:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            query = text(f"SELECT * FROM {table} WHERE user_id = :user_id")
            df = pd.read_sql(query, conn, params={"user_id": user_id})
        return df
    except Exception as e:
        st.error(f"Error loading data from {table}: {e}")
        return pd.DataFrame()

def get_user_data(email):
    """
    Fetches the latest analysis data for a given user email.
    Uses email as the user_id everywhere for maximum security.
    """
    if not email:
        st.warning("No user email provided. Cannot fetch data.")
        return pd.DataFrame()

    with engine.connect() as conn:
        # Step 1: Ensure user exists
        user_result = conn.execute(
            text("SELECT email FROM users WHERE email = :email"),
            {"email": email}
        ).fetchone()
        if not user_result:
            st.error(f"No user found in the database for email: {email}")
            return pd.DataFrame()

        # Step 2: Fetch all data for that email from the dashboard view/table
        query = text("SELECT * FROM full_dashboard_data WHERE user_id = :user_id")
        user_df = pd.read_sql(query, conn, params={"user_id": email})
        if user_df.empty:
            st.warning(f"No analysis data found for user: {email}.")
            return pd.DataFrame()

        # Step 3: Find and filter by the latest analysis_id for this user
        if 'analysis_id' in user_df.columns and not user_df['analysis_id'].isnull().all():
            latest_analysis_id = user_df['analysis_id'].max()
            latest_df = user_df[user_df['analysis_id'] == latest_analysis_id].copy()
            return latest_df
        else:
            st.warning("No 'analysis_id' found or all are null. Cannot determine the latest analysis.")
            return pd.DataFrame()

# --- Main App ---
st.set_page_config(page_title="AutoInsights Dashboard", layout="wide", page_icon="📊")
st.title("AutoInsights Dashboard")

user_id = get_user_id()
if not user_id:
    st.error("You are not authenticated. Please log in to view your dashboard.")
    st.stop()

# --- Load user-specific data ---
churn_data = load_data_from_mysql('churn_data', user_id)
revenue_data = load_data_from_mysql('revenue_data', user_id)

if churn_data.empty and revenue_data.empty:
    st.error("No data found for your account. Please check your login and data upload.")
    st.stop()

# --- Overview Section ---
st.header("Overview")
if not churn_data.empty:
    st.write("### Churn Data Summary")
    st.write(churn_data.describe())
else:
    st.warning("No churn data available for your account.")

if not revenue_data.empty:
    st.write("### Revenue Data Summary")
    st.write(revenue_data.describe())
else:
    st.warning("No revenue data available for your account.")

# --- Visualizations ---
st.write("### Churn Analysis")
if not churn_data.empty and 'churn_label' in churn_data.columns:
    fig1 = px.histogram(churn_data, x='churn_label', title="Customer Churn Distribution")
    st.plotly_chart(fig1)
else:
    st.warning("No churn label data to visualize.")

st.write("### Revenue Segmentation")
if not revenue_data.empty and 'revenue_segment' in revenue_data.columns and 'revenue' in revenue_data.columns:
    fig2 = px.bar(revenue_data, x='revenue_segment', y='revenue', title="Revenue by Value Segment")
    st.plotly_chart(fig2)
else:
    st.warning("No revenue segment data to visualize.")

# --- Detailed Sections ---
st.header("Detailed Analysis")
if not churn_data.empty:
    st.subheader("Churn Data")
    st.write(churn_data.head())
else:
    st.warning("No churn data to display.")

if not revenue_data.empty:
    st.subheader("Revenue Data")
    st.write(revenue_data.head())
else:
    st.warning("No revenue data to display.")
