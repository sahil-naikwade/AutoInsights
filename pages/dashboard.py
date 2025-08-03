import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from utils.config import Config
import datetime as dt

# --- Page Configuration ---
st.set_page_config(page_title="AutoInsights Dashboard", layout="wide", page_icon="📊")

# --- Database Connection ---
@st.cache_resource
def get_db_engine():
    """Creates and caches a database engine connection."""
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

# --- Data Loading and User Verification ---
def get_user_data(email):
    """
    Fetches the latest analysis data for a given user email.
    Uses email as the user_id everywhere for maximum security.
    """
    if not email:
        st.error("Authentication error: No user email in session. Please log in again.")
        st.stop()

    with engine.connect() as conn:
        # Step 1: Ensure user exists
        user_result = conn.execute(
            text("SELECT email FROM users WHERE email = :email"),
            {"email": email}
        ).fetchone()
        if not user_result:
            st.error(f"No user found in the database for email: {email}")
            st.stop()

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

# --- Main Application Logic ---
user_email = st.session_state.get('user_id')
if not user_email:
    st.error("Authentication error: No user email in session. Please log in again.")
    st.stop()

df = get_user_data(user_email)

# --- Debug Output: Show columns, user_ids, and sample data ---
st.write('## Debug Info')
st.write('Columns:', df.columns.tolist())
if 'user_id' in df.columns:
    st.write('Unique user_ids in data:', df['user_id'].unique())
st.write(df.head())

# If no data is returned, stop the app gracefully
if df.empty:
    st.info("Please run an analysis in the main application to see your dashboard.")
    st.stop()

# ============================ START: DATA COMPLETENESS CHECK ============================
# Add a check to see if the loaded data is sparse or incomplete.
key_cols_to_check = ['country', 'gender', 'last_purchase_date', 'total_revenue']
missing_data_warning = False
for col in key_cols_to_check:
    if col in df.columns:
        # Check if the column is mostly or entirely null
        if df[col].isnull().all():
            missing_data_warning = True
            break
    else:
        # The column itself is missing
        missing_data_warning = True
        break

if missing_data_warning:
    st.warning("⚠️ Your latest analysis data was found, but appears to be incomplete.")
    st.info(
        "Key columns like Country, Gender, or Revenue are missing. "
        "This might be due to an issue during the original data upload in the main app. "
        "Please try running a new analysis."
    )
    st.dataframe(df) # Show the incomplete data for debugging
    st.stop()
# ============================= END: DATA COMPLETENESS CHECK =============================

# --- Data Processing and Filters ---
st.sidebar.header("🔎 Filters")

# Convert date columns safely
for col in ['last_purchase_date', 'first_purchase_date']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Clean data BEFORE creating filters to handle NULLs gracefully.
if 'country' in df.columns:
    df['country'] = df['country'].fillna('Unknown')
if 'gender' in df.columns:
    df['gender'] = df['gender'].fillna('Unknown')
if 'revenue_segment' in df.columns:
    df['revenue_segment'] = df['revenue_segment'].fillna('Unknown')


# Create filters based on the cleaned data
country_options = sorted(df['country'].unique())
country = st.sidebar.multiselect('Country', options=country_options, default=country_options)

gender_options = sorted(df['gender'].unique())
gender = st.sidebar.multiselect('Gender', options=gender_options, default=gender_options)

revenue_segment_options = sorted(df['revenue_segment'].unique())
revenue_segment = st.sidebar.multiselect('Revenue Segment', options=revenue_segment_options, default=revenue_segment_options)

# Date range filter
last_purchase_dates = df['last_purchase_date'].dropna()
if not last_purchase_dates.empty:
    date_min = last_purchase_dates.min().date()
    date_max = last_purchase_dates.max().date()
else:
    date_max = dt.date.today()
    date_min = date_max - pd.Timedelta(days=365)

if date_min > date_max:
    date_min, date_max = date_max, date_min

# Handle potential date range errors
try:
    date_range = st.sidebar.date_input(
        'Last Purchase Date Range',
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max
    )
    date_start = pd.to_datetime(date_range[0])
    date_end = pd.to_datetime(date_range[1])
except (IndexError, TypeError, ValueError):
    date_start = pd.to_datetime(date_min)
    date_end = pd.to_datetime(date_max)


# Apply filters sequentially. The data has already been cleaned, so .isin() will work correctly.
mask = pd.Series(True, index=df.index)

if country:
    mask &= df['country'].isin(country)
if gender:
    mask &= df['gender'].isin(gender)
if revenue_segment:
    mask &= df['revenue_segment'].isin(revenue_segment)

# Apply date filter, but intelligently ignore rows with no date (NaT)
if 'last_purchase_date' in df.columns:
    date_mask = (
        (df['last_purchase_date'] >= date_start) &
        (df['last_purchase_date'] <= date_end)
    )
    # Include rows where the date is null (NaT), so they are not filtered out by the date slider.
    mask &= date_mask.fillna(True)

df_filtered = df[mask]


if df_filtered.empty:
    st.warning('No data available for the selected filters. Try adjusting the date range or other filter options.')
    st.stop()

# --- Dashboard UI ---
st.title("📊 AutoInsights Interactive Dashboard")
st.markdown(f"Displaying latest analysis for **{user_email}**")

# --- Key Metrics ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_rev = df_filtered['total_revenue'].sum()
    st.metric("Total Revenue", f"${total_rev:,.2f}")
with col2:
    total_cust = df_filtered['customer_id'].nunique()
    st.metric("Total Customers", f"{total_cust}")
with col3:
    high_risk_count = (df_filtered['risk_level'] == 'High Risk').sum()
    churn_rate = (high_risk_count / total_cust) * 100 if total_cust > 0 else 0
    st.metric("Churn Rate", f"{churn_rate:.1f}%")
with col4:
    st.metric("High Risk Customers", f"{high_risk_count}")

st.markdown("---")

# --- Visualizations ---
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

with row1_col1:
    st.subheader("Customer Risk Profile")
    risk_counts = df_filtered['risk_level'].value_counts().reset_index()
    risk_counts.columns = ['risk_level', 'count']
    fig = px.bar(
        risk_counts,
        x='risk_level',
        y='count',
        color='risk_level',
        title='Customer Count by Risk Level',
        labels={'count': 'Number of Customers', 'risk_level': 'Risk Level'},
        color_discrete_map={'Low Risk': '#2E8B57', 'Medium Risk': '#FFA500', 'High Risk': '#FF6B6B', 'Unknown': '#808080'}
    )
    st.plotly_chart(fig, use_container_width=True)

with row1_col2:
    st.subheader("Revenue by Segment")
    rev_seg = df_filtered.groupby('revenue_segment')['total_revenue'].sum().reset_index()
    fig = px.pie(
        rev_seg,
        names='revenue_segment',
        values='total_revenue',
        title='Total Revenue by Customer Segment',
        color='revenue_segment',
        color_discrete_map={'Low Value': '#FF6B6B', 'Medium Value': '#FFA500', 'High Value': '#2E8B57', 'Unknown': '#808080'}
    )
    st.plotly_chart(fig, use_container_width=True)

with row2_col1:
    st.subheader("Revenue Over Time")
    if 'last_purchase_date' in df_filtered.columns:
        # Ensure NaT values are dropped before resampling
        df_time_filtered = df_filtered.dropna(subset=['last_purchase_date'])
        if not df_time_filtered.empty:
            df_time = df_time_filtered.set_index('last_purchase_date').resample('M')['total_revenue'].sum().reset_index()
            df_time['last_purchase_date'] = df_time['last_purchase_date'].dt.strftime('%Y-%m')
            fig = px.line(
                df_time,
                x='last_purchase_date',
                y='total_revenue',
                title='Monthly Revenue Trend',
                labels={'last_purchase_date': 'Month', 'total_revenue': 'Total Revenue'},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data with valid dates in the selected range to display revenue over time.")


with row2_col2:
    st.subheader("Revenue by Country")
    country_rev = df_filtered.groupby('country')['total_revenue'].sum().nlargest(10).reset_index()
    fig = px.bar(
        country_rev,
        x='country',
        y='total_revenue',
        color='country',
        title='Top 10 Countries by Revenue',
        labels={'country': 'Country', 'total_revenue': 'Total Revenue'}
    )
    st.plotly_chart(fig, use_container_width=True)


st.markdown("---")

st.subheader("Customer Data Table (Filtered)")
st.dataframe(df_filtered, use_container_width=True)
