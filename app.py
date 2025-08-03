import streamlit as st
import os
import pandas as pd
import plotly.express as px
import io
import numpy as np
from datetime import datetime, timedelta
import hashlib
import time
import re
import bcrypt
from processor.clean_and_merge import clean_and_merge, process_single_dataset, create_customer_transaction_data
from processor.churn_model import generate_churn_data
from processor.revenue_model import generate_revenue_data
from sqlalchemy import create_engine, text, exc
from pathlib import Path
from utils.config import Config
import collections
from scripts.sql_upload import upload_to_sql
from streamlit_oauth import OAuth2Component
import requests
import webbrowser
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AutoInsights Dashboard Generator", 
    layout="wide",
    page_icon="📊"
)

# Session security: expire session after 30 minutes of inactivity
SESSION_TIMEOUT_MINUTES = 30
if 'last_active' in st.session_state:
    if (datetime.now() - st.session_state['last_active']).total_seconds() > SESSION_TIMEOUT_MINUTES * 60:
        st.session_state.clear()
        st.warning('Session expired due to inactivity. Please log in again.')
        st.stop()
st.session_state['last_active'] = datetime.now()

# Rate limiting: simple in-memory rate limit for login attempts per IP (3 per 5 minutes)
def get_client_ip():
    # Streamlit does not expose client IP directly; in production, use a reverse proxy to set X-Forwarded-For
    return st.session_state.get('client_ip', 'local')
if 'login_attempts' not in st.session_state:
    st.session_state['login_attempts'] = collections.defaultdict(list)
def rate_limit_login():
    ip = get_client_ip()
    now = time.time()
    # Remove old attempts
    st.session_state['login_attempts'][ip] = [t for t in st.session_state['login_attempts'][ip] if now - t < 300]
    if len(st.session_state['login_attempts'][ip]) >= 3:
        return False
    st.session_state['login_attempts'][ip].append(now)
    return True

def log_security_event(event_type, email, details=""):
    """Log security events for monitoring"""
    timestamp = datetime.now().isoformat()
    log_entry = f"[{timestamp}] {event_type}: email='{email}' {details}"
    
    # Log to file
    log_file = Path(__file__).parent / 'logs' / 'security.log'
    log_file.parent.mkdir(exist_ok=True)
    
    try:
        with open(log_file, 'a') as f:
            f.write(log_entry + '\n')
    except Exception:
        pass  # Don't fail if logging fails
    
    # Also log to database if possible
    engine = get_db_connection()
    if engine:
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO security_log (event_type, email, details, timestamp)
                    VALUES (:event_type, :email, :details, :timestamp)
                """), {
                    "event_type": event_type,
                    "email": email,
                    "details": details,
                    "timestamp": timestamp
                })
                conn.commit()
        except Exception:
            pass  # Don't fail if database logging fails

def init_database_schema():
    """Initialize complete database schema for SQL-only storage with email as user_id."""
    engine = get_db_connection()
    if not engine:
        return False
    
    try:
        with engine.connect() as conn:
            # Create users table with email as PRIMARY KEY
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    email VARCHAR(100) PRIMARY KEY NOT NULL,
                    username VARCHAR(50) DEFAULT 'unknown',
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP NULL,
                    login_attempts INT DEFAULT 0,
                    account_locked BOOLEAN DEFAULT FALSE,
                    lockout_time TIMESTAMP NULL,
                    subscription VARCHAR(20) DEFAULT 'Free',
                    role VARCHAR(20) DEFAULT 'user'
                )
            """))
            
            # Create analysis_history table with user_id as email string
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS analysis_history (
                    analysis_id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(100) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_customers INT,
                    high_risk_customers INT,
                    total_predicted_revenue DECIMAL(15,2),
                    avg_predicted_revenue DECIMAL(15,2),
                    analysis_summary TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(email) ON DELETE CASCADE
                )
            """))
            
            # Create churn_data table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS churn_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(100) NOT NULL,
                    analysis_id INT,
                    customer_id VARCHAR(50),
                    churn_probability DECIMAL(5,4),
                    risk_level VARCHAR(20),
                    total_amount DECIMAL(15,2),
                    transaction_count INT,
                    days_since_last_purchase INT,
                    last_activity DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(email) ON DELETE CASCADE,
                    FOREIGN KEY (analysis_id) REFERENCES analysis_history(analysis_id) ON DELETE CASCADE
                )
            """))
            
            # Create revenue_data table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS revenue_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(100) NOT NULL,
                    analysis_id INT,
                    customer_id VARCHAR(50),
                    predicted_revenue DECIMAL(15,2),
                    total_revenue DECIMAL(15,2),  -- Ensure this column exists
                    revenue_segment VARCHAR(20),
                    transaction_count INT,
                    avg_transaction_value DECIMAL(15,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(email) ON DELETE CASCADE,
                    FOREIGN KEY (analysis_id) REFERENCES analysis_history(analysis_id) ON DELETE CASCADE
                )
            """))
            
            # Create security_log table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS security_log (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    event_type VARCHAR(50) NOT NULL,
                    email VARCHAR(100),
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address VARCHAR(45),
                    user_agent TEXT
                )
            """))
            
            # Create customer_data table for storing customer information
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS customer_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(100) NOT NULL,
                    analysis_id INT,
                    customer_id VARCHAR(50),
                    customer_name VARCHAR(100),
                    age INT,
                    gender VARCHAR(10),
                    country VARCHAR(50),
                    total_spent DECIMAL(15,2),
                    transaction_count INT,
                    first_purchase_date DATE,
                    last_purchase_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(email) ON DELETE CASCADE,
                    FOREIGN KEY (analysis_id) REFERENCES analysis_history(analysis_id) ON DELETE CASCADE
                )
            """))
            
            # Create full_dashboard_data table
            conn.execute(text("""
                CREATE OR REPLACE VIEW full_dashboard_data AS
                SELECT 
                    cd.analysis_id,
                    cd.user_id,
                    cd.customer_id,
                    cd.customer_name,
                    cd.age,
                    cd.gender,
                    cd.country,
                    rd.total_revenue,
                    rd.predicted_revenue,
                    chd.churn_probability,
                    chd.risk_level,
                    rd.revenue_segment,
                    cd.transaction_count,
                    rd.avg_transaction_value,
                    cd.first_purchase_date,
                    cd.last_purchase_date,
                    chd.days_since_last_purchase,
                    ah.timestamp AS created_at
                FROM customer_data cd
                LEFT JOIN analysis_history ah ON cd.analysis_id = ah.analysis_id
                LEFT JOIN churn_data chd ON cd.analysis_id = chd.analysis_id AND cd.customer_id = chd.customer_id
                LEFT JOIN revenue_data rd ON cd.analysis_id = rd.analysis_id AND cd.customer_id = rd.customer_id
            """))
            
            conn.commit()
            return True
    except Exception as e:
        st.error(f"❌ Database schema initialization failed: {str(e)}")
        return False

# Database connection function
def get_db_connection():
    """Get MySQL database connection using environment variables"""
    try:
        from urllib.parse import quote_plus
        db_user = Config.DB_USER
        db_password = quote_plus(Config.DB_PASSWORD)
        db_host = Config.DB_HOST
        db_name = Config.DB_NAME
        db_port = Config.DB_PORT
        connection_string = f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        st.info("💡 Please check your environment variables for DB credentials.")
        return None

def prepare_churn_features(df):
    """Prepare features for churn analysis by aggregating customer data"""
    df_copy = df.copy()
    
    # Automatic column mapping for common business datasets
    column_mapping = {
        'amount': ['amount', 'value', 'price', 'revenue', 'total_spent'],
        'transaction_count': ['transaction_count', 'num_orders', 'order_count', 'purchases'],
        'days_since_last_purchase': ['days_since_last_purchase', 'last_purchase_days_ago', 'days_since_order'],
        'customer_id': ['customer_id', 'user_id', 'client_id'],
        'age': ['age', 'customer_age'],
        'gender': ['gender', 'sex', 'customer_gender']
    }
    
    # Apply column mapping
    for target_col, possible_cols in column_mapping.items():
        if target_col not in df_copy.columns:
            for col in possible_cols:
                if col in df_copy.columns:
                    df_copy[target_col] = df_copy[col]
                    break
    
    # If the data is already pre-aggregated (one row per customer), use it directly
    if (
        'customer_id' in df_copy.columns and
        len(df_copy) == df_copy['customer_id'].nunique() and
        'transaction_count' in df_copy.columns and
        'total_amount' in df_copy.columns and
        'days_since_last_purchase' in df_copy.columns
    ):
        # Data is already aggregated at customer level - use as-is
        result_cols = ['customer_id', 'total_amount', 'transaction_count', 'days_since_last_purchase']
        demo_cols = [c for c in ['age', 'gender'] if c in df_copy.columns]
        return df_copy[result_cols + demo_cols].copy()
    
    # If we have transaction data, aggregate it
    elif 'amount' in df_copy.columns and 'customer_id' in df_copy.columns:
        customer_metrics = df_copy.groupby('customer_id').agg(
            total_amount=('amount', 'sum'),
            transaction_count=('amount', 'count')
        ).reset_index()
        if 'date' in df_copy.columns:
            df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
            last_purchase = df_copy.groupby('customer_id')['date'].max().reset_index()
            last_purchase.columns = ['customer_id', 'last_purchase']
            customer_metrics = customer_metrics.merge(last_purchase, on='customer_id', how='left')
            customer_metrics['days_since_last_purchase'] = (pd.Timestamp.now() - customer_metrics['last_purchase']).dt.days
        elif 'days_since_last_purchase' in df_copy.columns:
            days_data = df_copy[['customer_id', 'days_since_last_purchase']].drop_duplicates()
            customer_metrics = customer_metrics.merge(days_data, on='customer_id', how='left')
        demo_cols = [c for c in ['age', 'gender'] if c in df_copy.columns]
        if demo_cols:
            demo = df_copy[['customer_id'] + demo_cols].drop_duplicates('customer_id')
            customer_metrics = customer_metrics.merge(demo, on='customer_id', how='left')
        return customer_metrics
    else:
        # Fallback: ensure required columns exist
        required_cols = ['customer_id', 'total_amount', 'transaction_count', 'days_since_last_purchase']
        for col in required_cols:
            if col not in df_copy.columns:
                df_copy[col] = 0
        return df_copy

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Enhanced security functions
def hash_password(password):
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()

def check_password(password, hashed):
    """Check a password against a bcrypt hash"""
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    return True, ""

def sanitize_input(text):
    """Enhanced sanitize user input to prevent SQL injection"""
    if not text:
        return ""
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Comprehensive SQL keyword filtering
    sql_keywords = [
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'EXEC', 'UNION',
        'WHERE', 'FROM', 'JOIN', 'AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN', 'IS', 'NULL',
        'ORDER', 'BY', 'GROUP', 'HAVING', 'LIMIT', 'OFFSET', 'DISTINCT', 'AS', 'ON',
        'INNER', 'OUTER', 'LEFT', 'RIGHT', 'FULL', 'CROSS', 'NATURAL', 'CASE', 'WHEN',
        'THEN', 'ELSE', 'END', 'IF', 'EXISTS', 'ALL', 'ANY', 'SOME', 'INTO', 'VALUES',
        'SET', 'DEFAULT', 'AUTO_INCREMENT', 'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES',
        'CONSTRAINT', 'INDEX', 'UNIQUE', 'CHECK', 'TRIGGER', 'PROCEDURE', 'FUNCTION',
        'VIEW', 'TABLE', 'DATABASE', 'SCHEMA', 'USER', 'GRANT', 'REVOKE', 'COMMIT',
        'ROLLBACK', 'SAVEPOINT', 'TRANSACTION', 'LOCK', 'UNLOCK', 'SHOW', 'DESCRIBE',
        'EXPLAIN', 'ANALYZE', 'OPTIMIZE', 'REPAIR', 'CHECK', 'BACKUP', 'RESTORE',
        'LOAD', 'DUMP', 'IMPORT', 'EXPORT', 'FLUSH', 'RESET', 'KILL', 'SHUTDOWN',
        'START', 'STOP', 'RESTART', 'RELOAD', 'REFRESH', 'CACHE', 'BUFFER', 'POOL'
    ]
    
    # Check for SQL keywords (case insensitive)
    text_upper = text.upper()
    for keyword in sql_keywords:
        if keyword in text_upper:
            return ""  # Return empty string if SQL keyword found
    
    # Remove dangerous characters and patterns
    dangerous_patterns = [
        r'[;\'\"\\]',  # Semicolons, quotes, backslashes
        r'--',         # SQL comments
        r'/\*',        # SQL block comments
        r'\*/',        # SQL block comments
        r'xp_',        # SQL Server extended procedures
        r'sp_',        # SQL Server stored procedures
        r'@@',         # SQL Server system variables
        r'0x[0-9a-fA-F]+',  # Hex values
        r'UNION\s+ALL',      # UNION ALL attacks
        r'UNION\s+SELECT',   # UNION SELECT attacks
        r'OR\s+1\s*=\s*1',  # OR 1=1 attacks
        r'AND\s+1\s*=\s*1', # AND 1=1 attacks
        r'OR\s+\'1\'\s*=\s*\'1\'',  # OR '1'='1 attacks
        r'AND\s+\'1\'\s*=\s*\'1\'', # AND '1'='1 attacks
        r'DROP\s+TABLE',     # DROP TABLE attacks
        r'DELETE\s+FROM',    # DELETE FROM attacks
        r'UPDATE\s+SET',     # UPDATE SET attacks
        r'INSERT\s+INTO',    # INSERT INTO attacks
        r'CREATE\s+TABLE',   # CREATE TABLE attacks
        r'ALTER\s+TABLE',    # ALTER TABLE attacks
        r'EXEC\s*\(',        # EXEC() attacks
        r'EXECUTE\s*\(',     # EXECUTE() attacks
        r'WAITFOR\s+DELAY',  # Time-based attacks
        r'BENCHMARK\s*\(',   # MySQL benchmark attacks
        r'SLEEP\s*\(',       # MySQL sleep attacks
        r'PG_SLEEP\s*\(',    # PostgreSQL sleep attacks
        r'DBMS_PIPE\.RECEIVE_MESSAGE',  # Oracle time-based attacks
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return ""  # Return empty string if dangerous pattern found
    
    # Additional length and character restrictions
    if len(text) > 50:  # Limit input length
        return ""
    
    # Only allow alphanumeric characters, underscores, and hyphens for emails
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', text):
        return ""
    
    return text

# SQL-based user management
def load_users():
    """Load users from SQL database"""
    engine = get_db_connection()
    if not engine:
        return {}
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT email FROM users"))
            users = {}
            for row in result:
                users[row[0]] = {}  # Do not expose password hash
            return users
    except Exception as e:
        st.error(f"❌ Error loading users: {str(e)}")
        return {}

def save_users(users):
    """Save users to SQL database"""
    engine = get_db_connection()
    if not engine:
        return
    
    try:
        with engine.connect() as conn:
            for email, user_data in users.items():
                conn.execute(text("""
                    INSERT INTO users (email, password_hash) 
                    VALUES (:email, :password_hash)
                    ON DUPLICATE KEY UPDATE password_hash = :password_hash
                """), {
                    "email": email,
                    "password_hash": user_data['password']
                })
            conn.commit()
    except Exception as e:
        st.error(f"❌ Error saving users: {str(e)}")

def load_analysis_history(user_id=None):
    """Load analysis history from SQL database for a specific user (privacy-focused)"""
    engine = get_db_connection()
    if not engine:
        return {}
    
    try:
        with engine.connect() as conn:
            if user_id is not None:
                result = conn.execute(text("""
                    SELECT ah.analysis_id, ah.user_id, ah.timestamp, ah.total_customers, ah.high_risk_customers, ah.total_predicted_revenue, ah.avg_predicted_revenue, ah.analysis_summary
                    FROM analysis_history ah
                    WHERE ah.user_id = :user_id
                    ORDER BY ah.timestamp DESC
                """), {"user_id": user_id})
            else:
                # Fallback: return nothing if no user_id
                return {}
            history = []
            for row in result:
                history.append({
                    'analysis_id': row[0],
                    'timestamp': row[2].isoformat(),
                    'total_customers': row[3],
                    'high_risk_customers': row[4],
                    'total_predicted_revenue': float(row[5]) if row[5] is not None else 0.0,
                    'avg_predicted_revenue': float(row[6]) if row[6] is not None else 0.0,
                    'analysis_summary': row[7]
                })
            return history
    except Exception as e:
        st.error(f"❌ Error loading analysis history: {str(e)}")
        return {}

def get_analysis_details(analysis_id, user_id):
    """Get detailed analysis data for a specific analysis, only if it belongs to the user."""
    engine = get_db_connection()
    if not engine:
        return None, None, None
    
    try:
        with engine.connect() as conn:
            # Verify the analysis belongs to the user (user_id is email)
            result = conn.execute(text("""
                SELECT analysis_id FROM analysis_history WHERE analysis_id = :analysis_id AND user_id = :user_id
            """), {"analysis_id": analysis_id, "user_id": user_id})
            if not result.fetchone():
                return None, None, None
            # Get churn data
            churn_result = conn.execute(text("""
                SELECT customer_id, churn_probability, risk_level, total_amount, 
                       transaction_count
                FROM churn_data 
                WHERE analysis_id = :analysis_id
            """), {"analysis_id": analysis_id})
            churn_data = []
            for row in churn_result:
                churn_data.append({
                    'customer_id': row[0],
                    'churn_probability': float(row[1]),
                    'risk_level': row[2],
                    'total_amount': float(row[3]),
                    'transaction_count': row[4]
                })
            # Get revenue data
            revenue_result = conn.execute(text("""
                SELECT customer_id, predicted_revenue, total_revenue, revenue_segment, 
                       transaction_count, avg_transaction_value
                FROM revenue_data 
                WHERE analysis_id = :analysis_id
            """), {"analysis_id": analysis_id})
            revenue_data = []
            for row in revenue_result:
                revenue_data.append({
                    'customer_id': row[0],
                    'predicted_revenue': float(row[1]),
                    'total_revenue': float(row[2]),
                    'revenue_segment': row[3],
                    'transaction_count': row[4],
                    'avg_transaction_value': float(row[5])
                })
            # Get customer data
            customer_result = conn.execute(text("""
                SELECT customer_id, age, gender, country, total_spent, transaction_count
                FROM customer_data 
                WHERE analysis_id = :analysis_id
            """), {"analysis_id": analysis_id})
            customer_data = []
            for row in customer_result:
                customer_data.append({
                    'customer_id': row[0],
                    'age': row[1],
                    'gender': row[2],
                    'country': row[3],
                    'total_spent': float(row[4]),
                    'transaction_count': row[5]
                })
            return churn_data, revenue_data, customer_data
    except Exception as e:
        st.error(f"❌ Error loading analysis details: {str(e)}")
        return None, None, None

def cleanup_old_analyses(user_id=None):
    """Remove analyses and related data older than 3 days for the current user only"""
    engine = get_db_connection()
    if not engine:
        return {}
    try:
        with engine.connect() as conn:
            if user_id is not None:
                # Get old analysis IDs for this user
                old_analysis_ids = [row[0] for row in conn.execute(text("""
                    SELECT analysis_id FROM analysis_history 
                    WHERE user_id = :user_id AND timestamp < DATE_SUB(NOW(), INTERVAL 3 DAY)
                """), {"user_id": user_id})]
                # Delete related data for each old analysis
                for analysis_id in old_analysis_ids:
                    conn.execute(text("DELETE FROM customer_data WHERE analysis_id = :analysis_id"), {"analysis_id": analysis_id})
                    conn.execute(text("DELETE FROM churn_data WHERE analysis_id = :analysis_id"), {"analysis_id": analysis_id})
                    conn.execute(text("DELETE FROM revenue_data WHERE analysis_id = :analysis_id"), {"analysis_id": analysis_id})
                    conn.execute(text("DELETE FROM full_dashboard_data WHERE analysis_id = :analysis_id"), {"analysis_id": analysis_id})
                    conn.execute(text("DELETE FROM analysis_history WHERE analysis_id = :analysis_id"), {"analysis_id": analysis_id})
                conn.commit()
                return load_analysis_history(user_id)
            else:
                return {}
    except Exception as e:
        st.error(f"❌ Error during cleanup: {str(e)}")
        return {}

def delete_analysis(analysis_id, user_id):
    """Delete a specific analysis by ID, ensuring it belongs to the user (email)."""
    try:
        engine = get_db_connection()
        if not engine:
            return False, "Database connection failed"
        
        with engine.connect() as conn:
            # First verify the analysis belongs to the user (user_id is email)
            result = conn.execute(text("""
                SELECT analysis_id FROM analysis_history 
                WHERE analysis_id = :analysis_id AND user_id = :user_id
            """), {"analysis_id": analysis_id, "user_id": user_id})
            
            if not result.fetchone():
                return False, "Analysis not found or access denied"
            
            # The ON DELETE CASCADE will handle deletion of related data in child tables.
            conn.execute(text("""
                DELETE FROM analysis_history WHERE analysis_id = :analysis_id
            """), {"analysis_id": analysis_id})
            
            conn.commit()
            return True, "Analysis deleted successfully"
    except Exception as e:
        return False, f"Error deleting analysis: {str(e)}"

def save_current_analysis(user_id, churn_df, revenue_df, merged_df):
    """Save current analysis to SQL database using email as user_id."""
    engine = get_db_connection()
    if not engine:
        st.error("❌ Database connection failed during save.")
        return

    try:
        with engine.connect() as conn:
            # Start a transaction
            trans = conn.begin()
        try:
                # 1. Calculate summary statistics
            total_customers = len(churn_df)
            churn_prob_col = 'churn_probability_final' if 'churn_probability_final' in churn_df.columns else 'churn_probability'
            high_risk_customers = len(churn_df[churn_df[churn_prob_col] > 0.5])
            total_predicted_revenue = revenue_df['predicted_revenue'].sum() if 'predicted_revenue' in revenue_df.columns else 0
            avg_predicted_revenue = revenue_df['predicted_revenue'].mean() if 'predicted_revenue' in revenue_df.columns else 0
            analysis_summary = f"Analysis completed for {total_customers} customers."

                # 2. Insert into analysis_history
            analysis_result = conn.execute(text("""
                INSERT INTO analysis_history 
                (user_id, timestamp, total_customers, high_risk_customers, total_predicted_revenue, avg_predicted_revenue, analysis_summary)
                VALUES (:user_id, :timestamp, :total_customers, :high_risk_customers, :total_predicted_revenue, :avg_predicted_revenue, :analysis_summary)
            """), {
                    "user_id": user_id,  # This is the email
                "timestamp": datetime.now(),
                "total_customers": total_customers,
                "high_risk_customers": high_risk_customers,
                    "total_predicted_revenue": float(total_predicted_revenue),
                    "avg_predicted_revenue": float(avg_predicted_revenue),
                    "analysis_summary": analysis_summary
            })
            
                # 3. Get the new analysis_id
            analysis_id = analysis_result.lastrowid

                # 4. Prepare and save detailed data
                # Add user_id and analysis_id to DataFrames
                churn_df['user_id'] = user_id
                churn_df['analysis_id'] = analysis_id
                revenue_df['user_id'] = user_id
                revenue_df['analysis_id'] = analysis_id
                merged_df['user_id'] = user_id
                merged_df['analysis_id'] = analysis_id

                # --- FIX: Convert days_since_last_purchase to integer ---
                if 'days_since_last_purchase' in churn_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(churn_df['days_since_last_purchase']):
                        churn_df['days_since_last_purchase'] = (pd.Timestamp.now() - churn_df['days_since_last_purchase']).dt.days
                else:
                        churn_df['days_since_last_purchase'] = pd.to_numeric(churn_df['days_since_last_purchase'], errors='coerce').fillna(0).astype(int)

                # Define and filter columns for churn_data table
                churn_table_columns = [
                    'user_id', 'analysis_id', 'customer_id', 'churn_probability', 'risk_level',
                    'total_amount', 'transaction_count', 'days_since_last_purchase'
                ]
                if 'churn_probability_final' in churn_df.columns:
                    churn_df['churn_probability'] = churn_df['churn_probability_final']
                churn_df_to_save = churn_df[[col for col in churn_table_columns if col in churn_df.columns]].copy()
                churn_df_to_save.to_sql('churn_data', conn, if_exists='append', index=False)

                # Define and filter columns for revenue_data table
                revenue_table_columns = [
                    'user_id', 'analysis_id', 'customer_id', 'predicted_revenue', 
                    'total_revenue', 'revenue_segment'
                ]
                revenue_df_to_save = revenue_df[[col for col in revenue_table_columns if col in revenue_df.columns]].copy()
                revenue_df_to_save.to_sql('revenue_data', conn, if_exists='append', index=False)
                
                # Select only the columns that exist in customer_data table
                customer_data_cols = [
                    'user_id', 'analysis_id', 'customer_id', 'customer_name', 'age', 
                    'gender', 'country', 'total_spent', 'transaction_count', 
                    'first_purchase_date', 'last_purchase_date'
                ]
                customer_df_to_save = merged_df[[col for col in customer_data_cols if col in merged_df.columns]].copy()
                customer_df_to_save.to_sql('customer_data', conn, if_exists='append', index=False)

                # Commit the transaction
                trans.commit()
                st.success("💾 Analysis saved to your history!")
        except Exception as e:
                # Rollback in case of error
                trans.rollback()
                st.error(f"❌ Transaction failed, rolling back. Error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Error saving analysis: {str(e)}")

# --- Enhanced Authentication UI ---
def show_auth_ui():
    """Show enhanced authentication UI with security features, including Google OAuth login"""
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 30px;">
        <h1 style="color: white; margin: 0; font-size: 2.5em;">🤖 AutoInsights</h1>
        <p style="color: white; margin: 10px 0 0 0; font-size: 1.2em;">Smart Business Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("## 🔐 Welcome to AutoInsights")
    st.info("Please sign in or create an account to start your analysis.")

    if 'db_initialized' not in st.session_state:
        if init_database_schema():
            st.session_state.db_initialized = True
            st.success("✅ Database schema initialized successfully!")
        else:
            st.error("❌ Failed to initialize database schema. Please check your MySQL connection.")
            return

    # Google OAuth setup
    # Load Google OAuth credentials from environment variables (see .env)
    from utils.config import Config
    client_id = Config.GOOGLE_CLIENT_ID
    client_secret = Config.GOOGLE_CLIENT_SECRET
    redirect_uri = "http://localhost:8501"  # Change if deploying
    oauth2 = OAuth2Component(
        client_id,
        client_secret,
        "https://accounts.google.com/o/oauth2/v2/auth",
        "https://oauth2.googleapis.com/token",
        revoke_token_endpoint="https://oauth2.googleapis.com/revoke"
    )

    tab1, tab2 = st.tabs(["🔑 Sign In", "📝 Sign Up"])

    with tab1:
        st.subheader("Sign In")
        email = st.text_input("Email", key="login_email")
        email = sanitize_input(email)
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign In", key="login_btn", type="primary"):
            if not email or not password:
                st.error("❌ Please enter both email and password")
                return
            is_authenticated, login_msg = secure_login_check_email(email, password)
            if is_authenticated:
                st.session_state.authenticated = True
                st.session_state.user_id = email
                st.success("✅ Sign in successful!")
                st.rerun()
            else:
                st.error(f"❌ {login_msg}")
        # Forgot Password link
        forgot = st.button("Forgot Password?", key="forgot_password_btn", type="secondary")
        if forgot or st.session_state.get('show_forgot_password', False):
            st.session_state['show_forgot_password'] = True
            st.info("Enter your email and new password to reset your password.")
            reset_email = st.text_input("Email for Password Reset", key="reset_email")
            reset_email = sanitize_input(reset_email)
            new_password = st.text_input("New Password", type="password", key="reset_new_password")
            confirm_new_password = st.text_input("Confirm New Password", type="password", key="reset_confirm_new_password")
            if st.button("Reset Password", key="reset_password_btn", type="primary"):
                if not reset_email or not new_password or not confirm_new_password:
                    st.error("❌ Please fill in all fields")
                elif new_password != confirm_new_password:
                    st.error("❌ Passwords do not match")
                else:
                    valid, msg = validate_password(new_password)
                    if not valid:
                        st.error(f"❌ {msg}")
                    else:
                        engine = get_db_connection()
                        if not engine:
                            st.error("❌ Database connection failed")
                        else:
                            try:
                                with engine.connect() as conn:
                                    result = conn.execute(text("SELECT email FROM users WHERE email = :email"), {"email": reset_email})
                                    if not result.fetchone():
                                        st.error("❌ Email not found. Please check and try again.")
                                    else:
                                        conn.execute(text("UPDATE users SET password_hash = :password_hash WHERE email = :email"), {"password_hash": hash_password(new_password), "email": reset_email})
                                        conn.commit()
                                        st.success("✅ Password reset successfully! You can now sign in with your new password.")
                                        st.session_state['show_forgot_password'] = False
                            except Exception as e:
                                st.error(f"❌ Error resetting password: {str(e)}")
        # Google login button
        st.write("Or sign in with Google:")
        result = oauth2.authorize_button("Sign in with Google", redirect_uri, "openid email profile", key="google-signin")
        if result and "token" in result:
            userinfo = requests.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {result['token']['access_token']}"}
            ).json()
            engine = get_db_connection()
            if engine:
                with engine.connect() as conn:
                    res = conn.execute(text("SELECT email FROM users WHERE email = :email"), {"email": userinfo["email"]})
                    row = res.fetchone()
                    if row:
                        # Email exists, log the user in
                        st.session_state.authenticated = True
                        st.session_state.user_id = userinfo["email"]
                        st.success(f"Signed in as {userinfo['email']}")
                        st.rerun()
                    else:
                        # Email does not exist, create new user and log in
                        conn.execute(text("""
                            INSERT INTO users (email, password_hash, subscription, role)
                            VALUES (:email, :password_hash, :subscription, :role)
                        """), {
                            "email": userinfo["email"],
                            "password_hash": hash_password(os.urandom(16).hex()),
                            "subscription": "Free",
                            "role": "user"
                        })
                        conn.commit()
                        st.session_state.authenticated = True
                        st.session_state.user_id = userinfo["email"]
                        st.success(f"Signed in as {userinfo['email']}")
                        st.rerun()

    with tab2:
        st.subheader("Create Account")
        new_email = st.text_input("Email", key="register_email")
        new_email = sanitize_input(new_email)
        new_password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        st.info("""
        **Password Requirements:**
        - At least 8 characters long
        - Contains uppercase and lowercase letters
        - Contains at least one number
        """)
        if st.button("Create Account", key="register_btn", type="primary"):
            if not new_email or not new_password or not confirm_password:
                st.error("❌ Please fill in all fields")
                return
            if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", new_email):
                st.error("❌ Invalid email format")
                return
            engine = get_db_connection()
            if not engine:
                st.error("❌ Database connection failed")
                return
            try:
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT email FROM users WHERE email = :email"), {"email": new_email})
                    if result.fetchone():
                        st.error("❌ Email already in use, try another.")
                        return
                    subscription = 'Free'
                    role = 'user'
                    conn.execute(text("""
                        INSERT INTO users (email, password_hash, subscription, role) 
                        VALUES (:email, :password_hash, :subscription, :role)
                    """), {
                        "email": new_email,
                        "password_hash": hash_password(new_password),
                        "subscription": subscription,
                        "role": role
                    })
                    conn.commit()
                    st.success("✅ Account created successfully! Please sign in.")
                    st.info("You can now use your email and password to sign in.")
            except Exception as e:
                st.error(f"❌ Registration error: {str(e)}")
        # Google sign up button (same as sign in)
        st.write("Or sign up with Google:")
        result = oauth2.authorize_button("Sign up with Google", redirect_uri, "openid email profile", key="google-signup")
        if result and "token" in result:
            userinfo = requests.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {result['token']['access_token']}"}
            ).json()
            engine = get_db_connection()
            if engine:
                with engine.connect() as conn:
                    res = conn.execute(text("SELECT email FROM users WHERE email = :email"), {"email": userinfo["email"]})
                    row = res.fetchone()
                    if row:
                        st.error("❌ Email already in use, try another.")
                        return
                    else:
                        conn.execute(text("""
                            INSERT INTO users (email, password_hash, subscription, role)
                            VALUES (:email, :password_hash, :subscription, :role)
                        """), {
                            "email": userinfo["email"],
                            "password_hash": hash_password(os.urandom(16).hex()),
                            "subscription": "Free",
                            "role": "user"
                        })
                        conn.commit()
                        st.session_state.authenticated = True
                        st.session_state.user_id = userinfo["email"]
                        st.success(f"Signed up as {userinfo['email']}")
                        st.rerun()

# --- Secure login check using email ---
def secure_login_check_email(email, password):
       sanitized_email = sanitize_input(email)
       if not sanitized_email:
           log_security_event("SQL_INJECTION_ATTEMPT", email, "Email sanitization failed")
           return False, "Invalid email format"
       engine = get_db_connection()
       if not engine:
           return False, "Database connection failed"
       try:
           with engine.connect() as conn:
               result = conn.execute(text("""
                   SELECT email, password_hash, account_locked, login_attempts 
                   FROM users 
                   WHERE email = :email
               """), {"email": sanitized_email})
               row = result.fetchone()
               if not row:
                   log_security_event("LOGIN_FAILED", email, "User not found")
                   return False, "Invalid email or password"
               user_email, stored_hash, account_locked, login_attempts = row
               if account_locked:
                   log_security_event("ACCOUNT_LOCKED", email, "Attempted login to locked account")
                   return False, "Account is locked due to too many failed attempts"
               if check_password(password, stored_hash):
                   log_security_event("LOGIN_SUCCESS", email, "Successful login")
                   return True, "Login successful"
               else:
                   log_security_event("LOGIN_FAILED", email, "Invalid password")
                   return False, "Invalid email or password"
       except Exception as e:
           return False, f"Database error: {str(e)}"

def get_user_subscription_and_role_by_email(email):
    """Get user subscription and role from database using email"""
    user_subscription = 'Free'
    user_role = 'user'
    try:
        engine = get_db_connection()
        if engine:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT subscription, role FROM users WHERE email = :email
                """), {"email": email})
                row = result.fetchone()
                if row:
                    user_subscription, user_role = row
    except Exception:
        pass
    return user_subscription, user_role

def check_daily_analysis_limit_by_email(email):
    """Check if user has reached daily analysis limit (2 for free users) using email"""
    try:
        engine = get_db_connection()
        if engine:
            with engine.connect() as conn:
                # Get user ID
                result = conn.execute(text("""
                    SELECT id FROM users WHERE email = :email
                """), {"email": email})
                user_row = result.fetchone()
                if not user_row:
                    return False, "User not found"
                
                user_id = user_row[0]
                
                # Get today's analysis count
                today = datetime.now().date()
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM analysis_history 
                    WHERE user_id = :user_id AND DATE(timestamp) = :today
                """), {"user_id": user_id, "today": today})
                
                today_count = result.fetchone()[0]
                
                # Get user subscription to determine limit
                user_subscription, user_role = get_user_subscription_and_role_by_email(email)
                
                # Set limits based on subscription
                if user_role == 'developer':
                    return True, "Developer has unlimited access"
                elif user_subscription == 'Premium':
                    return True, "Premium has unlimited access"
                else:  # Free user
                    limit = 2
                    if today_count >= limit:
                        return False, f"Daily limit reached ({limit} analyses per day). Upgrade to Premium for unlimited analyses!"
                    else:
                        remaining = limit - today_count
                        return True, f"Analyses remaining today: {remaining}"
    except Exception as e:
        return False, f"Error checking limit: {str(e)}"
    
    return False, "Database connection failed" 

# Show analysis history
def show_analysis_history():
    """Show user's analysis history from SQL database"""
    st.subheader("📊 Your Analysis History")
    st.info("Your analyses are stored for 3 days. Click on any analysis to view details.")
    
    engine = get_db_connection()
    if not engine:
        st.error("❌ Database connection failed")
        return
    
    try:
        with engine.connect() as conn:
            # Get user ID
            result = conn.execute(text("""
                SELECT email FROM users WHERE email = :email
            """), {"email": st.session_state.user_id})
            user_row = result.fetchone()
            if not user_row:
                st.error("❌ User not found")
                return
            user_id = user_row[0]
            
            # Get user's analysis history (using existing table structure)
            result = conn.execute(text("""
                SELECT ah.analysis_id, ah.user_id, ah.timestamp, ah.total_customers, ah.high_risk_customers, ah.total_predicted_revenue, ah.avg_predicted_revenue, ah.analysis_summary
                FROM analysis_history ah
                WHERE ah.user_id = :user_id
                ORDER BY ah.timestamp DESC
            """), {"user_id": user_id})
            
            analyses = []
            for row in result:
                analyses.append({
                    'analysis_id': row[0],
                    'timestamp': row[2].isoformat(),
                    'total_customers': row[3],
                    'high_risk_customers': row[4],
                    'total_predicted_revenue': float(row[5]) if row[5] is not None else 0.0,
                    'avg_predicted_revenue': float(row[6]) if row[6] is not None else 0.0,
                    'analysis_summary': row[7]
                })
            
            if not analyses:
                st.info("No previous analyses found. Run your first analysis to see it here!")
                return
            
            # Show analyses in reverse chronological order
            for i, analysis in enumerate(analyses):
                with st.expander(f"📈 Analysis from {analysis['timestamp'][:19]}"):
                    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                    with col1:
                        st.metric("Customers", analysis['total_customers'])
                    with col2:
                        st.metric("⚠️ High Risk", analysis['high_risk_customers'])
                    with col3:
                        st.metric("💰 Revenue", f"${analysis['total_predicted_revenue']:.2f}")
                    with col4:
                        st.metric("📊 Avg Rev", f"${analysis['avg_predicted_revenue']:.2f}")
                    with col5:
                        st.metric("🎯 High Risk", f"{analysis['high_risk_customers']} high-risk customers")
                    with col6:
                        st.metric("💵 Avg Revenue", f"{st.session_state.get('currency_symbol', '$')}{analysis['avg_predicted_revenue']:.2f}")
                    with col7:
                        st.metric("📝 Analysis Summary", analysis['analysis_summary'])
                    
                    # Show detailed data
                    if st.button(f"View Details", key=f"view_{i}"):
                        st.session_state.current_analysis_id = analysis['analysis_id']
                        st.rerun()
                    
                    # Action buttons
                    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                    
                    # Create unique key using analysis_id and timestamp
                    unique_key = f"{analysis['analysis_id']}_{analysis['timestamp'].replace(':', '_').replace('.', '_')}"
                    
                    with col1:
                        if st.button(f"📊 View Details", key=f"view_{unique_key}"):
                            st.session_state.current_analysis_id = analysis['analysis_id']
                            st.rerun()
                    
                    with col2:
                        if st.button(f"📥 Download", key=f"download_{unique_key}"):
                            # Get detailed data for download
                            churn_data, revenue_data, customer_data = get_analysis_details(analysis['analysis_id'], st.session_state.user_id)
                            if churn_data and revenue_data:
                                churn_csv = pd.DataFrame(churn_data).to_csv(index=False)
                                revenue_csv = pd.DataFrame(revenue_data).to_csv(index=False)
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.download_button(
                                        "📊 Download Churn Data",
                                        churn_csv,
                                        f"churn_analysis_{analysis['timestamp'][:10]}.csv",
                                        "text/csv"
                                    )
                                with col2:
                                    st.download_button(
                                        "💰 Download Revenue Data",
                                        revenue_csv,
                                        f"revenue_analysis_{analysis['timestamp'][:10]}.csv",
                                        "text/csv"
                                    )
                    
                    with col3:
                        # Add confirmation for deletion
                        delete_key = f"confirm_delete_{unique_key}"
                        if delete_key not in st.session_state:
                            st.session_state[delete_key] = False
                        
                        if not st.session_state[delete_key]:
                            if st.button(f"🗑️ Delete", key=f"delete_{unique_key}", type="secondary"):
                                st.session_state[delete_key] = True
                                st.rerun()
                        else:
                            st.warning("⚠️ Are you sure you want to delete this analysis?")
                            col_confirm1, col_confirm2 = st.columns(2)
                            with col_confirm1:
                                if st.button("✅ Yes, Delete", key=f"yes_delete_{unique_key}", type="primary"):
                                    # Get user ID for deletion
                                    result = conn.execute(text("""
                                        SELECT id FROM users WHERE email = :email
                                    """), {"email": st.session_state.user_id})
                                    user_row = result.fetchone()
                                    if user_row:
                                        db_user_id = user_row[0]
                                        success, message = delete_analysis(analysis['analysis_id'], db_user_id)
                                        if success:
                                            st.success("✅ Analysis deleted successfully!")
                                            st.session_state[delete_key] = False
                                            st.rerun()
                                        else:
                                            st.error(f"❌ {message}")
                                            st.session_state[delete_key] = False
                                    else:
                                        st.error("❌ User not found")
                                        st.session_state[delete_key] = False
                            with col_confirm2:
                                if st.button("❌ Cancel", key=f"cancel_delete_{unique_key}"):
                                    st.session_state[delete_key] = False
                                    st.rerun()
    except Exception as e:
        st.error(f"❌ Error loading analysis history: {str(e)}")

# Custom CSS
def local_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .stFileUploader>div>div>div>div {
        color: #4CAF50;
    }
    .css-1aumxhk {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .column-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .required {
        border-left: 4px solid #ff5252;
    }
    .recommended {
        border-left: 4px solid #ffd740;
    }
    .optional {
        border-left: 4px solid #66bb6a;
    }
    /* Sidebar width - 4.5cm */
    .css-1d391kg {
        width: 4.5cm !important;
        min-width: 4.5cm !important;
        max-width: 4.5cm !important;
    }
    /* Ensure sidebar content fits within 4.5cm */
    .css-1d391kg .css-1d391kg {
        width: 100% !important;
        max-width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'df' not in st.session_state:
    st.session_state.df = None

# Initialize database and cleanup old analyses
if 'db_initialized' not in st.session_state:
    if init_database_schema():
        st.session_state.db_initialized = True
        # Ensure developer privileges for the specified user immediately after DB init
        engine = get_db_connection()
        if engine:
            with engine.connect() as conn:
                conn.execute(text("""
                    UPDATE users SET role = 'developer', subscription = 'Premium' WHERE email = :email
                """), {"email": "sah.nai.rt22@dypatil.edu"})
                conn.commit()
    else:
        st.error("❌ Failed to initialize database schema. Please check your MySQL connection.")
        st.stop()

cleanup_old_analyses()

# Check authentication
if not st.session_state.authenticated:
    show_auth_ui()
    st.stop()

# Get user subscription and role for feature restrictions
user_subscription, user_role = get_user_subscription_and_role_by_email(st.session_state.user_id)

# Main app interface
# Remove the existing hamburger menu dropdown and move functionality to sidebar

# Header with title
st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 30px;">
    <h1 style="color: white; margin: 0; font-size: 2.5em;">🤖 AutoInsights</h1>
    <p style="color: white; margin: 10px 0 0 0; font-size: 1.2em;">Smart Business Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

st.title("📊 AutoInsights Dashboard")

# Security notice
st.info("🔒 **Security Notice:** This application uses secure authentication with 3-attempt login limits and automatic account lockout for failed attempts.")

# User info and logout
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    **Transform your customer data into actionable business insights!**

    Upload your customer and transaction data, and we'll automatically:
    - 🎯 **Predict which customers might leave** (churn prediction)
    - 💰 **Forecast future revenue** from each customer
    - 📊 **Identify your best customers** for targeted marketing
    - 🚀 **Generate actionable recommendations** for your business

    *No technical knowledge required - just upload your data and get instant insights!*
    """)

# Show daily limit status for free users
if user_subscription == 'Free' and user_role != 'developer':
    can_analyze, limit_message = check_daily_analysis_limit_by_email(st.session_state.user_id)
    if can_analyze:
        st.success(f"📊 {limit_message}")
    else:
        st.error(f"❌ {limit_message}")
        st.info("💡 **Upgrade to Premium for unlimited daily analyses!**")

# Remove the right column content (welcome message and sign out button)

# Show analysis history
show_analysis_history()

# Show detailed view of selected analysis
if 'current_analysis' in st.session_state:
    st.markdown("---")
    st.subheader("📊 Historical Analysis Details")
    
    analysis = st.session_state.current_analysis
    
    # Show summary metrics
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.metric("📊 Total Customers", analysis['total_customers'])
    with col2:
        st.metric("🚨 High Risk Customers", analysis['high_risk_customers'])
    with col3:
        st.metric("💰 Total Predicted Revenue", f"${analysis['total_predicted_revenue']:.2f}")
    with col4:
        st.metric("💵 Avg Predicted Revenue", f"${analysis['avg_predicted_revenue']:.2f}")
    with col5:
        st.metric("🎯 High Risk", f"{analysis['high_risk_customers']} high-risk customers")
    with col6:
        st.metric("💵 Avg Revenue", f"{st.session_state.get('currency_symbol', '$')}{analysis['avg_predicted_revenue']:.2f}")
    with col7:
        st.metric("📝 Analysis Summary", analysis['analysis_summary'])
    
    # Show data tables
    tab1, tab2 = st.tabs(["📊 Churn Data", "💰 Revenue Data"])
    
    with tab1:
        churn_df = pd.DataFrame(analysis['churn_data'])
        st.dataframe(churn_df.head(10), use_container_width=True)
        st.caption(f"Showing first 10 rows of {len(churn_df)} customers")
    
    with tab2:
        revenue_df = pd.DataFrame(analysis['revenue_data'])
        st.dataframe(revenue_df.head(10), use_container_width=True)
        st.caption(f"Showing first 10 rows of {len(revenue_df)} customers")
    
    if st.button("❌ Close Details"):
        del st.session_state.current_analysis
        st.rerun()

# Enhanced column detection
def detect_column_type(col_name, sample_data):
    """Intelligently detect column type based on name and data"""
    col_name = str(col_name).lower()
    
    # ID detection - more comprehensive
    if any(kw in col_name for kw in ['id', 'customer', 'client', 'user', 'member', 'account']):
        # Check if it looks like an ID (unique values, alphanumeric)
        if sample_data.nunique() > len(sample_data) * 0.8:  # High uniqueness
            return 'customer_id'
    
    # Date detection - more comprehensive
    if any(kw in col_name for kw in ['date', 'time', 'created', 'purchased', 'transaction', 'order', 'billing', 'payment']):
        try:
            # Try to parse as datetime
            pd.to_datetime(sample_data, errors='raise')
            return 'date'
        except:
            pass
    
    # Amount detection - more comprehensive
    if any(kw in col_name for kw in ['amount', 'price', 'value', 'cost', 'revenue', 'total', 'sum', 'payment', 'charge', 'fee', 'price', 'cost']):
        try:
            # Try to convert to numeric
            numeric_data = pd.to_numeric(sample_data, errors='coerce')
            if not numeric_data.isna().all():  # At least some numeric values
                return 'amount'
        except:
            pass
    
    # Customer info detection
    if any(kw in col_name for kw in ['name', 'first', 'last', 'full', 'customer', 'client']):
        return 'customer_name'
    
    # Location detection
    if any(kw in col_name for kw in ['country', 'city', 'state', 'region', 'location', 'address', 'zip', 'postal']):
        return 'location'
    
    # Category detection
    if any(kw in col_name for kw in ['category', 'type', 'class', 'group', 'product', 'item', 'service']):
        return 'category'
    
    # Age detection
    if any(kw in col_name for kw in ['age', 'dob', 'birth', 'year']):
        try:
            ages = pd.to_numeric(sample_data, errors='coerce')
            if not ages.isna().all() and ages.between(15, 100).any():
                return 'age'
        except:
            pass
    
    # Gender detection
    if any(kw in col_name for kw in ['gender', 'sex']):
        gender_values = sample_data.astype(str).str.lower()
        if gender_values.isin(['m', 'f', 'male', 'female', 'm', 'f']).any():
            return 'gender'
    
    # Enhanced data type inference
    try:
        # Check if numeric (could be amount)
        if pd.api.types.is_numeric_dtype(sample_data):
            # If it's numeric and has reasonable values, likely amount
            if sample_data.dtype in ['int64', 'float64']:
                if sample_data.min() > 0 and sample_data.max() < 1000000:  # Reasonable amount range
                    return 'amount'
                else:
                    return 'numeric'
        
        # Check if datetime
        if pd.api.types.is_datetime64_any_dtype(sample_data):
            return 'date'
        
        # Check if categorical (low cardinality)
        if sample_data.nunique() < len(sample_data) * 0.3:
            return 'category'
        
        # Check if it looks like an ID (high cardinality, alphanumeric)
        if sample_data.nunique() > len(sample_data) * 0.8:
            return 'customer_id'
            
    except:
        pass
    
    return 'unknown'

def analyze_columns(df):
    """Analyze all columns in a dataframe"""
    results = {}
    for col in df.columns:
        sample = df[col].dropna().head(100)
        results[col] = detect_column_type(col, sample)
    return results

# Column mapping UI
def column_mapping_ui(df, title):
    """Create user-friendly UI for mapping columns"""
    st.subheader(f"📋 Step 2: Map Your Data Columns")
    st.info("""
    **Don't worry if you're not sure!** We'll automatically suggest the best matches for your data.
    Just review and confirm the selections below.
    """)
    
    # Analyze columns
    col_analysis = analyze_columns(df)
    
    # Required columns with simple descriptions
    required_columns = {
        'customer_id': {'priority': 'required', 'description': 'Unique customer identifier (like customer number or ID)'},
        'date': {'priority': 'required', 'description': 'When the transaction happened'},
        'amount': {'priority': 'required', 'description': 'How much money was spent'},
        'customer_name': {'priority': 'recommended', 'description': 'Customer name (optional)'},
        'age': {'priority': 'recommended', 'description': 'Customer age (optional)'},
        'gender': {'priority': 'recommended', 'description': 'Customer gender (optional)'},
        'location': {'priority': 'recommended', 'description': 'Customer location/country (optional)'},
        'category': {'priority': 'optional', 'description': 'Product category (optional)'},
    }
    
    # Create mapping dictionary
    mapping = {}
    
    # Show data preview first
    st.markdown("### 📊 Your Data Preview")
    st.dataframe(df.head(5), use_container_width=True)
    st.caption(f"Showing first 5 rows of your {len(df)} total records")
    
    st.markdown("### 🔗 Column Mapping")
    st.info("We've automatically detected the best matches for your data. Please review:")
    
    for col_type, info in required_columns.items():
        # Find best match
        possible_matches = [c for c, t in col_analysis.items() if t == col_type]
        
        # Auto-select the best match
        auto_selected = None
        if possible_matches:
            auto_selected = possible_matches[0]
        else:
            # Smart defaults
            if col_type == 'customer_id':
                for col in df.columns:
                    if df[col].nunique() > len(df) * 0.8:
                        auto_selected = col
                        break
            elif col_type == 'date':
                for col in df.columns:
                    try:
                        pd.to_datetime(df[col], errors='raise')
                        auto_selected = col
                        break
                    except:
                        continue
            elif col_type == 'amount':
                for col in df.columns:
                    try:
                        numeric_data = pd.to_numeric(df[col], errors='coerce')
                        if not numeric_data.isna().all() and numeric_data.min() > 0:
                            auto_selected = col
                            break
                    except:
                        continue
        
        # Create simple card
        with st.container():
            # Color code based on priority
            if info['priority'] == 'required':
                st.markdown("**🔴 Required:** " + col_type.replace('_', ' ').title())
            elif info['priority'] == 'recommended':
                st.markdown("**🟡 Recommended:** " + col_type.replace('_', ' ').title())
            else:
                st.markdown("**🟢 Optional:** " + col_type.replace('_', ' ').title())
            
            st.caption(info['description'])
            
            # Simple select box
            options = [f"❌ Not available"] + list(df.columns)
            
            # Find the index of auto-selected option
            default_index = 0
            if auto_selected:
                try:
                    default_index = options.index(auto_selected)
                except ValueError:
                    default_index = 0
            
            selected = st.selectbox(
                f"Select column for {col_type.replace('_', ' ')}:",
                options,
                index=default_index,
                key=f"{title}_{col_type}"
            )
            
            if selected != "❌ Not available":
                mapping[col_type] = selected
                if auto_selected and selected == auto_selected:
                    st.success(f"✅ Perfect! Auto-detected: **{selected}**")
                else:
                    st.success(f"✅ Selected: **{selected}**")
            else:
                mapping[col_type] = None
                if info['priority'] == 'required':
                    st.error(f"⚠️ **{col_type.replace('_', ' ').title()}** is required! Please select a column.")
                else:
                    st.info(f"ℹ️ **{col_type.replace('_', ' ').title()}** is optional - you can skip this.")
        
        st.markdown("---")
    
    return mapping

# Main processing function
def process_data(customers_df, transactions_df, customer_mapping, transaction_mapping):
    """Process data with flexible column mapping"""
    # Rename columns based on mapping
    def rename_columns(df, mapping):
        rename_dict = {}
        for new_col, old_col in mapping.items():
            if old_col:
                rename_dict[old_col] = new_col
        return df.rename(columns=rename_dict)
    
    # Apply mapping
    customers_mapped = rename_columns(customers_df, customer_mapping)
    transactions_mapped = rename_columns(transactions_df, transaction_mapping)
    
    # Fill in missing required columns with placeholders
    required_cols = ['customer_id', 'date', 'amount']
    
    # Handle missing customer_id
    if 'customer_id' not in customers_mapped.columns:
        if 'customer_id' in transactions_mapped.columns:
            # Create customer IDs from transactions
            customer_ids = transactions_mapped['customer_id'].unique()
            customers_mapped['customer_id'] = customer_ids[:len(customers_mapped)]
        else:
            # Generate sequential IDs
            customers_mapped['customer_id'] = range(1, len(customers_mapped) + 1)
    
    # Handle missing date
    if 'date' not in transactions_mapped.columns:
        transactions_mapped['date'] = pd.date_range(
            start='2020-01-01', 
            periods=len(transactions_mapped),
            freq='D'
        )
    
    # Handle missing amount
    if 'amount' not in transactions_mapped.columns:
        # Try to find a numeric column
        numeric_cols = [col for col in transactions_mapped.columns 
                       if pd.api.types.is_numeric_dtype(transactions_mapped[col])]
        
        if numeric_cols:
            transactions_mapped['amount'] = transactions_mapped[numeric_cols[0]]
        else:
            # Generate random amounts
            transactions_mapped['amount'] = np.random.uniform(10, 500, len(transactions_mapped))
    
    # Add any missing recommended columns with defaults
    for col in ['customer_name', 'age', 'gender', 'location']:
        if col not in customers_mapped.columns:
            customers_mapped[col] = None
    
    if 'category' not in transactions_mapped.columns:
        transactions_mapped['category'] = 'General'
    
    # Ensure data types
    if 'date' in transactions_mapped.columns:
        transactions_mapped['date'] = pd.to_datetime(transactions_mapped['date'], errors='coerce')
    
    if 'amount' in transactions_mapped.columns:
        transactions_mapped['amount'] = pd.to_numeric(transactions_mapped['amount'], errors='coerce')
    
    return customers_mapped, transactions_mapped

# --- Currency detection utility ---
def detect_currency(df):
    # Try to detect currency from column names or sample values
    currency_symbols = ['$', '₹', '€', '£', '¥', '₩', '₽', '₺', '₪', '₫', '฿', '₴', '₦', '₲', '₵', '₡', '₱', '₲', '₸', '₭', '₮', '₯', '₠', '₢', '₣', '₤', '₧', '₨', '₩', '₪', '₫', '₭', '₮', '₯', '₠', '₢', '₣', '₤', '₧', '₨']
    # Check column names
    for col in df.columns:
        for symbol in currency_symbols:
            if symbol in col:
                return symbol
    # Check sample values in amount column
    if 'amount' in df.columns:
        sample = df['amount'].astype(str).head(20)
        for val in sample:
            for symbol in currency_symbols:
                if symbol in val:
                    return symbol
    # Default to $
    return '$'

# File upload section
st.markdown("---")
st.markdown("## 📁 Step 1: Upload Your Data")

st.info("""
**What data do you need?**
- Customer information (names, IDs, etc.)
- Transaction data (dates, amounts, etc.)
- You can upload either one file with everything, or separate files
""")

upload_option = st.radio(
    "Choose your upload method:",
    ["📄 Single File (Recommended)", "📁 Separate Files"],
    help="Single file is easier - just upload one CSV with all your data!"
)

# --- Single File Upload Flow ---
if upload_option == "📄 Single File (Recommended)":
    st.markdown("### 📄 Upload Your Data File")
    st.info("""
    **Supported formats:** CSV, Excel (.xlsx, .xls)
    
    **What should your file contain?**
    - Customer IDs or names
    - Transaction dates
    - Amounts spent
    - Any other customer info (age, location, etc.)
    """)
    
    uploaded_file = st.file_uploader(
        "Choose your data file:",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your customer and transaction data"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success(f"✅ Successfully loaded {len(df)} records with {len(df.columns)} columns!")
            # ... existing code ...
            mapping = column_mapping_ui(df, "Single File")
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                # --- Clear session state for new analysis ---
                for key in ["customers_df", "transactions_df", "merged_df", "churn_df", "revenue_df"]:
                    if key in st.session_state:
                        del st.session_state[key]
                # --- Set fixed random seed for reproducibility ---
                import numpy as np
                np.random.seed(42)
                merged_df = None
                churn_df = None
                revenue_df = None
                try:
                    user_id = st.session_state.user_id
                    customers_df, transactions_df, merged_df = process_single_dataset(df, user_id, None)
                    churn_features = prepare_churn_features(merged_df)
                    churn_df = generate_churn_data(churn_features)
                    revenue_features = prepare_churn_features(merged_df)
                    revenue_df = generate_revenue_data(revenue_features)
                    st.session_state.customers_df = customers_df
                    st.session_state.transactions_df = transactions_df
                    st.session_state.merged_df = merged_df
                    st.session_state.churn_df = churn_df
                    st.session_state.revenue_df = revenue_df
                    engine = get_db_connection()
                    with engine.connect() as conn:
                        result = conn.execute(text("SELECT email FROM users WHERE email = :email"), {"email": st.session_state.user_id})
                        user_row = result.fetchone()
                        if not user_row:
                            st.error("❌ User not found in database")
                            st.stop()
                        db_user_id = user_row[0]
                        conn.execute(text("""
                            INSERT INTO analysis_history (user_id, timestamp)
                            VALUES (:user_id, :timestamp)
                        """), {
                            "user_id": db_user_id,
                            "timestamp": datetime.now()
                        })
                        analysis_id = conn.execute(text("SELECT LAST_INSERT_ID()")).fetchone()[0]
                        conn.commit()  # <-- Ensure the analysis_id is committed and visible
                    churn_df['analysis_id'] = analysis_id
                    revenue_df['analysis_id'] = analysis_id
                    print('DEBUG: churn_df columns before writing CSV:', churn_df.columns.tolist())
                    print('DEBUG: churn_df head before writing CSV:', churn_df.head())
                    output_dir = Path(__file__).parent / 'output'
                    churn_path = output_dir / 'churn_data.csv'
                    revenue_path = output_dir / 'revenue_data.csv'
                    print('DEBUG: Writing churn data to:', churn_path.resolve())
                    churn_df.to_csv(churn_path, index=False)
                    print('DEBUG: Writing revenue data to:', revenue_path.resolve())
                    revenue_df.to_csv(revenue_path, index=False)
                    upload_success, upload_msg = upload_to_sql(st.session_state.user_id, analysis_id, churn_df, revenue_df)
                    if upload_success:
                        st.success(upload_msg)
                    else:
                        st.error(upload_msg)
                    cleanup_old_analyses(db_user_id)
                except Exception as e:
                    st.error(f"❌ Error during processing: {e}")
                if merged_df is not None and churn_df is not None and revenue_df is not None:
                    save_current_analysis(st.session_state.user_id, churn_df, revenue_df, merged_df)
                    st.success("💾 Analysis saved to your history!")
                else:
                    st.error("❌ Analysis could not be completed due to a processing error.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
# --- Separate File Upload Flow ---
else:
    st.markdown("### 📁 Upload Separate Files")
    st.info("Upload your customer and transaction data as separate files.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**👥 Customer Data**")
        customer_file = st.file_uploader(
            "Upload customer file:",
            type=['csv', 'xlsx', 'xls'],
            key="customer_file"
        )
    
    with col2:
        st.markdown("**💰 Transaction Data**")
        transaction_file = st.file_uploader(
            "Upload transaction file:",
            type=['csv', 'xlsx', 'xls'],
            key="transaction_file"
        )
    
    if customer_file and transaction_file:
        try:
            if customer_file.name.endswith('.csv'):
                customers_df = pd.read_csv(customer_file)
            else:
                customers_df = pd.read_excel(customer_file)
            if transaction_file.name.endswith('.csv'):
                transactions_df = pd.read_csv(transaction_file)
            else:
                transactions_df = pd.read_excel(transaction_file)
            st.success(f"✅ Loaded {len(customers_df)} customers and {len(transactions_df)} transactions!")
            # ... existing code ...
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 👥 Customer Data Mapping")
                customer_mapping = column_mapping_ui(customers_df, "Customers")
            with col2:
                st.markdown("### 💰 Transaction Data Mapping")
                transaction_mapping = column_mapping_ui(transactions_df, "Transactions")
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                # --- Clear session state for new analysis ---
                for key in ["customers_df", "transactions_df", "merged_df", "churn_df", "revenue_df"]:
                    if key in st.session_state:
                        del st.session_state[key]
                # --- Set fixed random seed for reproducibility ---
                import numpy as np
                np.random.seed(42)
                merged_df = None
                churn_df = None
                revenue_df = None
                if not customer_mapping.get('customer_id') or not transaction_mapping.get('date') or not transaction_mapping.get('amount'):
                    st.error("❌ Please map the required columns before starting analysis.")
                else:
                    can_analyze, limit_message = check_daily_analysis_limit_by_email(st.session_state.user_id)
                    if not can_analyze:
                        st.error(f"❌ {limit_message}")
                        st.info("💡 **Upgrade to Premium (₹500/month) for unlimited daily analyses!**")
                        st.markdown("""
                        **Premium Benefits:**
                        - 🚀 Unlimited daily analyses
                        - 📊 Advanced analytics and detailed insights
                        - 💰 Comprehensive revenue forecasting
                        - 🎯 Advanced churn prediction models
                        """)
                        if st.button("🆙 Upgrade Now (₹500/month)", type="primary", key="upgrade_limit_2"):
                            st.info("Contact support@autobi.com to complete your upgrade!")
                        st.stop()
                    else:
                        st.info(f"ℹ️ {limit_message}")
                    with st.spinner("🔄 Processing your data..."):
                        try:
                            customers_processed, transactions_processed = process_data(
                                customers_df, transactions_df, customer_mapping, transaction_mapping
                            )
                            merged_df = pd.merge(
                                transactions_processed,
                                customers_processed,
                                on='customer_id',
                                how='left'
                            )
                            st.session_state.merged_df = merged_df
                            st.success("✅ Data processed successfully!")
                            churn_features = prepare_churn_features(merged_df)
                            churn_df = generate_churn_data(churn_features)
                            revenue_features = prepare_churn_features(merged_df)
                            revenue_df = generate_revenue_data(revenue_features)
                            st.session_state.churn_df = churn_df
                            st.session_state.revenue_df = revenue_df
                            engine = get_db_connection()
                            with engine.connect() as conn:
                                result = conn.execute(text("SELECT email FROM users WHERE email = :email"), {"email": st.session_state.user_id})
                                user_row = result.fetchone()
                                if not user_row:
                                    st.error("❌ User not found in database")
                                    st.stop()
                                db_user_id = user_row[0]
                                conn.execute(text("""
                                    INSERT INTO analysis_history (user_id, timestamp)
                                    VALUES (:user_id, :timestamp)
                                """), {
                                    "user_id": db_user_id,
                                    "timestamp": datetime.now()
                                })
                                analysis_id = conn.execute(text("SELECT LAST_INSERT_ID()")).fetchone()[0]
                                conn.commit()  # <-- Ensure the analysis_id is committed and visible
                            churn_df['analysis_id'] = analysis_id
                            revenue_df['analysis_id'] = analysis_id
                            print('DEBUG: churn_df columns before writing CSV:', churn_df.columns.tolist())
                            print('DEBUG: churn_df head before writing CSV:', churn_df.head())
                            output_dir = Path(__file__).parent / 'output'
                            churn_path = output_dir / 'churn_data.csv'
                            revenue_path = output_dir / 'revenue_data.csv'
                            print('DEBUG: Writing churn data to:', churn_path.resolve())
                            churn_df.to_csv(churn_path, index=False)
                            print('DEBUG: Writing revenue data to:', revenue_path.resolve())
                            revenue_df.to_csv(revenue_path, index=False)
                            upload_success, upload_msg = upload_to_sql(st.session_state.user_id, analysis_id, churn_df, revenue_df)
                            if upload_success:
                                st.success(upload_msg)
                            else:
                                st.error(upload_msg)
                            cleanup_old_analyses(db_user_id)
                        except Exception as e:
                            st.error(f"❌ Processing error: {str(e)}")
                if merged_df is not None and churn_df is not None and revenue_df is not None:
                    save_current_analysis(st.session_state.user_id, churn_df, revenue_df, merged_df)
                    st.success("💾 Analysis saved to your history!")
                else:
                    st.error("❌ Analysis could not be completed due to a processing error.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Results section
if 'churn_df' in st.session_state and 'revenue_df' in st.session_state and 'merged_df' in st.session_state:
    st.markdown("---")
    st.header("🎉 Your Analysis Results")
    
    st.success("""
    **Great job!** Your data has been analyzed and we've found some valuable insights for your business.
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Customer Risk", "💰 Revenue Insights", "🤖 Smart Recommendations"])
    
    with tab1:
        st.subheader("📊 Your Business Overview")
        
        # Simple metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_customers = len(st.session_state.merged_df)
            st.metric("👥 Total Customers", f"{total_customers:,}")
        with col2:
            total_revenue = st.session_state.merged_df['amount'].sum()
            st.metric("💰 Total Revenue", f"{st.session_state.get('currency_symbol', '$')}{total_revenue:,.2f}")
        with col3:
            avg_transaction = st.session_state.merged_df['amount'].mean()
            st.metric("💳 Avg Transaction", f"{st.session_state.get('currency_symbol', '$')}{avg_transaction:.2f}")
        
        # Data summary
        st.markdown("### 📈 Your Data Summary")
        st.dataframe(st.session_state.merged_df.head(10), use_container_width=True)
        st.caption("Showing first 10 transactions from your data")
        
        # Download option
        st.download_button(
            "📥 Download Your Processed Data",
            st.session_state.merged_df.to_csv(index=False),
            "your_business_data.csv",
            "text/csv"
        )
    
    with tab2:
        st.subheader("🎯 Customer Risk Analysis")
        st.info("""
        **What this tells you:** Which customers are most likely to stop buying from you.
        This helps you focus your retention efforts on the right customers.
        """)

        churn_df = st.session_state.churn_df
        # Show risk level counts using the 'risk_level' column
        risk_counts = churn_df['risk_level'].value_counts().reindex(['High Risk', 'Medium Risk', 'Low Risk'], fill_value=0)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⚠️ High Risk Customers", risk_counts['High Risk'])
        with col2:
            st.metric("🟧 Medium Risk Customers", risk_counts['Medium Risk'])
        with col3:
            st.metric("✅ Low Risk Customers", risk_counts['Low Risk'])

        # Pie chart for risk distribution
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='Customer Risk Distribution',
            color_discrete_map={'Low Risk': '#2E8B57', 'Medium Risk': '#FFA500', 'High Risk': '#FF6B6B'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # High risk customers table
        st.markdown("### ⚠️ High Risk Customers (Need Attention)")
        high_risk_df = churn_df[churn_df['risk_level'] == 'High Risk'].sort_values('churn_probability_final', ascending=False)
        if len(high_risk_df) > 0:
            display_cols = ['customer_id', 'churn_probability_final', 'total_amount', 'transaction_count']
            display_df = high_risk_df[display_cols].head(10).copy()
            display_df['Risk Probability'] = display_df['churn_probability_final'].apply(lambda x: f"{x:.1%}")
            display_df['Total Spent'] = display_df['total_amount'].apply(lambda x: f"{st.session_state.get('currency_symbol', '$')}{x:,.2f}")
            display_df['Transactions'] = display_df['transaction_count']
            display_df = display_df[['customer_id', 'Risk Probability', 'Total Spent', 'Transactions']]
            display_df.columns = ['Customer ID', 'Risk Probability', 'Total Spent', 'Transactions']
            st.dataframe(display_df, use_container_width=True)
        else:
            st.success("🎉 No high-risk customers found!")

        # Medium risk customers table
        st.markdown("### 🟧 Medium Risk Customers")
        medium_risk_df = churn_df[churn_df['risk_level'] == 'Medium Risk'].sort_values('churn_probability_final', ascending=False)
        if len(medium_risk_df) > 0:
            display_cols = ['customer_id', 'churn_probability_final', 'total_amount', 'transaction_count']
            display_df = medium_risk_df[display_cols].head(10).copy()
            display_df['Risk Probability'] = display_df['churn_probability_final'].apply(lambda x: f"{x:.1%}")
            display_df['Total Spent'] = display_df['total_amount'].apply(lambda x: f"{st.session_state.get('currency_symbol', '$')}{x:,.2f}")
            display_df['Transactions'] = display_df['transaction_count']
            display_df = display_df[['customer_id', 'Risk Probability', 'Total Spent', 'Transactions']]
            display_df.columns = ['Customer ID', 'Risk Probability', 'Total Spent', 'Transactions']
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No medium-risk customers found.")

        # Download
        st.download_button(
            "📥 Download Risk Analysis",
            churn_df.to_csv(index=False),
            "customer_risk_analysis.csv",
            "text/csv"
        )
    
    with tab3:
        st.subheader("💰 Revenue Insights")
        st.info("""
        **What this tells you:** How much revenue each customer is likely to generate in the future.
        This helps you identify your most valuable customers and growth opportunities.
        """)
        
        # Model performance
        if hasattr(st.session_state.revenue_df, 'attrs') and 'model_info' in st.session_state.revenue_df.attrs:
            model_info = st.session_state.revenue_df.attrs['model_info']
            performance = model_info['performance']
            
            # Check if revenue predictions are all zero
            if 'predicted_revenue' in st.session_state.revenue_df.columns:
                total_predicted = st.session_state.revenue_df['predicted_revenue'].sum()
                if total_predicted == 0:
                    st.warning("⚠️ **Revenue Warning:** All predicted revenue values are zero. This might indicate:")
                    st.markdown("""
                    - Your transaction amounts are very small or zero
                    - The model needs more diverse data to make predictions
                    - Check that your amount column contains reasonable values
                    """)
                    # Show some debug info
                    if 'total_revenue' in st.session_state.revenue_df.columns:
                        st.info(f"💰 Total historical revenue: {st.session_state.revenue_df['total_revenue'].sum():.2f}")
                    if 'avg_transaction' in st.session_state.revenue_df.columns:
                        st.info(f"💵 Average transaction value: {st.session_state.revenue_df['avg_transaction'].mean():.2f}")
            
            col1, col2 = st.columns(2)
            with col1:
                total_predicted = st.session_state.revenue_df['predicted_revenue'].sum()
                if total_predicted > 0:
                    st.metric("💰 Total Predicted Revenue", f"{st.session_state.get('currency_symbol', '$')}{total_predicted:,.2f}")
                else:
                    st.metric("💰 Total Predicted Revenue", "No data", help="Revenue predictions are zero - check your data")
            with col2:
                avg_predicted = st.session_state.revenue_df['predicted_revenue'].mean()
                if avg_predicted > 0:
                    st.metric("💵 Avg Predicted Revenue", f"{st.session_state.get('currency_symbol', '$')}{avg_predicted:.2f}")
                else:
                    st.metric("💵 Avg Predicted Revenue", "No data", help="Average revenue is zero - check your data")
        
        # Revenue segments
        col1, col2 = st.columns(2)
        
        with col1:
            if 'revenue_segment' in st.session_state.revenue_df.columns:
                segment_counts = st.session_state.revenue_df['revenue_segment'].value_counts()
                fig = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title='Customer Value Segments',
                    color_discrete_map={'Low Value': '#FF6B6B', 'Medium Value': '#FFA500', 'High Value': '#2E8B57'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
        with col2:
            # High value customers
            st.markdown("### 💎 High Value Customers")
            if 'predicted_revenue' in st.session_state.revenue_df.columns:
                high_value_threshold = st.session_state.revenue_df['predicted_revenue'].quantile(0.75)
                high_value_df = st.session_state.revenue_df[st.session_state.revenue_df['predicted_revenue'] > high_value_threshold].sort_values('predicted_revenue', ascending=False)
                
                if len(high_value_df) > 0:
                    # Check which columns are available
                    available_cols = ['customer_id', 'predicted_revenue']
                    if 'total_revenue' in high_value_df.columns:
                        available_cols.append('total_revenue')
                    if 'transaction_count' in high_value_df.columns:
                        available_cols.append('transaction_count')
                    
                    display_df = high_value_df[available_cols].head(10).copy()
                    display_df['Predicted Revenue'] = display_df['predicted_revenue'].apply(lambda x: f"{st.session_state.get('currency_symbol', '$')}{x:,.2f}")
                    
                    if 'total_revenue' in display_df.columns:
                        display_df['Current Revenue'] = display_df['total_revenue'].apply(lambda x: f"{st.session_state.get('currency_symbol', '$')}{x:,.2f}")
                        display_cols = ['customer_id', 'Predicted Revenue', 'Current Revenue']
                    else:
                        display_cols = ['customer_id', 'Predicted Revenue']
                    
                    if 'transaction_count' in display_df.columns:
                        display_cols.append('transaction_count')
                    
                    display_df = display_df[display_cols]
                    display_df.columns = ['Customer ID', 'Predicted Revenue'] + (['Current Revenue', 'Transactions'] if 'total_revenue' in high_value_df.columns else ['Transactions'] if 'transaction_count' in high_value_df.columns else [])
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("No high-value customers found in the current analysis.")
            else:
                st.warning("Revenue predictions not available in the current analysis.")
        
        # Download
        st.download_button(
            "📥 Download Revenue Analysis",
            st.session_state.revenue_df.to_csv(index=False),
            "revenue_analysis.csv",
            "text/csv"
        )
    
    with tab4:
        st.subheader("🤖 Smart Business Recommendations")
        st.info("""
        **Actionable insights** to help you grow your business and retain customers.
        """)
        # Existing metrics and info remain unchanged
        # --- Dynamic Recommendations Section ---
        st.markdown("### 📋 Your Action Plan")
        recommendations = [
            "🎯 **Focus on high-risk, high-value customers** - These are your most important customers to retain",
            "💰 **Upsell to high-value, low-risk customers** - They're ready to buy more from you",
            "📊 **Monitor customer behavior patterns** - Use the insights to predict future trends",
            "📈 **Track model performance** - Regular updates will improve predictions",
            "🎉 **Celebrate your loyal customers** - They're your business foundation"
        ]
        for rec in recommendations:
            st.markdown(f"- {rec}")
        # --- Dynamic, data-driven recommendations ---
        churn_df = st.session_state.get('churn_df')
        if churn_df is not None:
            st.markdown("#### Top 3 High-Risk, High-Value Customers (Retention Focus)")
            high_risk_high_value = churn_df[
                (churn_df['risk_level'] == 'High Risk') &
                (churn_df['total_amount'] > churn_df['total_amount'].quantile(0.75))
            ].sort_values('churn_probability_final', ascending=False).head(3)
            if not high_risk_high_value.empty:
                st.dataframe(high_risk_high_value[['customer_id', 'total_amount', 'churn_probability_final', 'risk_level']], use_container_width=True)
            else:
                st.info("No high-risk, high-value customers found in the current analysis.")

            st.markdown("#### Top 3 High-Value, Low-Risk Customers (Upsell Focus)")
            high_value_low_risk = churn_df[
                (churn_df['risk_level'] == 'Low Risk') &
                (churn_df['total_amount'] > churn_df['total_amount'].quantile(0.75))
            ].sort_values('total_amount', ascending=False).head(3)
            if not high_value_low_risk.empty:
                st.dataframe(high_value_low_risk[['customer_id', 'total_amount', 'churn_probability_final', 'risk_level']], use_container_width=True)
            else:
                st.info("No high-value, low-risk customers found in the current analysis.")

            st.markdown("#### Top 3 Customers with Rising Risk (if available)")
            # If you have previous churn probabilities, you could compare and show rising risk
            # For now, just show top 3 by churn_probability_final not already in high risk
            rising_risk = churn_df[
                (churn_df['risk_level'] != 'High Risk')
            ].sort_values('churn_probability_final', ascending=False).head(3)
            if not rising_risk.empty:
                st.dataframe(rising_risk[['customer_id', 'total_amount', 'churn_probability_final', 'risk_level']], use_container_width=True)
            else:
                st.info("No rising risk customers found in the current analysis.")
    
    # Data Management section
    st.markdown("---")
    st.subheader("💾 Data Management")
    
# Only show download/results UI if analysis has been run
if 'churn_df' in st.session_state and 'revenue_df' in st.session_state and 'merged_df' in st.session_state:
        st.markdown("### 📥 Download Results")
        st.info("Download your analysis results as CSV files for further analysis.")
        
        if st.button("📊 Download Churn Analysis", use_container_width=True):
            csv = st.session_state.churn_df.to_csv(index=False)
            st.download_button(
                "💾 Download CSV",
                csv,
                "churn_analysis.csv",
                "text/csv",
                key="download_churn"
            )
        
        if st.button("💰 Download Revenue Analysis", use_container_width=True):
            csv = st.session_state.revenue_df.to_csv(index=False)
            st.download_button(
                "💾 Download CSV",
                csv,
                "revenue_analysis.csv",
                "text/csv",
                key="download_revenue"
            )
    
        st.markdown("---")
        if st.button("🚀 Launch Interactive Dashboard", use_container_width=True):
            if user_subscription == 'Premium' or user_role == 'developer':
                st.info("Redirecting to your interactive dashboard...")
                st.session_state['dashboard_user_id'] = st.session_state.user_id
                import streamlit as st
                st.switch_page("pages/dashboard.py")
            else:
                st.warning("🚀 **Premium Feature:** Interactive Dashboard is only available for Premium users.")
                st.info("Upgrade to Premium (₹500/month) to access the interactive dashboard and all advanced features!")

# Add organized sidebar with hamburger menu
with st.sidebar:
    # Initialize session state for sidebar menu - start as True (open)
    if 'show_sidebar_menu' not in st.session_state:
        st.session_state.show_sidebar_menu = True
    if 'show_sidebar_upgrade' not in st.session_state:
        st.session_state.show_sidebar_upgrade = False
    if 'show_sidebar_profile' not in st.session_state:
        st.session_state.show_sidebar_profile = False
    if 'show_sidebar_contact' not in st.session_state:
        st.session_state.show_sidebar_contact = False
    
    # Set sidebar to be open by default
    st.session_state.show_sidebar_menu = True
    
    # Show menu options (always visible since sidebar is always open)
    st.markdown('---')
    st.markdown("### 📋 Menu Options")
    
    # Profile Management section with expander (opens downward)
    st.markdown("#### 👤 Profile Management")
    with st.expander("👤 Profile", expanded=False):
        # Fetch user subscription and role from DB
        user_subscription, user_role = get_user_subscription_and_role_by_email(st.session_state.user_id)
        
        # Prepare user data
        email = st.session_state.user_id or "Not set"
        account_type = "Developer" if user_role == 'developer' else ("Premium" if user_subscription == 'Premium' else "Free")
        
        st.markdown("**👤 Profile**")
        st.write(f"Email: {email}")
        st.write(f"Type: {account_type}")
        st.write(f"Plan: {user_subscription}")
        
        # Show daily analysis limit for free users
        if user_subscription == 'Free' and user_role != 'developer':
            can_analyze, limit_message = check_daily_analysis_limit_by_email(st.session_state.user_id)
            if can_analyze:
                st.success(f"✅ {limit_message}")
            else:
                st.error(f"❌ {limit_message}")
        elif user_role == 'developer':
            st.success("Developer privileges!")
        elif user_subscription == 'Premium':
            st.success("Premium - Unlimited analyses!")
        
        st.markdown("---")
        st.markdown("**Actions:**")
        
        # Account action buttons - very compact
        if st.button("🔑 Password", key="profile_change_password_sidebar", use_container_width=True):
            st.info("Password change functionality would be implemented here!")
        if st.button("📧 Email", key="profile_update_email_sidebar", use_container_width=True):
            st.info("Email update functionality would be implemented here!")
        if st.button("❌ Delete", key="profile_delete_account_sidebar", use_container_width=True):
            st.warning("Account deletion functionality would be implemented here!")
    
    # Help section with expander (opens downward)
    st.markdown("#### ❓ Help")
    with st.expander("❓ Help", expanded=False):
        st.markdown("**❓ Help**")
        st.write("📧 support@autoinsights.com")
        st.write("📞 +91-1234567890")
        st.write("💬 Live Chat 24/7")
        st.write("📚 Docs: Coming soon")
    
    # Upgrade section with expander (opens downward)
    st.markdown("#### 🆙 Upgrade")
    with st.expander("🆙 Upgrade", expanded=False):
        st.markdown("**🎯 Plans**")
        
        st.markdown("**🚀 PREMIUM ₹500/mo**")
        st.write("🚀 Unlimited customers")
        st.write("📊 Advanced analytics")
        st.write("💰 Revenue forecasting")
        st.write("🎯 Churn prediction")
        st.write("📈 Detailed reports")
        st.write("�� Priority support")
        st.write("♾️ Unlimited daily analyses")
        
        st.markdown("---")
        st.markdown("**🆓 FREE**")
        st.write("Basic analytics (100 customers)")
        st.write("Simple churn prediction")
        st.write("Basic revenue insights")
        st.write("📊 2 analyses per day")
        st.write("❌ Advanced features")
        
        st.markdown("---")
        st.markdown("**Actions:**")
        
        # Upgrade action buttons
        if st.button("🆙 Upgrade", key="upgrade_premium_sidebar", use_container_width=True, type="primary"):
            st.success("Contact support@autoinsights.com to complete your upgrade!")
    
    # Sign Out section
    st.markdown("#### 🚪 Sign Out")
    if st.button("🚪 Sign Out", key="sidebar_signout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.rerun()
    
    st.markdown('---')

# Call cleanup_old_analyses for the current user at login/session start
if 'db_initialized' not in st.session_state:
    if init_database_schema():
        st.session_state.db_initialized = True
    else:
        st.error("❌ Failed to initialize database schema. Please check your MySQL connection.")
        st.stop()

if st.session_state.get('authenticated', False):
    cleanup_old_analyses(st.session_state.user_id)

def delete_my_account_and_data(user_id, email):
    """Delete all user data and account, and log the action"""
    engine = get_db_connection()
    if not engine:
        return False, "Database connection failed"
    try:
        with engine.connect() as conn:
            # Get all analysis IDs for this user
            analysis_ids = [row[0] for row in conn.execute(text("SELECT analysis_id FROM analysis_history WHERE user_id = :user_id"), {"user_id": user_id})]
            # Delete related data for each analysis
            for analysis_id in analysis_ids:
                conn.execute(text("DELETE FROM customer_data WHERE analysis_id = :analysis_id"), {"analysis_id": analysis_id})
                conn.execute(text("DELETE FROM churn_data WHERE analysis_id = :analysis_id"), {"analysis_id": analysis_id})
                conn.execute(text("DELETE FROM revenue_data WHERE analysis_id = :analysis_id"), {"analysis_id": analysis_id})
                conn.execute(text("DELETE FROM analysis_history WHERE analysis_id = :analysis_id"), {"analysis_id": analysis_id})
            # Delete user account
            conn.execute(text("DELETE FROM users WHERE email = :user_id"), {"user_id": user_id})
            # Log the action
            conn.execute(text("""
                INSERT INTO security_log (event_type, email, details, timestamp)
                VALUES (:event_type, :email, :details, :timestamp)
            """), {
                "event_type": "DELETE_ACCOUNT",
                "email": email,
                "details": "User deleted their account and all data.",
                "timestamp": datetime.now()
            })
            conn.commit()
        return True, "Your account and all data have been deleted."
    except Exception as e:
        return False, f"Error deleting account: {str(e)}"

# Add a button in the sidebar for account deletion
with st.sidebar:
    # Remove the extra space so the button is as high as possible
    if st.session_state.get('authenticated', False):
        if st.button('🗑️ Delete My Account and Data', key='delete_account_btn'):
            confirm = st.checkbox('Are you sure? This action is irreversible.', key='delete_account_confirm')
            if confirm:
                # Get user_id and email
                engine = get_db_connection()
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT email FROM users WHERE email = :email"), {"email": st.session_state.user_id})
                    user_row = result.fetchone()
                    if user_row:
                        db_user_id = user_row[0]
                        success, msg = delete_my_account_and_data(db_user_id, st.session_state.user_id)
                        if success:
                            st.success(msg)
                            st.session_state.clear()
                            st.experimental_rerun()
                        else:
                            st.error(msg)
    st.markdown('---')

def register_user(email, password):
    # ... existing code ...
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO users (email, password_hash)
                VALUES (:email, :password_hash)
            """), {"email": email, "password_hash": hash_password})
            conn.commit()
        return True
    except exc.IntegrityError:
        return "Email already registered."
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        return "Registration failed due to a database error."

def authenticate_user(email, password):
    # ... existing code ...
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT password_hash FROM users WHERE email = :email"), {"email": email}).fetchone()
        if result and bcrypt.checkpw(password.encode('utf-8'), result[0].encode('utf-8')):
            return True
        return False
    except Exception as e:
        logger.error(f"Error authenticating user: {e}")
        return False

# ... existing code ...
        st.session_state['user_id'] = email
        st.session_state['logged_in'] = True
# ... existing code ...
def save_analysis(user_id, churn_df, revenue_df, merged_df, summary):
    # ... existing code ...
    with engine.connect() as conn:
        result = conn.execute(text("INSERT INTO analysis_history (user_id, summary) VALUES (:user_id, :summary)"),
                                  {"user_id": user_id, "summary": json.dumps(summary)})
        analysis_id = result.lastrowid
        conn.commit()
    # ... existing code ...