import pandas as pd
from sqlalchemy import create_engine, exc
import os
from pathlib import Path
import logging
from utils.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_timestamps_to_str(df):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df

def upload_to_sql(user_id, analysis_id, churn_df, revenue_df):
    """Upload processed data to SQL database with enhanced error handling, for a specific user_id only"""
    try:
        # Configure database connection using Config
        db_url = Config.get_db_url()
        engine = create_engine(db_url, pool_pre_ping=True)
        
        # Read processed data
        output_dir = Path(__file__).parent.parent / 'output'
        churn_path = output_dir / 'churn_data.csv'
        revenue_path = output_dir / 'revenue_data.csv'
        
        print('DEBUG: Reading churn data from:', churn_path)
        print('DEBUG: File exists?', churn_path.exists())
        churn_df = pd.read_csv(churn_path)
        print('DEBUG: churn_df shape after read:', churn_df.shape)
        print('DEBUG: churn_df head after read:', churn_df.head())
        revenue_df = pd.read_csv(revenue_path)
        
        # Data validation
        if churn_df.empty or revenue_df.empty:
            logger.error("Empty dataframes detected")
            return False, "Empty dataframes detected"
        
        # Ensure user_id column exists and filter for current user
        if 'user_id' not in churn_df.columns:
            churn_df['user_id'] = user_id
        else:
            churn_df = churn_df[churn_df['user_id'] == user_id]
        print('DEBUG: churn_df shape after user_id filter:', churn_df.shape)
        print('DEBUG: churn_df head after user_id filter:', churn_df.head())
        if 'user_id' not in revenue_df.columns:
            revenue_df['user_id'] = user_id
        else:
            revenue_df = revenue_df[revenue_df['user_id'] == user_id]
        churn_df.columns = churn_df.columns.str.strip()
        revenue_df.columns = revenue_df.columns.str.strip()
        print('DEBUG (upload_to_sql): churn_df columns at upload:', churn_df.columns.tolist())
        print('DEBUG (upload_to_sql): revenue_df columns at upload:', revenue_df.columns.tolist())
        # Map columns to database schema
        # Use churn_probability_final if available, otherwise use churn_probability
        if 'churn_probability_final' in churn_df.columns:
            churn_df['churn_probability'] = churn_df['churn_probability_final']
        
        # Create last_activity from last_purchase if available
        if 'last_purchase' in churn_df.columns and 'last_activity' not in churn_df.columns:
            churn_df['last_activity'] = pd.to_datetime(churn_df['last_purchase']).dt.date
        
        # Fix days_since_last_purchase: ensure it is an integer (number of days)
        if 'days_since_last_purchase' in churn_df.columns:
            # If it's a datetime or string, convert to int (days)
            if pd.api.types.is_datetime64_any_dtype(churn_df['days_since_last_purchase']):
                churn_df['days_since_last_purchase'] = (pd.Timestamp.now() - churn_df['days_since_last_purchase']).dt.days
            else:
                # Try to convert to int, if fails, set to 0
                churn_df['days_since_last_purchase'] = pd.to_numeric(churn_df['days_since_last_purchase'], errors='coerce').fillna(0).astype(int)
        # Drop last_purchase if present in churn_df and revenue_df
        if 'last_purchase' in churn_df.columns:
            churn_df = churn_df.drop(columns=['last_purchase'])
        if 'last_purchase' in revenue_df.columns:
            revenue_df = revenue_df.drop(columns=['last_purchase'])
        # Filter churn_df to only include columns that exist in the churn_data table
        churn_table_columns = [
            'user_id', 'analysis_id', 'customer_id', 'churn_probability', 'risk_level',
            'total_amount', 'transaction_count', 'days_since_last_purchase', 'last_activity'
        ]
        churn_df_filtered = churn_df[[col for col in churn_table_columns if col in churn_df.columns]].copy()
        # Filter revenue_df to only include columns that exist in the revenue_data table
        revenue_table_columns = [
            'user_id', 'customer_id', 'analysis_id', 'predicted_revenue', 'total_revenue', 'revenue_segment',
            'transaction_count', 'avg_transaction_value'
        ]
        if 'user_id' not in revenue_df.columns:
            revenue_df['user_id'] = user_id
        revenue_df_filtered = revenue_df[[col for col in revenue_table_columns if col in revenue_df.columns]].copy()
        
        # Remove any rows not matching the current user_id
        churn_df = churn_df[churn_df['user_id'] == user_id]
        revenue_df = revenue_df[revenue_df['user_id'] == user_id]
        # Log if any rows were dropped
        if churn_df.shape[0] == 0 or revenue_df.shape[0] == 0:
            logger.warning('No data for current user_id after filtering. Nothing will be uploaded.')
            return False, 'No data for current user_id after filtering.'

        # Ensure required columns exist and fill missing with defaults
        required_cols = ['country', 'gender', 'total_revenue', 'risk_level', 'revenue_segment']
        for col in required_cols:
            if col not in churn_df.columns:
                churn_df[col] = 'Unknown' if col in ['country', 'gender', 'risk_level', 'revenue_segment'] else 0
            churn_df[col] = churn_df[col].fillna('Unknown' if col in ['country', 'gender', 'risk_level', 'revenue_segment'] else 0)
            if col not in revenue_df.columns:
                revenue_df[col] = 'Unknown' if col in ['country', 'gender', 'risk_level', 'revenue_segment'] else 0
            revenue_df[col] = revenue_df[col].fillna('Unknown' if col in ['country', 'gender', 'risk_level', 'revenue_segment'] else 0)
        # Convert all Timestamp columns to string
        churn_df = convert_timestamps_to_str(churn_df)
        revenue_df = convert_timestamps_to_str(revenue_df)
        
        # Upload to SQL database with transaction handling
        with engine.begin() as connection:
            try:
                # Upload churn data
                churn_df_filtered.to_sql(
                    'churn_data', 
                    con=connection, 
                    if_exists='append', 
                    index=False
                )
                # Upload revenue data
                revenue_df_filtered.to_sql(
                    'revenue_data', 
                    con=connection, 
                    if_exists='append', 
                    index=False
                )
                logger.info("Data uploaded to SQL successfully")
                return True, "✅ Data uploaded to SQL successfully."
                
            except exc.SQLAlchemyError as e:
                logger.error(f"Database error: {str(e)}")
                connection.rollback()
                return False, f"❌ Database error: {str(e)}"
                
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False, f"❌ Unexpected error: {str(e)}"