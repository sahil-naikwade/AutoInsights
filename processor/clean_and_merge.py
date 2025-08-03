import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def clean_and_merge(customer_path, transaction_path):
    """Enhanced cleaning and merging with data validation"""
    try:
        # Read with data type specification
        customers = pd.read_csv(customer_path, dtype={
            'customer_id': 'int',
            'gender': 'str',
            'age': 'int',
            'country': 'str',
            'signup_date': 'str'
        })
        
        transactions = pd.read_csv(transaction_path, dtype={
            'transaction_id': 'str',
            'customer_id': 'int',
            'amount': 'float',
            'date': 'str',
            'category': 'str'
        })

        # Data validation
        if customers.empty or transactions.empty:
            raise ValueError("Empty input file(s)")
        
        # Clean data
        customers = customers.dropna(subset=['customer_id'])
        transactions = transactions.dropna(subset=['customer_id', 'amount'])
        
        # Date conversion with error handling
        try:
            transactions['date'] = pd.to_datetime(transactions['date'], errors='coerce')
            customers['signup_date'] = pd.to_datetime(customers['signup_date'], errors='coerce')
        except Exception as e:
            logger.warning(f"Date conversion issues: {e}")
        
        # Merge with validation
        merged = pd.merge(transactions, customers, on='customer_id', how='inner', validate='many_to_one')
        
        if merged.empty:
            raise ValueError("Merge resulted in empty dataframe - check customer IDs")
        
        # Additional cleaning
        merged = merged[merged['amount'] > 0]  # Remove negative amounts
        merged['country'] = merged['country'].str.upper().str.strip()
        
        return merged
    
    except Exception as e:
        logger.error(f"Clean and merge failed: {e}")
        raise