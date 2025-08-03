# In scripts/database_utils.py
# ... existing code ...
def get_user_id(email):
    with engine.connect() as conn:
        result = conn.execute(text("SELECT email FROM users WHERE email = :email"), {"email": email}).fetchone()
        return result[0] if result else None

def delete_all_user_data(email):
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM users WHERE email = :email"), {"email": email})
        conn.commit()

# In scripts/sql_upload.py
# ... existing code ...
def upload_to_sql(user_id, churn_df, revenue_df):
    # ... (user_id is now the email) ...

# In processor/clean_and_merge.py
# ... existing code ...
def clean_and_merge(customers_df, transactions_df, user_id):
    # ... (user_id is now the email) ...
