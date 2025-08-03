from sqlalchemy import create_engine, text
from utils.config import Config

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(Config.get_db_url())

def get_user_id(email, engine):
    with engine.connect() as conn:
        result = conn.execute(text("SELECT email FROM users WHERE email = :email"), {"email": email}).fetchone()
        return result[0] if result else None

def delete_all_user_data(email, engine):
    """Deletes all data associated with a user's email."""
    with engine.connect() as conn:
        # The ON DELETE CASCADE in the schema will handle deleting related data
        # in analysis_history, customer_data, churn_data, and revenue_data.
        conn.execute(text("DELETE FROM users WHERE email = :email"), {"email": email})
        conn.commit()
    print(f"✅ All data for user '{email}' has been deleted.")