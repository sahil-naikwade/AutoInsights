import mysql.connector
from sqlalchemy import create_engine, text
import pandas as pd
from utils.config import Config

def check_db_connection():
    """Checks the database connection and lists tables."""
    try:
        # Use the centralized config for the database connection
        engine = create_engine(Config.get_db_url())
        with engine.connect() as connection:
            print("Successfully connected to the database.")

        # Check if table exists
        result = connection.execute(text("SHOW TABLES LIKE 'full_dashboard_data'"))
        table_exists = result.fetchone()

        if table_exists:
            print("✅ full_dashboard_data table exists")

            # Get table structure
            print("\n📋 Table structure:")
            result = connection.execute(text("DESCRIBE full_dashboard_data"))
            for row in result:
                print(f"  {row[0]} ({row[1]})")

            # Count records
            result = connection.execute(text("SELECT COUNT(*) FROM full_dashboard_data"))
            count = result.fetchone()[0]
            print(f"\n📊 Total records: {count}")

            if count > 0:
                # Show sample data
                print("\n📄 Sample data (first 3 rows):")
                result = connection.execute(text("SELECT * FROM full_dashboard_data LIMIT 3"))
                for row in result:
                    print(f"  {row}")
            else:
                print("\n⚠️  Table exists but is empty")

        else:
            print("❌ full_dashboard_data table does not exist")

    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
