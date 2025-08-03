import mysql.connector
from mysql.connector import Error
import pandas as pd

def check_database():
    try:
        # Connect to the database
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='your_password',  # Replace with your MySQL root password
            database='autoinsightsdb',
        )
        cursor = conn.cursor()
        
        if conn.is_connected():
            print("✓ Successfully connected to MySQL database")
            
            # Check if full_dashboard_data table exists
            cursor.execute("SHOW TABLES LIKE 'full_dashboard_data'")
            table_exists = cursor.fetchone()
            
            if table_exists:
                print("✓ full_dashboard_data table exists")
                
                # Get table structure
                cursor.execute("DESCRIBE full_dashboard_data")
                columns = cursor.fetchall()
                print("\nTable structure:")
                for col in columns:
                    print(f"  {col[0]} - {col[1]} {'(NULL)' if col[2] == 'YES' else '(NOT NULL)'}")
                
                # Count total records
                cursor.execute("SELECT COUNT(*) FROM full_dashboard_data")
                total_count = cursor.fetchone()[0]
                print(f"\nTotal records in table: {total_count}")
                
                if total_count > 0:
                    # Show unique user_ids
                    cursor.execute("SELECT DISTINCT user_id FROM full_dashboard_data")
                    user_ids = [row[0] for row in cursor.fetchall()]
                    print(f"User IDs with data: {user_ids}")
                    
                    # Show sample records
                    cursor.execute("SELECT * FROM full_dashboard_data LIMIT 5")
                    sample_data = cursor.fetchall()
                    print("\nSample records (first 5):")
                    if sample_data:
                        # Get column names
                        cursor.execute("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'autoinsightsdb' AND TABLE_NAME = 'full_dashboard_data' ORDER BY ORDINAL_POSITION")
                        column_names = [row[0] for row in cursor.fetchall()]
                        
                        for i, record in enumerate(sample_data):
                            print(f"\nRecord {i+1}:")
                            for j, value in enumerate(record):
                                print(f"  {column_names[j]}: {value}")
                else:
                    print("⚠ Table exists but contains no data")
                    
            else:
                print("❌ full_dashboard_data table does not exist")
                
                # Show all tables to see what exists
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                print(f"\nExisting tables in autoinsightsdb database:")
                for table in tables:
                    print(f"  - {table[0]}")
            
            # Check for recent analysis_history entries
            print("\n" + "="*50)
            cursor.execute("SHOW TABLES LIKE 'analysis_history'")
            if cursor.fetchone():
                cursor.execute("SELECT user_id, analysis_id, created_at FROM analysis_history ORDER BY created_at DESC LIMIT 5")
                recent_analyses = cursor.fetchall()
                print("Recent analysis_history entries:")
                for analysis in recent_analyses:
                    print(f"  User: {analysis[0]}, Analysis: {analysis[1]}, Created: {analysis[2]}")
            
    except Error as e:
        print(f"❌ Error connecting to MySQL: {e}")
        
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    check_database()
