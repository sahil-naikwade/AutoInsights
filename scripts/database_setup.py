#!/usr/bin/env python3
"""
Database Setup Script for AutoBI
This script helps set up and troubleshoot the MySQL database connection.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymysql
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from utils.config import Config

def test_mysql_connection():
    """Test basic MySQL connection"""
    print("🔍 Testing MySQL connection...")
    
    db_url = Config.get_db_url()
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print(f"✅ Connection successful!")
            return engine
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return None

def test_pymysql_direct():
    """Test direct PyMySQL connection"""
    print("\n🔍 Testing direct PyMySQL connection...")
    try:
        connection = pymysql.connect(
            host=Config.DB_HOST,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            database=Config.DB_NAME,
            charset='utf8mb4'
        )
        print("✅ Direct PyMySQL connection successful!")
        connection.close()
        return True
    except Exception as e:
        print(f"❌ Direct PyMySQL connection failed: {str(e)}")
        return False

def check_database_exists():
    """Check if the database exists"""
    print("\n🗄️ Checking database existence...")
    try:
        connection = pymysql.connect(
            host=Config.DB_HOST,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            charset='utf8mb4'
        )
        with connection.cursor() as cursor:
            cursor.execute("SHOW DATABASES")
            databases = [row[0] for row in cursor.fetchall()]
            if Config.DB_NAME in databases:
                print(f"✅ Database '{Config.DB_NAME}' exists!")
                return True
            else:
                print(f"❌ Database '{Config.DB_NAME}' does not exist!")
                print("📋 Available databases:", databases)
                return False
    except Exception as e:
        print(f"❌ Error checking databases: {str(e)}")
        return False

def create_database():
    """Create the database if it doesn't exist"""
    print("\n🏗️ Creating database...")
    try:
        connection = pymysql.connect(
            host=Config.DB_HOST,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            charset='utf8mb4'
        )
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {Config.DB_NAME}")
            print(f"✅ Database '{Config.DB_NAME}' created successfully!")
        connection.close()
        return True
    except Exception as e:
        print(f"❌ Error creating database: {str(e)}")
        return False

def create_tables(engine):
        with engine.connect() as conn:
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
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS analysis_history (
                analysis_id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(100) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    churn_data JSON,
                    revenue_data JSON,
                    merged_data JSON,
                    summary JSON,
                total_customers INT,
                high_risk_customers INT,
                total_predicted_revenue DECIMAL(15,2),
                avg_predicted_revenue DECIMAL(15,2),
                analysis_summary TEXT,
                FOREIGN KEY (user_id) REFERENCES users(email) ON DELETE CASCADE
            )
        """))
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
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS churn_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(100) NOT NULL,
                analysis_id INT,
                    customer_id VARCHAR(50),
                risk_level VARCHAR(20),
                    churn_probability FLOAT,
                FOREIGN KEY (user_id) REFERENCES users(email) ON DELETE CASCADE,
                FOREIGN KEY (analysis_id) REFERENCES analysis_history(analysis_id) ON DELETE CASCADE
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS revenue_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(100) NOT NULL,
                analysis_id INT,
                    customer_id VARCHAR(50),
                revenue_segment VARCHAR(50),
                predicted_revenue DECIMAL(15,2),
                FOREIGN KEY (user_id) REFERENCES users(email) ON DELETE CASCADE,
                FOREIGN KEY (analysis_id) REFERENCES analysis_history(analysis_id) ON DELETE CASCADE
                )
            """))
            conn.execute(text("""
            CREATE OR REPLACE VIEW full_dashboard_data AS
            SELECT
                c.customer_id,
                c.age,
                c.gender,
                c.country,
                c.total_spent,
                c.transaction_count,
                c.first_purchase_date,
                c.last_purchase_date,
                ch.risk_level,
                ch.churn_probability,
                r.revenue_segment,
                r.predicted_revenue,
                a.analysis_id,
                a.user_id,
                a.timestamp AS analysis_date
            FROM customer_data c
            LEFT JOIN churn_data ch ON c.customer_id = ch.customer_id AND c.analysis_id = ch.analysis_id
            LEFT JOIN revenue_data r ON c.customer_id = r.customer_id AND c.analysis_id = r.analysis_id
            JOIN analysis_history a ON c.analysis_id = a.analysis_id
            """))
            conn.commit()
    print("✅ All tables and views created successfully.")

def test_table_operations(engine):
    """Test basic table operations"""
    print("\n🧪 Testing table operations...")
    
    try:
        with engine.connect() as conn:
            # Test inserting a user
            conn.execute(text("""
                INSERT INTO users (email, username, password_hash) 
                VALUES (:email, :username, :password_hash)
                ON DUPLICATE KEY UPDATE username = username
            """), {
                "email": "test@example.com",
                "username": "test_user",
                "password_hash": "test_hash"
            })
            
            # Test selecting the user
            result = conn.execute(text("SELECT email FROM users WHERE email = :email"), {
                "email": "test@example.com"
            })
            user = result.fetchone()
            
            if user:
                print("✅ User operations successful!")
                
                # Test analysis history insert
                conn.execute(text("""
                    INSERT INTO analysis_history 
                    (user_id, timestamp, churn_data, revenue_data, merged_data, summary)
                    VALUES (:user_id, NOW(), :churn_data, :revenue_data, :merged_data, :summary)
                """), {
                    "user_id": user[0],
                    "churn_data": "[{\"test\": \"data\"}]",
                    "revenue_data": "[{\"test\": \"data\"}]",
                    "merged_data": "[{\"test\": \"data\"}]",
                    "summary": "{\"test\": \"summary\"}"
                })
                
                print("✅ Analysis history operations successful!")
                return True
            else:
                print("❌ User operations failed!")
                return False
                
    except Exception as e:
        print(f"❌ Error testing table operations: {str(e)}")
        return False

def main():
    """Main setup function"""
    print("🚀 AutoBI Database Setup")
    print("=" * 50)
    
    # Test direct PyMySQL connection
    if not test_pymysql_direct():
        print("\n❌ **CRITICAL ERROR:** Cannot connect to MySQL!")
        print("💡 **Please check:**")
        print("1. MySQL server is running")
        print("2. MySQL is accessible on localhost:3306")
        print("3. User 'root' exists with password 'Sahil@2004'")
        print("4. MySQL service is started")
        return False
    
    # Check if database exists
    if not check_database_exists():
        print("\n📝 Database doesn't exist. Creating it...")
        if not create_database():
            print("❌ Failed to create database!")
            return False
    
    # Test SQLAlchemy connection
    engine = test_mysql_connection()
    if not engine:
        print("\n❌ **CRITICAL ERROR:** Cannot connect via SQLAlchemy!")
        return False
    
    # Create tables
    if not create_tables(engine):
        print("❌ Failed to create tables!")
        return False
    
    # Test table operations
    if not test_table_operations(engine):
        print("❌ Failed to test table operations!")
        return False
    
    print("\n🎉 **SUCCESS!** Database setup completed successfully!")
    print("💡 You can now run the AutoBI application.")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ **SETUP FAILED!** Please fix the issues above and try again.")
        sys.exit(1)
    else:
        print("\n✅ **SETUP COMPLETE!** Your database is ready for AutoBI.") 