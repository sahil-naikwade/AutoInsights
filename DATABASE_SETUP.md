# AutoInsights Database Setup Guide

## 🛠️ MySQL Database Setup

### Prerequisites
- MySQL Server installed and running
- Python with required packages
- Access to MySQL root user

### Quick Setup

1. **Install MySQL Connector** (if not already installed):
   ```bash
   pip install mysql-connector-python
   ```

2. **Run the Database Setup Script**:
   ```bash
   python setup_database.py
   ```

3. **Start the AutoInsights App**:
   ```bash
   streamlit run app.py
   ```

### Manual Setup

If the automatic setup fails, follow these steps:

#### 1. Start MySQL Server
```bash
# Windows
net start mysql

# Linux/Mac
sudo systemctl start mysql
# or
sudo service mysql start
```

#### 2. Connect to MySQL
```bash
mysql -u root -p
# Enter password: <your-password>
```

#### 3. Create Database
```sql
CREATE DATABASE IF NOT EXISTS autoinsightsdb;
USE autoinsightsdb;
```

#### 4. Create Tables
```sql
-- Users table
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
);

-- Analysis history table
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
);

-- Churn data table
CREATE TABLE IF NOT EXISTS churn_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    analysis_id INT,
    customer_id VARCHAR(50),
    risk_level VARCHAR(20),
    churn_probability FLOAT,
    FOREIGN KEY (user_id) REFERENCES users(email) ON DELETE CASCADE,
    FOREIGN KEY (analysis_id) REFERENCES analysis_history(analysis_id) ON DELETE CASCADE
);

-- Revenue data table
CREATE TABLE IF NOT EXISTS revenue_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    analysis_id INT,
    customer_id VARCHAR(50),
    revenue_segment VARCHAR(50),
    predicted_revenue DECIMAL(15,2),
    FOREIGN KEY (user_id) REFERENCES users(email) ON DELETE CASCADE,
    FOREIGN KEY (analysis_id) REFERENCES analysis_history(analysis_id) ON DELETE CASCADE
);

-- Customer data table
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
);

-- View for dashboard
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
JOIN analysis_history a ON c.analysis_id = a.analysis_id;
```

### Troubleshooting

#### Connection Error: "Can't connect to MySQL server"
1. **Check if MySQL is running**:
   ```bash
   # Windows
   net start mysql
   
   # Linux/Mac
   sudo systemctl status mysql
   ```

2. **Verify MySQL installation**:
   ```bash
   mysql --version
   ```

3. **Test connection**:
   ```bash
   mysql -u root -p
   ```

#### Password Issues
If you get password errors:

1. **Reset MySQL root password**:
   ```sql
   ALTER USER 'root'@'localhost' IDENTIFIED BY '<your-password>';
   FLUSH PRIVILEGES;
   ```

2. **Create new user** (alternative):
   ```sql
   CREATE USER 'autoinsights_user'@'localhost' IDENTIFIED BY '<your-password>';
   GRANT ALL PRIVILEGES ON autoinsightsdb.* TO 'autoinsights_user'@'localhost';
   FLUSH PRIVILEGES;
   ```

#### Database Not Found
If the database doesn't exist:
```sql
CREATE DATABASE autoinsightsdb;
```

### Configuration

#### Database Connection Details
- **Host**: localhost
- **Port**: 3306 (default)
- **Database**: autoinsightsdb
- **User**: root
- **Password**: <your-password>

#### Alternative Configuration
If you want to use different credentials, modify `app.py`:
```python
# In get_db_connection() function
connection_string = 'mysql+pymysql://your_user:your_password@localhost/your_database'
```

### Security Notes

1. **Change default password** in production