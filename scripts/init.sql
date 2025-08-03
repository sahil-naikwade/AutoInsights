-- AutoInsights Database Initialization Script
-- This script creates the necessary tables for the AutoInsights application

USE autoinsightsdb;

-- Create customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(20),
    registration_date DATE,
    last_purchase_date DATE,
    total_purchases DECIMAL(10,2) DEFAULT 0.00,
    total_spent DECIMAL(10,2) DEFAULT 0.00,
    purchase_frequency INT DEFAULT 0,
    days_since_last_purchase INT,
    churn_probability DECIMAL(5,4) DEFAULT 0.0000,
    customer_segment VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create transactions table
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT,
    transaction_date DATETIME,
    amount DECIMAL(10,2),
    product_category VARCHAR(100),
    payment_method VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Create revenue_forecasts table
CREATE TABLE IF NOT EXISTS revenue_forecasts (
    forecast_id INT PRIMARY KEY AUTO_INCREMENT,
    forecast_date DATE,
    predicted_revenue DECIMAL(12,2),
    confidence_interval_lower DECIMAL(12,2),
    confidence_interval_upper DECIMAL(12,2),
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create model_performance table
CREATE TABLE IF NOT EXISTS model_performance (
    performance_id INT PRIMARY KEY AUTO_INCREMENT,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    accuracy_score DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    training_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create application_logs table
CREATE TABLE IF NOT EXISTS application_logs (
    log_id INT PRIMARY KEY AUTO_INCREMENT,
    log_level VARCHAR(20),
    message TEXT,
    module VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_churn_probability ON customers(churn_probability);
CREATE INDEX idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_revenue_forecasts_date ON revenue_forecasts(forecast_date);
CREATE INDEX idx_model_performance_model_name ON model_performance(model_name);

-- Insert sample data (optional) - Replace with your actual data
-- INSERT INTO customers (customer_name, email, phone, registration_date, last_purchase_date, total_purchases, total_spent, purchase_frequency, days_since_last_purchase, churn_probability, customer_segment) VALUES
-- ('John Doe', 'john.doe@example.com', '+1234567890', '2023-01-15', '2024-01-10', 15, 1250.00, 3, 5, 0.1500, 'High Value'),
-- ('Jane Smith', 'jane.smith@example.com', '+1234567891', '2023-02-20', '2024-01-05', 8, 450.00, 2, 10, 0.2500, 'Medium Value'),
-- ('Bob Johnson', 'bob.johnson@example.com', '+1234567892', '2023-03-10', '2023-12-20', 3, 150.00, 1, 25, 0.7500, 'Low Value');

-- Insert sample transactions (optional) - Replace with your actual data
-- INSERT INTO transactions (customer_id, transaction_date, amount, product_category, payment_method) VALUES
-- (1, '2024-01-10 14:30:00', 85.00, 'Electronics', 'Credit Card'),
-- (1, '2024-01-05 10:15:00', 120.00, 'Clothing', 'Credit Card'),
-- (2, '2024-01-05 16:45:00', 75.00, 'Home & Garden', 'Debit Card'),
-- (3, '2023-12-20 11:20:00', 50.00, 'Books', 'Cash');

-- Insert sample revenue forecast (optional) - Replace with your actual data
-- INSERT INTO revenue_forecasts (forecast_date, predicted_revenue, confidence_interval_lower, confidence_interval_upper, model_version) VALUES
-- ('2024-02-01', 125000.00, 115000.00, 135000.00, 'v1.0.0'),
-- ('2024-03-01', 132000.00, 122000.00, 142000.00, 'v1.0.0'),
-- ('2024-04-01', 138000.00, 128000.00, 148000.00, 'v1.0.0'); 