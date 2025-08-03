#!/usr/bin/env python3
"""
Generate secure passwords and secret keys for AutoInsights application.
This script helps create cryptographically secure credentials.
"""

import secrets
import string
import argparse
import sys
from pathlib import Path


def generate_secure_password(length=16):
    """Generate a cryptographically secure password."""
    # Define character sets
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    # Ensure at least one character from each set
    password = [
        secrets.choice(lowercase),
        secrets.choice(uppercase),
        secrets.choice(digits),
        secrets.choice(symbols)
    ]
    
    # Fill the rest with random characters
    all_chars = lowercase + uppercase + digits + symbols
    password.extend(secrets.choice(all_chars) for _ in range(length - 4))
    
    # Shuffle the password
    password_list = list(password)
    secrets.SystemRandom().shuffle(password_list)
    
    return ''.join(password_list)


def generate_secret_key():
    """Generate a secure secret key for Flask."""
    return secrets.token_urlsafe(32)


def generate_jwt_secret():
    """Generate a secure JWT secret key."""
    return secrets.token_hex(32)


def generate_mysql_password():
    """Generate a MySQL-compatible password."""
    # MySQL has some restrictions on special characters
    mysql_symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    password = generate_secure_password(16)
    
    # Ensure MySQL compatibility
    password = password.replace('"', '')
    password = password.replace("'", '')
    password = password.replace('\\', '')
    
    return password


def create_env_template():
    """Create a template .env file with generated secure values."""
    env_content = f"""# AutoInsights Environment Configuration
# Generated on: {secrets.token_hex(8)}

# Database Configuration
DB_HOST=localhost
DB_NAME=autoinsightsdb
DB_USER=autoinsights_user
DB_PASSWORD={generate_mysql_password()}
DB_PORT=3306

# Dashboard Configuration
DASHBOARD_URL=http://localhost:8501
FLASK_DEBUG=False
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost

# Security (Generated Securely)
SECRET_KEY={generate_secret_key()}
JWT_SECRET_KEY={generate_jwt_secret()}

# Application Settings
LOG_LEVEL=INFO
ENVIRONMENT=development
DEBUG_MODE=True

# API Configuration
API_HOST=localhost
API_PORT=5000
CORS_ORIGINS=http://localhost:8501,http://localhost:3000

# Model Configuration
MODEL_PATH=models/
CHURN_MODEL_FILE=churn_model.pkl
REVENUE_MODEL_FILE=revenue_model.pkl

# Data Configuration
DATA_PATH=data/
OUTPUT_PATH=output/
LOGS_PATH=logs/

# Email Configuration (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_secure_app_password_here

# External APIs (Optional)
OPENAI_API_KEY=your_secure_openai_api_key_here
TWILIO_ACCOUNT_SID=your_secure_twilio_account_sid_here
TWILIO_AUTH_TOKEN=your_secure_twilio_auth_token_here

# Docker Database (for docker-compose)
MYSQL_ROOT_PASSWORD={generate_mysql_password()}
"""
    
    return env_content


def main():
    parser = argparse.ArgumentParser(description='Generate secure credentials for AutoInsights')
    parser.add_argument('--type', choices=['password', 'secret', 'jwt', 'mysql', 'env'], 
                       default='env', help='Type of credential to generate')
    parser.add_argument('--length', type=int, default=16, help='Password length')
    parser.add_argument('--output', help='Output file for .env template')
    
    args = parser.parse_args()
    
    try:
        if args.type == 'password':
            print(f"Secure Password: {generate_secure_password(args.length)}")
        
        elif args.type == 'secret':
            print(f"Secret Key: {generate_secret_key()}")
        
        elif args.type == 'jwt':
            print(f"JWT Secret: {generate_jwt_secret()}")
        
        elif args.type == 'mysql':
            print(f"MySQL Password: {generate_mysql_password()}")
        
        elif args.type == 'env':
            env_content = create_env_template()
            
            if args.output:
                output_path = Path(args.output)
                output_path.write_text(env_content)
                print(f"Environment template saved to: {output_path}")
                print("⚠️  IMPORTANT: Review and customize the generated values!")
            else:
                print("Generated .env template:")
                print("=" * 50)
                print(env_content)
                print("=" * 50)
                print("⚠️  IMPORTANT: Review and customize the generated values!")
        
        print("\n🔐 Security Tips:")
        print("- Store credentials securely")
        print("- Never commit .env files to version control")
        print("- Use different passwords for each environment")
        print("- Rotate credentials regularly")
        
    except Exception as e:
        print(f"Error generating credentials: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 