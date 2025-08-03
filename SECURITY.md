# Security Guide for AutoInsights

This guide provides best practices for securing your AutoInsights deployment and protecting sensitive information.

## 🔐 Security Best Practices

### 1. Environment Variables
Never commit sensitive information to version control. Always use environment variables for:
- Database passwords
- API keys
- Secret keys
- Email credentials
- External service tokens

### 2. Password Security

#### Database Passwords
- Use strong, unique passwords (minimum 12 characters)
- Include uppercase, lowercase, numbers, and special characters
- Avoid common words or patterns
- Change passwords regularly

**Example strong password:**
```
K9#mN2$pL8@vX5!qR3&wE7
```

#### Secret Keys
Generate cryptographically secure random keys:

```bash
# Generate a secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate a secure JWT secret
python -c "import secrets; print(secrets.token_hex(32))"
```

### 3. Environment Setup

#### Create Secure .env File
```bash
# Copy the template
cp env.example .env

# Edit with secure values
nano .env
```

#### Required Environment Variables
```env
# Database Configuration
DB_HOST=localhost
DB_NAME=autoinsightsdb
DB_USER=autoinsights_user
DB_PASSWORD=your_very_secure_password_here
DB_PORT=3306

# Security Keys (generate these securely)
SECRET_KEY=your_generated_secret_key_here
JWT_SECRET_KEY=your_generated_jwt_secret_here

# Application Settings
ENVIRONMENT=production
DEBUG_MODE=False
LOG_LEVEL=WARNING
```

### 4. Database Security

#### MySQL Security Configuration
```sql
-- Create a dedicated user with minimal privileges
CREATE USER 'autoinsights_user'@'localhost' IDENTIFIED BY 'your_secure_password';
GRANT SELECT, INSERT, UPDATE, DELETE ON autoinsightsdb.* TO 'autoinsights_user'@'localhost';
FLUSH PRIVILEGES;

-- Remove root access from remote hosts
DELETE FROM mysql.user WHERE User='root' AND Host NOT IN ('localhost', '127.0.0.1', '::1');
FLUSH PRIVILEGES;
```

#### Docker Database Security
```yaml
# In docker-compose.yml, use environment variables
environment:
  - MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD}
  - MYSQL_DATABASE=${DB_NAME}
  - MYSQL_USER=${DB_USER}
  - MYSQL_PASSWORD=${DB_PASSWORD}
```

### 5. API Security

#### Rate Limiting
Implement rate limiting for API endpoints:
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
```

#### CORS Configuration
```python
# Configure CORS properly
CORS_ORIGINS = [
    "http://localhost:8501",
    "https://yourdomain.com"
]
```

### 6. File Permissions

#### Secure File Permissions
```bash
# Set proper file permissions
chmod 600 .env
chmod 644 *.py
chmod 755 logs/
chmod 755 output/
```

#### Docker Volume Permissions
```yaml
# In docker-compose.yml
volumes:
  - ./data:/app/data:ro  # Read-only for data
  - ./logs:/app/logs:rw  # Read-write for logs
```

### 7. Network Security

#### Firewall Configuration
```bash
# Allow only necessary ports
sudo ufw allow 8501/tcp  # Streamlit
sudo ufw allow 3306/tcp  # MySQL (if external)
sudo ufw deny 22/tcp      # SSH (if not needed)
```

#### SSL/TLS Configuration
```nginx
# In nginx.conf
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
}
```

### 8. Logging and Monitoring

#### Secure Logging
```python
# Configure secure logging
import logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

#### Audit Logging
```sql
-- Create audit log table
CREATE TABLE audit_logs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id VARCHAR(255),
    action VARCHAR(100),
    resource VARCHAR(255),
    ip_address VARCHAR(45),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 9. Data Protection

#### Data Encryption
```python
# Encrypt sensitive data
from cryptography.fernet import Fernet

def encrypt_data(data):
    key = Fernet.generate_key()
    f = Fernet(key)
    return f.encrypt(data.encode())
```

#### Data Masking
```python
# Mask sensitive data in logs
def mask_email(email):
    if '@' in email:
        username, domain = email.split('@')
        return f"{username[:2]}***@{domain}"
    return "***"
```

### 10. Regular Security Updates

#### Update Dependencies
```bash
# Regularly update dependencies
pip list --outdated
pip install --upgrade -r requirements.txt

# Check for security vulnerabilities
safety check
```

#### Database Updates
```sql
-- Regularly update database passwords
ALTER USER 'autoinsights_user'@'localhost' IDENTIFIED BY 'new_secure_password';
FLUSH PRIVILEGES;
```

### 11. Backup Security

#### Encrypted Backups
```bash
# Create encrypted database backups
mysqldump -u autoinsights_user -p autoinsightsdb | gpg -e -r your-email@domain.com > backup.sql.gpg
```

#### Secure Backup Storage
```bash
# Store backups securely
aws s3 cp backup.sql.gpg s3://your-secure-bucket/backups/
```

### 12. Monitoring and Alerts

#### Security Monitoring
```python
# Monitor for suspicious activities
def log_security_event(event_type, details):
    logging.warning(f"SECURITY_EVENT: {event_type} - {details}")
    # Send alert to security team
```

#### Health Checks
```python
# Implement health checks
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }
```

## 🚨 Security Checklist

- [ ] All passwords are strong and unique
- [ ] Environment variables are properly set
- [ ] Database user has minimal privileges
- [ ] SSL/TLS is configured for production
- [ ] Firewall rules are properly configured
- [ ] Logs are secured and monitored
- [ ] Regular security updates are scheduled
- [ ] Backups are encrypted and secure
- [ ] Rate limiting is implemented
- [ ] CORS is properly configured
- [ ] File permissions are secure
- [ ] Audit logging is enabled

## 🔍 Security Testing

### Vulnerability Scanning
```bash
# Run security scans
bandit -r .
safety check
npm audit  # If using Node.js components
```

### Penetration Testing
```bash
# Test for common vulnerabilities
sqlmap -u "http://localhost:8501" --batch
nikto -h http://localhost:8501
```

## 📞 Security Contacts

- **Security Issues**: security@private.com
- **Emergency Contact**: +1-XXX-XXX-XXXX

## 📚 Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [Python Security Best Practices](https://python-security.readthedocs.io/)

---

**Remember**: Security is an ongoing process. Regularly review and update your security measures to protect against new threats. 