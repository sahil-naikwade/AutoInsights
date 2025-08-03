import os
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import quote_plus  # Added for password encoding

# Load environment variables
load_dotenv()

class Config:
    # Database Configuration
    DB_HOST = os.getenv('DB_HOST')
    DB_NAME = os.getenv("DB_NAME", "autoinsightsdb")
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_PORT = os.getenv('DB_PORT')
    
    # Dashboard Configuration
    DASHBOARD_URL = os.getenv('DASHBOARD_URL')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() in ('true', '1', 't')
    
    # Google OAuth
    GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
    GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
    
    # Path Configuration
    PROJECT_ROOT = Path(__file__).parent.parent
    FRONTEND_DIR = PROJECT_ROOT / 'Frontend'
    BACKUP_DIR = PROJECT_ROOT / 'backups'
    
    @classmethod
    def get_db_url(cls):
        password = quote_plus(cls.DB_PASSWORD)  # URL-encode the password
        return f"mysql+pymysql://{cls.DB_USER}:{password}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"