# Configuration settings for Multiagent Stock Analyzer
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv('API_KEY')
    DATABASE_URL = os.getenv('DATABASE_URL')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Default settings
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3 