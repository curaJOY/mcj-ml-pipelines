"""
Configuration for CuraJOY Agentic Cyberbullying Detection System
================================================================

Google Gemini API Configuration and Settings
"""

import os
from typing import Optional

class Config:
    """Configuration class for the agentic detection system."""
    
    # Google Gemini API Configuration
    GOOGLE_API_KEY: Optional[str] = os.getenv('GOOGLE_API_KEY', None)
    GEMINI_MODEL: str = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')  # Fast model for production
    
    # Alternative models:
    # - gemini-1.5-pro: Higher accuracy, slower
    # - gemini-1.5-flash: Faster, good accuracy (recommended for challenge)
    
    # API Settings
    MAX_RETRIES: int = int(os.getenv('MAX_RETRIES', '3'))
    REQUEST_TIMEOUT: int = int(os.getenv('REQUEST_TIMEOUT', '30'))
    RATE_LIMIT_DELAY: float = float(os.getenv('RATE_LIMIT_DELAY', '1.0'))
    
    # Application Settings
    DEBUG: bool = os.getenv('DEBUG', 'true').lower() == 'true'
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    API_PORT: int = int(os.getenv('API_PORT', '8000'))
    
    # Performance Settings
    ENABLE_CACHING: bool = os.getenv('ENABLE_CACHING', 'true').lower() == 'true'
    CACHE_DURATION: int = int(os.getenv('CACHE_DURATION', '3600'))  # 1 hour
    
    # Security Settings
    ENABLE_INPUT_VALIDATION: bool = os.getenv('ENABLE_INPUT_VALIDATION', 'true').lower() == 'true'
    MAX_INPUT_LENGTH: int = int(os.getenv('MAX_INPUT_LENGTH', '1000'))
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv('RATE_LIMIT_PER_MINUTE', '100'))
    
    # Challenge-specific settings
    CHALLENGE_MODE: bool = os.getenv('CHALLENGE_MODE', 'true').lower() == 'true'
    ENABLE_TRADITIONAL_ML_FALLBACK: bool = True
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.GOOGLE_API_KEY:
            print("⚠️  Warning: GOOGLE_API_KEY not found. Using fallback analysis.")
            print("   To enable Gemini integration:")
            print("   1. Get API key from: https://aistudio.google.com/app/apikey")
            print("   2. Set environment variable: GOOGLE_API_KEY=your_key")
            print("   3. Or update this config file directly")
            return False
        return True
    
    @classmethod
    def get_gemini_config(cls) -> dict:
        """Get Gemini-specific configuration."""
        return {
            'api_key': cls.GOOGLE_API_KEY,
            'model': cls.GEMINI_MODEL,
            'max_retries': cls.MAX_RETRIES,
            'timeout': cls.REQUEST_TIMEOUT,
            'rate_limit_delay': cls.RATE_LIMIT_DELAY
        }

# Example environment setup instructions
SETUP_INSTRUCTIONS = """
🔧 SETUP INSTRUCTIONS:
=====================

1. Get Google Gemini API Key:
   - Visit: https://aistudio.google.com/app/apikey
   - Create new API key
   - Copy the key

2. Set Environment Variable:
   
   Windows (PowerShell):
   $env:GOOGLE_API_KEY="your_api_key_here"
   
   Windows (Command Prompt):
   set GOOGLE_API_KEY=your_api_key_here
   
   Linux/Mac:
   export GOOGLE_API_KEY="your_api_key_here"

3. Or update config.py directly:
   GOOGLE_API_KEY = "your_api_key_here"

4. Install required dependencies:
   pip install google-generativeai python-dotenv

5. Test the configuration:
   python -c "from config import Config; print('✅ Config loaded successfully' if Config.validate_config() else '❌ Config validation failed')"
"""

if __name__ == "__main__":
    print(SETUP_INSTRUCTIONS)
    Config.validate_config() 