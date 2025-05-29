from pydantic import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:LSI9dhUrj1V7kzxE@db.homjbtgwrphodegiswaf.supabase.co:5432/postgres"
    
    class Config:
        env_file = ".env"

settings = Settings() 