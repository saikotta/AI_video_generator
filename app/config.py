from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    JWT_SECRET: str = "super_secret_jwt_key_change_me_in_production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    NARRATION_WPM: int = 150
    PEXELS_API_KEY: str = ""
    MEDIA_STORAGE_PATH: str = "./media"
    TTS_VOICE: str = "en-US-ChristopherNeural"
    FREESOUND_API_KEY: str = ""

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_MAX_CONCURRENCY: int = 5
    GROQ_MAX_RETRIES: int = 5
    GROQ_RETRY_BASE_DELAY: float = 1.0
    ENABLE_SCRIPT_CACHE: bool = False
    SCRIPT_PROVIDER: str = "ollama"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()