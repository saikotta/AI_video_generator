from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    JWT_SECRET: str = "super_secret_jwt_key_change_me_in_production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    NARRATION_WPM: int = 130
    CALIBRATED_NARRATION_WPM: int = 130
    PEXELS_API_KEY: str = ""
    MEDIA_STORAGE_PATH: str = "./media"
    TTS_VOICE: str = "en-US-ChristopherNeural"
    FREESOUND_API_KEY: str = ""

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_MAX_CONCURRENCY: int = 5
    GROQ_MAX_RETRIES: int = 5
    GROQ_RETRY_BASE_DELAY: float = 1.0
    ENABLE_SCRIPT_CACHE: bool = False
    SCRIPT_PROVIDER: str = "groq"

    # What to do when no candidate clip is verified for a shot:
    #   "block"       - stop and wait for manual approval (strict, original behaviour)
    #   "best_effort" - use the best-scoring unverified clip and record it as UNCERTAIN
    #                   in the scene metadata so it can be reviewed later
    UNCERTAIN_FOOTAGE_POLICY: str = "block"

    # Shots about abstract/internal processes (enzymes, cellular respiration, ...) cannot be
    # filmed truthfully with stock footage. When True the pipeline renders a title card for
    # them instead of stopping or using unrelated footage.
    ABSTRACT_SHOT_CARD_ENABLED: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()