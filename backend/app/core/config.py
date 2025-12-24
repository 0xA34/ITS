from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "ITS"
    API_PREFIX: str = "/api"

settings = Settings()

