from pydantic_settings import BaseSettings
from pymongo import MongoClient

class Settings(BaseSettings):
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "ITS"
    API_PREFIX: str = "/api"

settings = Settings()

def get_client():
    return MongoClient(settings.MONGO_URI)

def get_db_name():
    return settings.MONGO_DB


def show_db():
    client = get_client()

    db = client[get_db_name()]

    col = db['link']

    cursor = col.find({}).limit(10)
    for read in cursor:
        print(read)
