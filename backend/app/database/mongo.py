from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
import asyncio

client: AsyncIOMotorClient | None = None

def get_client() -> AsyncIOMotorClient:
    assert client is not None, "Mongo client not initialized"
    return client

async def connect_mongo():
    global client
    client = AsyncIOMotorClient(settings.MONGO_URI)

async def close_mongo():
    global client
    if client:
        client.close()
        client = None

def get_db():
    return get_client()[settings.MONGO_DB]

async def main():
    client = AsyncIOMotorClient(settings.MONGO_URI)
    await client.admin.command("ping")
    print("Mongo OK ✅")
    client.close()

if __name__ == "__main__":
    asyncio.run(main())
    print(connect_mongo())