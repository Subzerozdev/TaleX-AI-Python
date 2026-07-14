from motor.motor_asyncio import AsyncIOMotorClient
from loguru import logger
from app.core.config import settings

client = None
db = None

async def init_mongodb():
    global client, db
    try:
        client = AsyncIOMotorClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[settings.MONGO_DB_NAME]
        # Test connection
        await client.admin.command('ping')
        logger.info(f"Connected to MongoDB Atlas database: {settings.MONGO_DB_NAME}")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")

async def close_mongodb():
    global client
    if client:
        client.close()
        logger.info("MongoDB connection closed")

async def get_series_metadata(series_id: str) -> dict:
    if db is None:
        logger.warning("MongoDB not initialized")
        return {}
    
    collection = db["series_metadata"]
    metadata = await collection.find_one({"_id": series_id})
    return metadata or {}
