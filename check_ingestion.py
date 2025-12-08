"""Check ingestion status."""
import asyncio
from db.qdrant_client import qdrant_client

async def check():
    await qdrant_client.create_collection_if_not_exists(384)
    info = await qdrant_client.client.get_collection('patents')
    print(f'Points: {info.points_count:,}')
    await qdrant_client.close()

if __name__ == "__main__":
    asyncio.run(check())


