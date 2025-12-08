"""Initialize PostgreSQL database schema."""
import asyncio
import sys
from pathlib import Path
import asyncpg

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings


async def init_database():
    """Create database tables if they don't exist."""
    # Connect to PostgreSQL
    conn = await asyncpg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        database=settings.postgres_database,
    )
    
    try:
        # Create patents_metadata table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS patents_metadata (
                id SERIAL PRIMARY KEY,
                publication_number VARCHAR(50) UNIQUE NOT NULL,
                title TEXT NOT NULL,
                publication_date DATE,
                cpc_codes JSONB,
                url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on publication_number for fast lookups
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_patents_publication_number 
            ON patents_metadata(publication_number)
        """)
        
        # Create litigation_cases table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS litigation_cases (
                id SERIAL PRIMARY KEY,
                case_number VARCHAR(100) NOT NULL,
                case_name TEXT,
                court_name VARCHAR(200),
                filing_date DATE,
                case_status VARCHAR(50),
                plaintiff_name TEXT,
                defendant_name TEXT,
                patent_id VARCHAR(50),
                outcome TEXT,
                source VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(case_number, patent_id)
            )
        """)
        
        # Create index on patent_id for fast lookups
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_litigation_patent_id 
            ON litigation_cases(patent_id)
        """)
        
        # Create index on case_number
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_litigation_case_number 
            ON litigation_cases(case_number)
        """)
        
        print("Database schema initialized successfully!")
        
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(init_database())

