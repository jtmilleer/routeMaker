# LAYER: Database
# PURPOSE: SQLAlchemy async engine and session factory for SQLite (locally).
#          Swap DATABASE_URL to postgresql+asyncpg://... for Postgres on AWS —
#          no other code needs to change.
#          get_db() is a FastAPI dependency that yields a session per request.

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from backend.core.config import settings

# Create the async engine.
# connect_args is SQLite-specific (allows use across threads in async context).
connect_args = {"check_same_thread": False, "timeout": 30} if "sqlite" in settings.database_url else {}

engine = create_async_engine(
    settings.database_url,
    connect_args=connect_args,
    echo=False,  # Set True to log all SQL queries during development
)

AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models. Import and inherit from this."""
    pass


async def get_db() -> AsyncSession:
    """
    FastAPI dependency. Yields an async DB session for the duration of a request,
    then commits on success or rolls back on exception.

    Usage in a route:
        @router.post("/example")
        async def my_route(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables():
    """Create all tables defined in ORM models. Called once on app startup."""
    # Import models here to ensure they are registered with Base before create_all
    import backend.models.db_models  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # create_all only creates missing tables — it never ALTERs existing ones.
        # This project has no migration framework, so add new columns to existing
        # tables here (idempotent). ALTER TABLE ... ADD COLUMN works on both SQLite
        # and Postgres, so this stays valid after a future DATABASE_URL swap.
        await conn.run_sync(_ensure_columns)


# Columns that may be missing on an already-created table, by table name.
# Each entry is (column_name, SQL column type). Keep types ANSI-portable.
_ADDED_COLUMNS = {
    "users": [
        ("home_lat", "FLOAT"),
        ("home_lng", "FLOAT"),
    ],
    "rides": [
        ("activity_type", "VARCHAR(16) DEFAULT 'ride'"),
    ],
}


def _ensure_columns(conn):
    """Add any missing columns listed in _ADDED_COLUMNS. Runs inside run_sync."""
    from sqlalchemy import inspect, text

    inspector = inspect(conn)
    existing_tables = set(inspector.get_table_names())
    for table, columns in _ADDED_COLUMNS.items():
        if table not in existing_tables:
            continue  # create_all already made it with all columns
        present = {c["name"] for c in inspector.get_columns(table)}
        for name, sql_type in columns:
            if name not in present:
                conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {name} {sql_type}'))
