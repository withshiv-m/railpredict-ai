"""
SQLAlchemy engine/session setup. Kept isolated so the underlying database
(currently SQLite) can be swapped for PostgreSQL later without touching the
rest of the app.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from backend.config import DATABASE_URL

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Safe to call repeatedly."""
    # Import models here so they're registered on Base before create_all
    from backend import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
