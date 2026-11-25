"""Core Database Module for Social AI Platform.

Provides unified exports for the application's SQLAlchemy engine,
session factory, declarative Base, and database dependency helpers.
"""
from app.db.session import engine, SessionLocal
from app.db.base import Base
from app.db.dependencies import get_db

__all__ = ["engine", "SessionLocal", "Base", "get_db", "check_db_connection"]


def check_db_connection() -> dict:
    """Test connection to the configured database and return status info."""
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DATABASE(), VERSION()")).fetchone()
        return {
            "connected": True,
            "database": result[0] if result else None,
            "version": result[1] if result else None,
        }
