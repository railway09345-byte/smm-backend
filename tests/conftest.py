import os
import sys

# Immediately enforce testing environment flags
os.environ["TESTING"] = "true"
os.environ["ENVIRONMENT"] = "testing"
os.environ["RATE_LIMIT_ENABLED"] = "false"

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.db.base import Base
import app.models.autonomous_pipeline  # Register pipeline tables on Base.metadata
import app.models.campaign
import app.models.content
import app.models.scheduled_post
import app.models.approval
import app.db.session as db_session_module
import src.config.db as src_db_module
from app.db.dependencies import get_db
from app.db.seed import seed_database
from app.main import app


def assert_test_database_safety(target_engine):
    """
    CRITICAL DATABASE SAFETY GUARD:
    Verifies that target_engine points exclusively to an isolated test database.
    Rejects any connection to production, development, or non-test databases.
    Raises RuntimeError immediately if an unsafe database engine is detected.
    """
    url_str = str(target_engine.url).lower()

    # Reject any MySQL, PostgreSQL or external database unless explicitly qualified as an isolated test DB
    if any(driver in url_str for driver in ["mysql", "postgresql", "postgres", "mariadb"]):
        if "test" not in url_str:
            raise RuntimeError(
                f"FATAL DATABASE SAFETY VIOLATION: Automated tests attempted to run against "
                f"production/development SQL engine ({url_str}). Tests must NEVER access non-test databases!"
            )

    # Require explicit sqlite or test moniker
    is_safe_sqlite = url_str.startswith("sqlite")
    has_test_identifier = any(term in url_str for term in ["test", "memory", "testing"])

    if not (is_safe_sqlite or has_test_identifier):
        raise RuntimeError(
            f"FATAL DATABASE SAFETY VIOLATION: Unrecognized database engine ({url_str}). "
            f"Automated tests are strictly restricted to isolated test engines."
        )


# Dedicated isolated SQLite test engine
TEST_SQLITE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_isolated.db"))
TEST_DATABASE_URL = f"sqlite:///{TEST_SQLITE_PATH}"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Safety check on initialization
assert_test_database_safety(test_engine)

# Monkeypatch the application's global session and engine references to guarantee complete isolation
db_session_module.engine = test_engine
db_session_module.SessionLocal = TestSessionLocal
src_db_module.engine = test_engine
src_db_module.SessionLocal = TestSessionLocal


def override_get_db():
    assert_test_database_safety(test_engine)
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Ensure all TestClient instances automatically use the isolated test database
app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """
    Prepares the isolated test database and safely drops ONLY the test database upon completion.
    Never touches production or development MySQL databases.
    """
    assert_test_database_safety(test_engine)
    Base.metadata.create_all(bind=test_engine)
    with TestSessionLocal() as db:
        seed_database(db)
    yield
    # Ensure drop_all runs exclusively on the verified isolated test engine
    assert_test_database_safety(test_engine)
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()
    if os.path.exists(TEST_SQLITE_PATH):
        try:
            os.remove(TEST_SQLITE_PATH)
        except Exception:
            pass


@pytest.fixture
def db_session():
    assert_test_database_safety(test_engine)
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


