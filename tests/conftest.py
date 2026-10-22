import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.api.deps import get_db
from app.main import app

# ---------------------------------------------------------
# Test Database Configuration
# ---------------------------------------------------------

TEST_DATABASE_URL = (
    os.getenv("TEST_DATABASE_URL")
    or os.getenv("DATABASE_URL")
    or "postgresql+asyncpg://postgres:postgres@localhost:5432/ragdb_test"
)


# ---------------------------------------------------------
# Safety Check
# ---------------------------------------------------------
# Tests must use the dedicated test database on a local endpoint.
# This also supports GitHub Actions service containers accessed
# through localhost when the job runs directly on the runner.
# ---------------------------------------------------------

database_url = make_url(TEST_DATABASE_URL)

if database_url.get_backend_name() != "postgresql":
    raise RuntimeError("Unsafe test database configuration: PostgreSQL is required.")

if database_url.database != "ragdb_test":
    raise RuntimeError("Unsafe test database configuration: database must be 'ragdb_test'.")

allowed_test_hosts = {"localhost", "127.0.0.1", "::1"}
test_host = (database_url.host or "").lower().rstrip(".")

if test_host not in allowed_test_hosts:
    raise RuntimeError(
        "Unsafe test database configuration: host must be a local loopback address."
    )

if database_url.port not in (None, 5432):
    raise RuntimeError("Unsafe test database configuration: unexpected PostgreSQL port.")


# ---------------------------------------------------------
# Async Test Engine
# ---------------------------------------------------------
# NullPool prevents asyncpg connections from being reused
# across different pytest/TestClient event loops.
# ---------------------------------------------------------

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    poolclass=NullPool,
)


TestSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ---------------------------------------------------------
# Dependency Override
# ---------------------------------------------------------

async def override_get_db():
    async with TestSessionLocal() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


# ---------------------------------------------------------
# Synchronous Cleanup Engine
# ---------------------------------------------------------
# TestClient tests are synchronous, so database cleanup is
# intentionally performed through a synchronous SQLAlchemy
# engine.
# ---------------------------------------------------------

sync_database_url = database_url.set(
    drivername="postgresql+psycopg",
)

cleanup_engine = create_engine(
    sync_database_url,
    pool_pre_ping=True,
)


# ---------------------------------------------------------
# Test Database Reset
# ---------------------------------------------------------

def reset_test_database():
    with cleanup_engine.begin() as connection:
        connection.execute(
            text(
                """
                TRUNCATE TABLE
                    user_memories,
                    messages,
                    conversations,
                    document_versions,
                    documents,
                    users,
                    tenants
                RESTART IDENTITY CASCADE;
                """
            )
        )


@pytest.fixture(autouse=True)
def clean_test_database():
    """
    Start every test with a clean PostgreSQL test database.
    """

    reset_test_database()

    yield


# ---------------------------------------------------------
# FastAPI Test Client
# ---------------------------------------------------------

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client