from sqlalchemy.engine import make_url

from app.core.config import get_settings


def require_test_database() -> None:
    """Prevent standalone test scripts from using a non-test database."""
    database_url = make_url(get_settings().database_url)

    if database_url.database != "ragdb_test":
        raise RuntimeError(
            "Unsafe database configuration. "
            "Standalone test scripts must use 'ragdb_test'."
        )
