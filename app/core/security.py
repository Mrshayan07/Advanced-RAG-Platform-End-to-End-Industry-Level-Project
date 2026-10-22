from datetime import datetime, timedelta, UTC
from uuid import UUID

import jwt
from pwdlib import PasswordHash

from app.core.config import get_settings

settings = get_settings()

JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60

password_hash = PasswordHash.recommended()


def create_access_token(
    user_id: UUID,
    tenant_id: UUID,
    role: str,
) -> str:
    expire = datetime.now(UTC) + timedelta(
        minutes=JWT_EXPIRE_MINUTES
    )

    payload = {
        "sub": str(user_id),
        "tenant_id": str(tenant_id),
        "role": role,
        "exp": expire,
    }

    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> dict:
    return jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[JWT_ALGORITHM],
    )


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    return password_hash.verify(
        plain_password,
        hashed_password,
    )