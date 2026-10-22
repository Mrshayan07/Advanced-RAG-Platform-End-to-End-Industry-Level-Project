import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

STORAGE_DIR = Path("storage/documents")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


async def save_upload(
    file: UploadFile,
) -> tuple[str, str]:
    content = await file.read()

    checksum = hashlib.sha256(content).hexdigest()

    extension = Path(file.filename or "").suffix
    filename = f"{uuid4()}{extension}"

    file_path = STORAGE_DIR / filename

    file_path.write_bytes(content)

    return str(file_path), checksum