from dotenv import load_dotenv

# ---------------------------------------------------------
# Load environment variables before importing application
# modules that depend on them.
# ---------------------------------------------------------

load_dotenv()


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.conversations import router as conversations_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.rag import router as rag_router
from app.api.users import router as users_router
from app.core.config import get_settings

settings = get_settings()


app = FastAPI(
    title="Advanced RAG Platform",
    description="Production-oriented Multi-Tenant RAG Platform",
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------
# Origins are controlled through the CORS_ORIGINS
# environment variable.
#
# Example:
# CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
#
# For production, replace these with the actual frontend
# domain(s).
# ---------------------------------------------------------

cors_origins = [
    origin.strip()
    for origin in settings.cors_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS",
    ],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
    ],
)


# ---------------------------------------------------------
# API Routers
# ---------------------------------------------------------

app.include_router(
    health_router,
)

app.include_router(
    auth_router,
)

app.include_router(
    documents_router,
)

app.include_router(
    rag_router,
)

app.include_router(
    conversations_router,
)

app.include_router(
    users_router,
)


# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "Advanced RAG Platform is running"
    }