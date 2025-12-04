import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.routes import jobs, recommendations
from app.config import configure_logging, settings
from app.core.cache import cache_manager
from app.core.startup import trigger_job_ingestion
from app.database import engine

configure_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    if settings.JOB_INGEST_ON_STARTUP:
        await trigger_job_ingestion()

    yield

    logger.info("Shutting down...")
    await cache_manager.close()
    await engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    description="API for JobFinder application",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(recommendations.router)
app.include_router(jobs.router)


@app.get("/")
def read_root():
    return HTMLResponse(content="<h1>JobFinder API is running</h1>")
