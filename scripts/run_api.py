import datetime
import logging
import os
import sys
from pathlib import Path
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

from settings import settings
from src.api.athena_api import athena_router
from src.api.dashboard_router import dashboard_router
from src.api.ws_router import ws_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


# Initialize FastAPI app
app = FastAPI(
    title="Lumos Trade API",
    description="Athena market intelligence and trading API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(athena_router)
app.include_router(dashboard_router)
app.include_router(ws_router)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "timestamp": str(Path(__file__).stat().st_mtime),
        "api_only_mode": os.environ.get("API_ONLY_MODE", "false"),
        "version": "1.0.0",
        "build_time": str(datetime.datetime.now())
    }

def main():
    host = settings.api_host
    port = settings.api_port

    # Display startup information
    auth_token = settings.deriv_auth_token
    if auth_token:
        logger.info(f" ✅ DERIV_AUTH_TOKEN loaded: {auth_token[:10]}...")
    else:
        logger.warning(" ❌ DERIV_AUTH_TOKEN not set! Some features may not work properly")
    
    logger.info(f"Starting API server at http://{host}:{port}")
    logger.info(f"API documentation available at http://{host}:{port}/docs")
    
    # Start Uvicorn server
    uvicorn.run(
        "run_api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()