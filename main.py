import uvicorn
from fastapi import FastAPI
from api import router as audio_router
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Kokoro TTS Audio API Service",
    description="Provides Text-to-Speech generation using the Kokoro library.",
    version="0.1.0",
)

# Include the router from api.py
app.include_router(audio_router, prefix="/api", tags=["audio"])

@app.get("/", tags=["root"])
async def read_root():
    """ Basic root endpoint to confirm the server is running. """
    return {"message": "Welcome to the Kokoro TTS Audio API Service. See /docs for API details."}

if __name__ == "__main__":
    logger.info("Starting Kokoro TTS Audio API Service...")
    # Use host="0.0.0.0" to make it accessible on the network
    # Use reload=True for development to automatically reload on code changes
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
