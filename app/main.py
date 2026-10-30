from fastapi import FastAPI
from app.routers import auth_router, health_router, projects_router

app = FastAPI(
    title="AI Text-to-Video Engine Backend",
    version="1.0.0",
    description="FastAPI service for Text-to-Video generation pipeline"
)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(projects_router)

@app.get("/")
async def root():
    return {"message": "Text-to-Video API is running"}
