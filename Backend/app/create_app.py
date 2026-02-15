from fastapi import FastAPI
from app.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="KinetoCheck API",
        version="1.0.0",
        description="Movement correctness analysis using Spatial-Temporal Graph Attention",
    )
    app.include_router(router)
    return app