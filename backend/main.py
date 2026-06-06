from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import CORS_ORIGINS, FIGURES_DIR
from backend.routers import experiments, health, models, prediction, training_router


app = FastAPI(
    title="EEG ADHD Classifier API",
    description="API para clasificación ADHD/Control a partir de señales EEG.",
    version="0.1.0",
)

app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(prediction.router)
app.include_router(training_router.router, prefix="/training", tags=["training"])
app.include_router(experiments.router, tags=["experiments"])
