from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.health import router as health_router
from backend.api.tasks import router as tasks_router
from backend.core.config import CORS_ORIGINS, FIGURES_DIR
from backend.datasets.router import router as datasets_router
from backend.experiments.router import router as experiments_router
from backend.inference.router import router as inference_router
from backend.model_registry.router import router as model_registry_router
from backend.training.router import router as training_router


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

app.include_router(health_router)
app.include_router(tasks_router)
app.include_router(model_registry_router)
app.include_router(inference_router)
app.include_router(training_router)
app.include_router(experiments_router, tags=["experiments"])
app.include_router(datasets_router)
