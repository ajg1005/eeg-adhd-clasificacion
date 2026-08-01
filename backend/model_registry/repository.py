from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.db.engine import SessionLocal
from backend.db.models import Experiment, TrainedModel


def save_trained_model(experiment_id: int, record: dict[str, Any]) -> int:
    """Registra los metadatos de un modelo final entrenado para un experimento."""
    with SessionLocal() as session:
        trained_model = TrainedModel(
            experiment_id=experiment_id,
            model_type=str(record["model_type"]),
            model_name=str(record["model_name"]),
            model_family=str(record["model_family"]),
            artifact_path=str(record["artifact_path"]),
            feature_columns_path=record.get("feature_columns_path"),
            n_features=record.get("n_features"),
            n_epochs_training=int(record["n_epochs_training"]),
            n_subjects_training=int(record["n_subjects_training"]),
            file_size_bytes=record.get("file_size_bytes"),
            threshold=record.get("threshold"),
            model_metadata=record.get("model_metadata", {}),
            is_selected=bool(record.get("is_selected", False)),
        )
        session.add(trained_model)
        session.commit()
        return int(trained_model.id)


def get_trained_model_by_experiment(experiment_id: int):
    """Devuelve el modelo entrenado asociado a un experimento, si existe."""
    with SessionLocal() as session:
        return session.scalar(
            select(TrainedModel).where(TrainedModel.experiment_id == experiment_id)
        )


def list_trained_models(owner_id: int, limit: int = 100, offset: int = 0):
    """Lista los modelos registrados que pertenecen al usuario."""
    with SessionLocal() as session:
        stmt = (
            select(TrainedModel)
            .join(TrainedModel.experiment)
            .where(Experiment.owner_id == owner_id)
            .order_by(TrainedModel.created_at.desc(), TrainedModel.id.desc())
            .offset(max(0, offset))
            .limit(max(1, min(limit, 200)))
        )
        return list(session.scalars(stmt).all())


def list_trained_models_ranked(owner_id: int):
    """Ordena los modelos del usuario por el resultado de su experimento."""
    with SessionLocal() as session:
        stmt = (
            select(TrainedModel)
            .join(TrainedModel.experiment)
            .where(Experiment.owner_id == owner_id)
            .options(
                selectinload(TrainedModel.experiment).selectinload(Experiment.dataset)
            )
            .order_by(
                Experiment.balanced_accuracy.desc(),
                Experiment.f1_score.desc(),
                Experiment.created_at.desc(),
                TrainedModel.id.desc(),
            )
        )
        return list(session.scalars(stmt).all())


def get_trained_model(trained_model_id: int, owner_id: int):
    """Devuelve un modelo entrenado si pertenece al usuario."""
    with SessionLocal() as session:
        stmt = (
            select(TrainedModel)
            .join(TrainedModel.experiment)
            .where(
                TrainedModel.id == trained_model_id,
                Experiment.owner_id == owner_id,
            )
        )
        return session.scalar(stmt)
