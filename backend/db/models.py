from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    # Columna DateTime sin timezone -> guardamos UTC naive para evitar warnings.
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    """Base declarativa de SQLAlchemy."""


class Dataset(Base):
    """Dataset EEG subido, identificado por hash y metadatos resumidos."""

    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("dataset_hash", name="uq_datasets_hash"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    rows: Mapped[int] = mapped_column(Integer, nullable=False)
    columns: Mapped[int] = mapped_column(Integer, nullable=False)
    n_subjects: Mapped[int] = mapped_column(Integer, nullable=False)
    class_distribution: Mapped[dict[str, int]] = mapped_column(JSON, nullable=False)
    eeg_columns: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    experiments: Mapped[list["Experiment"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
    )


class Experiment(Base):
    """Entrenamiento guardado con configuracion, metricas agregadas e informes."""

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    evaluation_mode: Mapped[str] = mapped_column(String(255), nullable=False)
    training_time_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    balanced_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    precision: Mapped[float] = mapped_column(Float, nullable=False)
    recall: Mapped[float] = mapped_column(Float, nullable=False)
    f1_score: Mapped[float] = mapped_column(Float, nullable=False)
    eeg_params: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    model_params: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    training_params: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    confusion_matrix: Mapped[list[list[int]]] = mapped_column(JSON, nullable=False)
    classification_report: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    dataset: Mapped[Dataset] = relationship(back_populates="experiments")
    fold_results: Mapped[list["ExperimentFold"]] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan",
    )


class ExperimentFold(Base):
    """Metricas por fold generadas durante la evaluacion cross-subject."""

    __tablename__ = "experiment_folds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=False, index=True)
    fold: Mapped[int] = mapped_column(Integer, nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    balanced_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    precision: Mapped[float] = mapped_column(Float, nullable=False)
    recall: Mapped[float] = mapped_column(Float, nullable=False)
    f1_score: Mapped[float] = mapped_column(Float, nullable=False)
    n_train_subjects: Mapped[int | None] = mapped_column(Integer)
    n_val_subjects: Mapped[int | None] = mapped_column(Integer)
    n_test_subjects: Mapped[int | None] = mapped_column(Integer)
    best_threshold: Mapped[float | None] = mapped_column(Float)

    experiment: Mapped[Experiment] = relationship(back_populates="fold_results")
