from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.db.engine import SessionLocal
from backend.db.models import Experiment


def list_experiments(
    model_type: str | None = None,
    model_name: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """Devuelve los experimentos guardados del mas reciente al mas antiguo.

    Eager-carga el dataset asociado para evitar consultas adicionales al
    serializar el listado.
    """
    with SessionLocal() as session:
        stmt = (
            select(Experiment)
            .options(selectinload(Experiment.dataset))
            .order_by(Experiment.created_at.desc(), Experiment.id.desc())
            .offset(max(0, offset))
            .limit(max(1, min(limit, 200)))
        )
        if model_type:
            stmt = stmt.where(Experiment.model_type == model_type)
        if model_name:
            stmt = stmt.where(Experiment.model_name == model_name)

        return list(session.scalars(stmt).all())


def get_experiment(experiment_id: int):
    """Devuelve un experimento con su dataset y sus resultados por fold."""
    with SessionLocal() as session:
        return session.get(
            Experiment,
            experiment_id,
            options=[
                selectinload(Experiment.dataset),
                selectinload(Experiment.fold_results),
            ],
        )
