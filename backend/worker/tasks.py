from backend.worker.celery_app import celery_app


@celery_app.task(name="system.ping")
def ping() -> dict[str, str]:
    """Comprueba que el broker entrega tareas al worker."""
    return {"status": "ok"}
